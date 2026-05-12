"""Unit tests for crash-safe Stockfish data generation.

Run from 02c-distill-scaled root:
    uv run python -m pytest tests/test_data_generation.py -v

Or directly:
    uv run python tests/test_data_generation.py
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import chess
import chess.engine

from azdistill_scaled.board import N_INPUT_PLANES
from azdistill_scaled.stockfish_data import (
    MATE_CP,
    _SF_HEADER_RE,
    _append_worker_pgn,
    _atomic_npz_save,
    _board_to_pgn,
    _multipv_to_distribution,
    _save_worker_chunk,
    _score_to_cp,
    _stockfish_path_or_die,
    detect_stockfish_version,
    finalize_library_path,
    library_dataset_path,
)


# ---------------------------------------------------------------------------
# Pure functions — no engine required
# ---------------------------------------------------------------------------


class TestVersionRegex:
    def test_matches_sf14(self):
        m = _SF_HEADER_RE.search("id name Stockfish 14.1")
        assert m and m.group(1) == "14.1"

    def test_matches_sf18(self):
        m = _SF_HEADER_RE.search("id name Stockfish 18")
        assert m and m.group(1) == "18"

    def test_matches_dev_build(self):
        m = _SF_HEADER_RE.search("id name Stockfish dev-20250101")
        assert m and m.group(1) == "dev-20250101"

    def test_case_insensitive(self):
        m = _SF_HEADER_RE.search("STOCKFISH 16.1")
        assert m and m.group(1) == "16.1"

    def test_no_match(self):
        assert _SF_HEADER_RE.search("not the engine banner") is None


class TestLibraryPath:
    def test_basic(self):
        p = library_dataset_path("data/lib", "18", 15, 8, 1.0, 4000, 42)
        assert p == "data/lib/sf-18/d15-mpv8-T1/g4000-seed42"

    def test_temperature_fraction(self):
        p = library_dataset_path("data/lib", "14.1", 10, 8, 0.3, 1000, 0)
        assert p == "data/lib/sf-14.1/d10-mpv8-T0.3/g1000-seed0"

    def test_deterministic(self):
        a = library_dataset_path("/r", "18", 12, 4, 1.0, 100, 7)
        b = library_dataset_path("/r", "18", 12, 4, 1.0, 100, 7)
        assert a == b


class TestAtomicSave:
    def test_writes_and_reads_back(self, tmp_path):
        p = tmp_path / "x.npz"
        _atomic_npz_save(str(p), a=np.array([1, 2, 3]), b=np.array([[1.0]]))
        d = np.load(str(p))
        assert np.array_equal(d["a"], np.array([1, 2, 3]))
        assert np.allclose(d["b"], np.array([[1.0]]))

    def test_no_tmp_left_behind(self, tmp_path):
        p = tmp_path / "x.npz"
        _atomic_npz_save(str(p), a=np.array([1]))
        assert not (tmp_path / "x.npz.tmp").exists()


# ---------------------------------------------------------------------------
# Chunked save + library finalize — the crash-recovery path
# ---------------------------------------------------------------------------


def _fake_chunk(multipv=8, n=3):
    """Build a tiny synthetic chunk's worth of arrays."""
    states = [np.zeros((N_INPUT_PLANES, 8, 8), dtype=np.float32) for _ in range(n)]
    moves = [i for i in range(n)]
    mpv_idx = [[i + j for j in range(multipv)] for i in range(n)]
    mpv_logp = [np.full((multipv,), -1.0, dtype=np.float32) for _ in range(n)]
    zs = [0.0] * n
    return states, moves, mpv_idx, mpv_logp, zs


class TestChunkRoundtrip:
    def test_save_worker_chunk_roundtrip(self, tmp_path):
        s, m, i, lp, z = _fake_chunk(multipv=8, n=4)
        path = _save_worker_chunk(str(tmp_path), worker_id=0, chunk_idx=0,
                                  states=s, moves=m, mpv_idx=i,
                                  mpv_logp=lp, zs=z, multipv=8)
        d = np.load(path)
        assert d["states"].shape == (4, N_INPUT_PLANES, 8, 8)
        assert d["moves"].shape == (4,)
        assert d["multipv_indices"].shape == (4, 8)
        assert d["multipv_logprobs"].shape == (4, 8)
        assert d["zs"].shape == (4,)

    def test_save_empty_chunk(self, tmp_path):
        path = _save_worker_chunk(str(tmp_path), worker_id=1, chunk_idx=0,
                                  states=[], moves=[], mpv_idx=[],
                                  mpv_logp=[], zs=[], multipv=8)
        d = np.load(path)
        assert d["states"].shape == (0, N_INPUT_PLANES, 8, 8)

    def test_append_pgn(self, tmp_path):
        _append_worker_pgn(str(tmp_path), 3, '[Event "A"]\n1. e4 e5 1-0')
        _append_worker_pgn(str(tmp_path), 3, '[Event "B"]\n1. d4 d5 0-1')
        text = (tmp_path / "worker_03.pgn").read_text()
        assert text.count('[Event "') == 2
        assert "1. e4" in text
        assert "1. d4" in text


class TestFinalize:
    def test_finalize_assembles_chunks(self, tmp_path):
        lib_path = tmp_path / "lib_run"
        chunks_dir = lib_path / "chunks"
        chunks_dir.mkdir(parents=True)

        # Two workers, two chunks each = 4 chunks of 3 games each = 12 games
        for wid in (0, 1):
            for cid in (0, 1):
                s, m, i, lp, z = _fake_chunk(multipv=4, n=3)
                _save_worker_chunk(str(chunks_dir), wid, cid, s, m, i, lp, z,
                                   multipv=4)
                _append_worker_pgn(str(chunks_dir), wid,
                                   f'[Event "w{wid}c{cid}"]\n1. e4 1-0')

        fin = finalize_library_path(str(lib_path), multipv=4,
                                    temperature_pawns=1.0)

        assert fin["used_chunks"] == 4
        assert fin["n_positions"] == 4 * 3
        assert os.path.exists(fin["data_npz"])
        assert os.path.exists(fin["games_pgn"])

        d = np.load(fin["data_npz"])
        assert d["states"].shape == (12, N_INPUT_PLANES, 8, 8)
        assert int(d["K"]) == 4
        assert float(d["temperature_pawns"]) == 1.0

    def test_finalize_partial_chunks_simulates_crash(self, tmp_path):
        """Crash recovery: only 1 worker wrote any chunks. finalize should
        still produce a valid (smaller) data.npz."""
        lib_path = tmp_path / "lib_partial"
        chunks_dir = lib_path / "chunks"
        chunks_dir.mkdir(parents=True)

        s, m, i, lp, z = _fake_chunk(multipv=8, n=2)
        _save_worker_chunk(str(chunks_dir), 0, 0, s, m, i, lp, z, multipv=8)

        fin = finalize_library_path(str(lib_path), multipv=8,
                                    temperature_pawns=1.0)
        assert fin["n_positions"] == 2
        d = np.load(fin["data_npz"])
        assert d["states"].shape == (2, N_INPUT_PLANES, 8, 8)

    def test_finalize_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            finalize_library_path(str(tmp_path / "no_such"), multipv=8,
                                  temperature_pawns=1.0)


# ---------------------------------------------------------------------------
# Score → cp conversion + multipv → distribution
# ---------------------------------------------------------------------------


class TestScoreToCp:
    def test_centipawn_score(self):
        sc = chess.engine.PovScore(chess.engine.Cp(123), chess.WHITE)
        assert _score_to_cp(sc) == 123.0

    def test_negative_centipawn(self):
        sc = chess.engine.PovScore(chess.engine.Cp(-45), chess.WHITE)
        assert _score_to_cp(sc) == -45.0

    def test_mate_for_is_large_positive(self):
        # Mate in 3 from STM POV → close to +MATE_CP (python-chess uses
        # mate_score - distance, so Mate(+3) → 997 when mate_score=1000).
        sc = chess.engine.PovScore(chess.engine.Mate(3), chess.WHITE)
        val = _score_to_cp(sc)
        assert MATE_CP - 10 <= val <= MATE_CP

    def test_mate_against_is_large_negative(self):
        sc = chess.engine.PovScore(chess.engine.Mate(-2), chess.WHITE)
        val = _score_to_cp(sc)
        assert -MATE_CP <= val <= -MATE_CP + 10


class TestMultipvToDistribution:
    def _info(self, score_cp: int, move: chess.Move) -> dict:
        return {"score": chess.engine.PovScore(chess.engine.Cp(score_cp),
                                                chess.WHITE),
                "pv": [move]}

    def test_padding_when_fewer_than_K(self):
        b = chess.Board()
        moves = list(b.legal_moves)[:3]
        info_list = [self._info(50 - i * 20, m) for i, m in enumerate(moves)]
        idx, logp, n_valid = _multipv_to_distribution(
            info_list, b, temperature_pawns=1.0, K=8,
        )
        assert n_valid == 3
        assert len(idx) == 8
        assert idx[3:] == [-1] * 5
        # Last K-n_valid slots are -inf.
        assert np.isinf(logp[3:]).all()

    def test_distribution_sums_to_one(self):
        b = chess.Board()
        moves = list(b.legal_moves)[:4]
        info_list = [self._info(50 - i * 30, m) for i, m in enumerate(moves)]
        idx, logp, n_valid = _multipv_to_distribution(
            info_list, b, temperature_pawns=1.0, K=4,
        )
        probs = np.exp(logp[:n_valid])
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_temperature_sharper_concentrates(self):
        """T=0.3 should put more mass on the best move than T=1.0 for the
        same cp gaps."""
        b = chess.Board()
        moves = list(b.legal_moves)[:3]
        info_list = [self._info(50 - i * 30, m) for i, m in enumerate(moves)]
        _, logp_smooth, _ = _multipv_to_distribution(
            info_list, b, temperature_pawns=1.0, K=3,
        )
        _, logp_sharp, _ = _multipv_to_distribution(
            info_list, b, temperature_pawns=0.3, K=3,
        )
        p_smooth_top = float(np.exp(logp_smooth[0]))
        p_sharp_top = float(np.exp(logp_sharp[0]))
        assert p_sharp_top > p_smooth_top

    def test_empty_info_list(self):
        """Defensive: pathological "no PVs at all". Should not crash."""
        b = chess.Board()
        idx, logp, n_valid = _multipv_to_distribution(
            [], b, temperature_pawns=1.0, K=4,
        )
        assert n_valid == 0
        assert idx == [-1] * 4
        assert np.isinf(logp).all()

    def test_illegal_pv_dropped(self):
        """If Stockfish hands us a PV whose first move isn't legal, drop it."""
        b = chess.Board()
        # Construct an illegal move (knight to its own square).
        illegal = chess.Move(chess.E2, chess.E2)
        legal = list(b.legal_moves)[0]
        info_list = [self._info(20, illegal), self._info(10, legal)]
        idx, logp, n_valid = _multipv_to_distribution(
            info_list, b, temperature_pawns=1.0, K=4,
        )
        assert n_valid == 1  # illegal dropped, only the legal one survives


# ---------------------------------------------------------------------------
# PGN rendering
# ---------------------------------------------------------------------------


class TestBoardToPgn:
    def test_writes_headers_and_result(self):
        b = chess.Board()
        moves = [chess.Move.from_uci("e2e4"),
                 chess.Move.from_uci("e7e5"),
                 chess.Move.from_uci("g1f3")]
        pgn = _board_to_pgn(b, moves, outcome_white_pov=1.0,
                            headers={"Event": "smoke test",
                                     "White": "A", "Black": "B"})
        assert '[Event "smoke test"]' in pgn
        assert '[Result "1-0"]' in pgn
        assert "1. e4 e5" in pgn

    def test_draw_outcome(self):
        pgn = _board_to_pgn(chess.Board(), [], outcome_white_pov=0.0,
                            headers={"Event": "draw"})
        assert '[Result "1/2-1/2"]' in pgn

    def test_loss_outcome(self):
        pgn = _board_to_pgn(chess.Board(), [], outcome_white_pov=-1.0,
                            headers={"Event": "loss"})
        assert '[Result "0-1"]' in pgn

    def test_stops_at_illegal_move(self):
        """If the move list goes bad partway through, the PGN should still
        be syntactically valid up to the break point."""
        import chess.pgn as _pgn  # local-only; don't shadow outer `chess`.
        b = chess.Board()
        moves = [chess.Move.from_uci("e2e4"),
                 chess.Move.from_uci("g1f6")]  # nonsense knight jump
        pgn_text = _board_to_pgn(b, moves, outcome_white_pov=0.0,
                                 headers={"Event": "partial"})
        game = _pgn.read_game(io.StringIO(pgn_text))
        assert game is not None
        played = list(game.mainline_moves())
        assert played[0].uci() == "e2e4"


# ---------------------------------------------------------------------------
# Stockfish path resolver
# ---------------------------------------------------------------------------


class TestStockfishPath:
    def test_explicit_path_returned(self, tmp_path):
        # Make a fake "binary" file — _stockfish_path_or_die doesn't
        # validate executability, just existence-via-shutil-which path.
        # When given an explicit path, it should return it as-is.
        fake = tmp_path / "stockfish"
        fake.write_text("#!/bin/sh\n")
        result = _stockfish_path_or_die(str(fake))
        assert result == str(fake)

    def test_none_finds_on_path_or_dies(self):
        # If `stockfish` is on PATH (it is in dev), returns its path.
        # If not, raises FileNotFoundError.
        import shutil
        on_path = shutil.which("stockfish")
        if on_path:
            assert _stockfish_path_or_die(None) == on_path
        else:
            with pytest.raises(FileNotFoundError):
                _stockfish_path_or_die(None)


# ---------------------------------------------------------------------------
# Live-engine smoke (skipped if stockfish not installed)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("stockfish") is None,
                    reason="stockfish binary not on PATH")
class TestLiveEngine:
    def test_detect_stockfish_version(self):
        v = detect_stockfish_version()
        # Some version string, not literal "unknown".
        assert v != "unknown"
        assert len(v) > 0

    def test_detect_returns_unknown_for_bad_path(self):
        v = detect_stockfish_version("/nonexistent/stockfish")
        assert v == "unknown"

    def test_play_one_game_produces_valid_pgn(self):
        """End-to-end: run one short game, parse the PGN back, check it
        replays cleanly."""
        import chess.engine
        import chess.pgn
        import random as _random
        from azdistill_scaled.stockfish_data import play_one_game

        engine = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish"))
        try:
            states, played, mpv_idx, mpv_logp, zs, stats, pgn_text = play_one_game(
                engine, depth=4, max_plies=20, random_opening_plies=2,
                multipv=4, temperature_pawns=1.0, rng=_random.Random(0),
            )
        finally:
            engine.quit()

        assert len(states) > 0
        assert len(states) == len(played) == len(mpv_idx) == len(zs)
        # PGN must reparse and have headers.
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        assert game is not None
        assert game.headers.get("Result") in {"1-0", "0-1", "1/2-1/2"}
        # Replay all moves cleanly.
        board = game.board()
        for mv in game.mainline_moves():
            assert mv in board.legal_moves
            board.push(mv)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
