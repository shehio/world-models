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

from azdistill_scaled.board import N_INPUT_PLANES
from azdistill_scaled.stockfish_data import (
    _SF_HEADER_RE,
    _append_worker_pgn,
    _atomic_npz_save,
    _save_worker_chunk,
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
