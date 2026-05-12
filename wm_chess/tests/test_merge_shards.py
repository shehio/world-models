"""Tests for wm_chess/scripts/merge_shards.py.

merge_shards concatenates N shard datasets (each a library leaf with
data.npz + games.pgn + metadata.json) into one merged dataset. This
runs on the laptop after the AWS fleet ships shards back — a bug here
would lose the entire run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_shards  # noqa: E402


def _make_shard(root: Path, depth: int, mpv: int, T: float, games: int, seed: int,
                n_positions: int, K: int = 8):
    """Build a fake shard directory with data.npz, games.pgn, metadata.json."""
    T_str = f"{float(T):g}"
    leaf = (root / f"sf-18" / f"d{depth}-mpv{mpv}-T{T_str}"
            / f"g{games}-seed{seed}")
    leaf.mkdir(parents=True, exist_ok=True)
    # Synthetic tensors (deterministic so we can verify concatenation order).
    states = np.full((n_positions, 19, 8, 8), float(seed), dtype=np.float32)
    moves = np.arange(n_positions, dtype=np.int64) + seed * 100_000
    mpv_idx = np.zeros((n_positions, K), dtype=np.int64)
    mpv_logp = np.full((n_positions, K), -np.log(K), dtype=np.float32)
    zs = np.zeros(n_positions, dtype=np.float32)
    np.savez_compressed(
        leaf / "data.npz",
        states=states, moves=moves,
        multipv_indices=mpv_idx, multipv_logprobs=mpv_logp,
        zs=zs, K=np.array(K, dtype=np.int32),
        temperature_pawns=np.array(float(T), dtype=np.float32),
    )
    (leaf / "games.pgn").write_text(
        f'[Event "shard seed={seed}"]\n[Result "1-0"]\n1. e4 e5 1-0\n\n'
    )
    (leaf / "metadata.json").write_text(json.dumps({
        "sf_version": "18", "depth": depth, "multipv": mpv,
        "temperature_pawns": float(T), "seed": seed,
        "n_positions": n_positions, "wall_seconds": 99.0,
    }))
    return leaf


class TestMergeRoundtrip:
    def test_basic_two_shards(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 42, n_positions=30)
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 1042, n_positions=20)
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()

        # Merged data.npz
        d = np.load(out / "data.npz")
        assert d["states"].shape == (50, 19, 8, 8)
        assert d["moves"].shape == (50,)
        assert d["multipv_indices"].shape == (50, 8)
        # Shards concatenate in sorted(glob) order — seed=1042 sorts before seed=42.
        # Both come back; order matters only for downstream deterministic ablations.
        seed_values = sorted({float(d["states"][i, 0, 0, 0]) for i in range(50)})
        assert seed_values == [42.0, 1042.0]
        assert int(d["K"]) == 8
        assert float(d["temperature_pawns"]) == 1.0

        # Merged PGN
        pgn = (out / "games.pgn").read_text()
        assert "shard seed=42" in pgn
        assert "shard seed=1042" in pgn
        assert pgn.count('[Event "') == 2

        # Merged metadata
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["n_shards"] == 2
        assert meta["n_positions"] == 50
        assert meta["n_games"] == 2
        assert meta["multipv"] == 8
        assert meta["temperature_pawns"] == 1.0
        assert len(meta["merged_from_shards"]) == 2

    def test_eight_shards_like_real_run(self, tmp_path, monkeypatch):
        """Realistic scenario: 8 VMs × 12,500 games = 100K total."""
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        for i in range(8):
            seed = 42 + i * 1000
            _make_shard(tmp_path, 15, 8, 1.0, 12500, seed, n_positions=100)
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()
        d = np.load(out / "data.npz")
        assert d["states"].shape[0] == 800
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["n_shards"] == 8
        assert len(meta["merged_from_shards"]) == 8


class TestMergeEdgeCases:
    def test_no_shards_matches_glob_errors(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(tmp_path / "no-such/*"),
            "--output", str(tmp_path / "out"),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        with pytest.raises(SystemExit, match="no complete shards"):
            merge_shards.main()

    def test_partial_shard_without_data_npz_is_skipped(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        # One complete shard.
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 42, n_positions=10)
        # One partial: directory exists but no data.npz (just empty chunks/).
        partial = cfg_dir / "g12500-seed1042"
        (partial / "chunks").mkdir(parents=True, exist_ok=True)
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()
        d = np.load(out / "data.npz")
        assert d["states"].shape[0] == 10
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["n_shards"] == 1

    def test_missing_pgn_handled(self, tmp_path, monkeypatch):
        """A shard with data.npz but no games.pgn shouldn't crash the merge."""
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        leaf = _make_shard(tmp_path, 15, 8, 1.0, 12500, 42, n_positions=10)
        (leaf / "games.pgn").unlink()
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()
        d = np.load(out / "data.npz")
        assert d["states"].shape[0] == 10
        # PGN file exists but is empty (or near-empty).
        assert (out / "games.pgn").exists()


class TestMergedShapes:
    def test_array_dtypes_preserved(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 42, n_positions=5)
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 1042, n_positions=5)
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()
        d = np.load(out / "data.npz")
        assert d["states"].dtype == np.float32
        assert d["moves"].dtype == np.int64
        assert d["multipv_indices"].dtype == np.int64
        assert d["multipv_logprobs"].dtype == np.float32
        assert d["zs"].dtype == np.float32

    def test_K_metadata_consistent(self, tmp_path, monkeypatch):
        cfg_dir = tmp_path / "sf-18" / "d15-mpv8-T1"
        _make_shard(tmp_path, 15, 8, 1.0, 12500, 42, n_positions=5, K=8)
        out = tmp_path / "merged"
        monkeypatch.setattr("sys.argv", [
            "merge_shards.py",
            "--shards-glob", str(cfg_dir / "g12500-seed*"),
            "--output", str(out),
            "--multipv", "8",
            "--temperature", "1.0",
        ])
        merge_shards.main()
        d = np.load(out / "data.npz")
        assert d["multipv_indices"].shape == (5, 8)
        assert d["multipv_logprobs"].shape == (5, 8)
