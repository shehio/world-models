"""Tests for wm_chess/scripts/merge_chunks.py.

merge_chunks is the streaming alternative to (generate_data --finalize-only
+ merge_shards), built for large pods where the in-memory finalize+merge
path OOMs. Peak memory should stay bounded by one chunk regardless of
total chunk count, so the tests deliberately use many small chunks.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import merge_chunks  # noqa: E402


def _make_chunk(
    root: Path,
    pod: int,
    worker: int,
    chunk_idx: int,
    n_positions: int,
    K: int = 8,
    sentinel: float | None = None,
    depth: int = 15,
    mpv: int = 8,
    T: float = 1.0,
    games: int = 12500,
    seed_override: int | None = None,
):
    """Write one chunk npz at the path layout merge_chunks expects to glob.

    Layout: <root>/pod-<P>/sf-15.1/d<D>-mpv<K>-T<T>/g<N>-seed<S>/chunks/worker_<WW>_chunk_<CCCC>.npz
    """
    seed = seed_override if seed_override is not None else 42 + pod * 1000
    T_str = f"{float(T):g}"
    chunks_dir = (root / f"pod-{pod}" / "sf-15.1" / f"d{depth}-mpv{mpv}-T{T_str}"
                  / f"g{games}-seed{seed}" / "chunks")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    # If sentinel is given, fill states with that value so we can identify
    # which chunk's data ended up where after the merge.
    fill = float(sentinel) if sentinel is not None else float(pod * 100 + worker)
    states = np.full((n_positions, 19, 8, 8), fill, dtype=np.float32)
    moves = np.arange(n_positions, dtype=np.int64) + pod * 1_000_000 + worker * 1000
    mpv_idx = np.zeros((n_positions, K), dtype=np.int64)
    mpv_logp = np.full((n_positions, K), -np.log(K), dtype=np.float32)
    zs = np.zeros(n_positions, dtype=np.float32)
    np.savez_compressed(
        chunks_dir / f"worker_{worker:02d}_chunk_{chunk_idx:04d}.npz",
        states=states,
        moves=moves,
        multipv_indices=mpv_idx,
        multipv_logprobs=mpv_logp,
        zs=zs,
    )
    # Worker PGN (only one per worker, regardless of chunk count).
    pgn_path = chunks_dir / f"worker_{worker:02d}.pgn"
    if not pgn_path.exists():
        pgn_path.write_text(
            f'[Event "pod={pod} worker={worker}"]\n[Result "1-0"]\n1. e4 e5 1-0\n\n'
        )
    return chunks_dir


def _run_merge(root: Path, out: Path, monkeypatch, K: int = 8, T: float = 1.0):
    monkeypatch.setattr("sys.argv", [
        "merge_chunks.py",
        "--shards-root", str(root),
        "--output", str(out),
        "--multipv", str(K),
        "--temperature", str(T),
    ])
    merge_chunks.main()


class TestBasicMerge:
    def test_two_pods_one_chunk_each(self, tmp_path, monkeypatch):
        _make_chunk(tmp_path, pod=0, worker=0, chunk_idx=0, n_positions=10)
        _make_chunk(tmp_path, pod=1, worker=0, chunk_idx=0, n_positions=15)
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch)

        d = np.load(out / "data.npz")
        assert d["states"].shape == (25, 19, 8, 8)
        assert d["moves"].shape == (25,)
        assert d["multipv_indices"].shape == (25, 8)
        assert d["multipv_logprobs"].shape == (25, 8)
        assert d["zs"].shape == (25,)
        assert int(d["K"]) == 8
        assert float(d["temperature_pawns"]) == 1.0

    def test_many_small_chunks_concatenate_correctly(self, tmp_path, monkeypatch):
        """20 chunks of 5 positions each → 100 positions total, in order."""
        for chunk_idx in range(20):
            _make_chunk(tmp_path, pod=0, worker=0, chunk_idx=chunk_idx,
                        n_positions=5, sentinel=float(chunk_idx))
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch)

        d = np.load(out / "data.npz")
        assert d["states"].shape[0] == 100
        # First 5 rows came from chunk 0; next 5 from chunk 1; ...
        for i in range(20):
            block = d["states"][i*5:(i+1)*5, 0, 0, 0]
            assert np.all(block == float(i)), \
                f"chunk {i} block has wrong sentinel: got {block[0]}"


class TestSortKey:
    def test_pod_then_worker_then_chunk_ordering(self):
        paths = [
            "/x/pod-2/.../chunks/worker_03_chunk_0001.npz",
            "/x/pod-0/.../chunks/worker_10_chunk_0000.npz",
            "/x/pod-0/.../chunks/worker_01_chunk_0002.npz",
            "/x/pod-0/.../chunks/worker_01_chunk_0000.npz",
            "/x/pod-1/.../chunks/worker_00_chunk_0000.npz",
        ]
        sorted_paths = sorted(paths, key=merge_chunks.chunk_sort_key)
        # Expected: pod-0 worker-01 chunk-0000, chunk-0002, then pod-0 worker-10,
        # then pod-1, then pod-2.
        assert sorted_paths == [
            "/x/pod-0/.../chunks/worker_01_chunk_0000.npz",
            "/x/pod-0/.../chunks/worker_01_chunk_0002.npz",
            "/x/pod-0/.../chunks/worker_10_chunk_0000.npz",
            "/x/pod-1/.../chunks/worker_00_chunk_0000.npz",
            "/x/pod-2/.../chunks/worker_03_chunk_0001.npz",
        ]


class TestDtypesPreserved:
    def test_array_dtypes_roundtrip(self, tmp_path, monkeypatch):
        _make_chunk(tmp_path, pod=0, worker=0, chunk_idx=0, n_positions=5)
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch)
        d = np.load(out / "data.npz")
        assert d["states"].dtype == np.float32
        assert d["moves"].dtype == np.int64
        assert d["multipv_indices"].dtype == np.int64
        assert d["multipv_logprobs"].dtype == np.float32
        assert d["zs"].dtype == np.float32
        assert d["K"].dtype == np.int32
        assert d["temperature_pawns"].dtype == np.float32


class TestPGNConcat:
    def test_pgn_contents_from_every_worker(self, tmp_path, monkeypatch):
        _make_chunk(tmp_path, pod=0, worker=0, chunk_idx=0, n_positions=5)
        _make_chunk(tmp_path, pod=0, worker=1, chunk_idx=0, n_positions=5)
        _make_chunk(tmp_path, pod=1, worker=0, chunk_idx=0, n_positions=5)
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch)
        pgn = (out / "games.pgn").read_text()
        for marker in ("pod=0 worker=0", "pod=0 worker=1", "pod=1 worker=0"):
            assert marker in pgn, f"missing {marker} in merged PGN"
        assert pgn.count('[Event "') == 3


class TestMetadata:
    def test_metadata_counts_and_settings(self, tmp_path, monkeypatch):
        # Two pods × 3 chunks each = 6 chunks total, 8 positions per chunk.
        for pod in (0, 1):
            for c in range(3):
                _make_chunk(tmp_path, pod=pod, worker=0, chunk_idx=c, n_positions=8)
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch, T=0.5)
        meta = json.loads((out / "metadata.json").read_text())
        assert meta["n_chunks"] == 6
        assert meta["n_pods"] == 2
        assert meta["n_positions"] == 48
        assert meta["n_games"] == 2  # one PGN per worker, one worker per pod
        assert meta["multipv"] == 8
        assert meta["temperature_pawns"] == 0.5
        assert "merge_ts" in meta
        assert "streaming" in meta["merge_tool"]


class TestEdgeCases:
    def test_no_chunks_exits(self, tmp_path, monkeypatch):
        out = tmp_path / "merged"
        with pytest.raises(SystemExit, match="no chunks"):
            _run_merge(tmp_path, out, monkeypatch)

    def test_shape_mismatch_across_chunks_raises(self, tmp_path, monkeypatch):
        # First chunk has inner shape (19,8,8); second has (20,8,8).
        chunks_dir = _make_chunk(tmp_path, pod=0, worker=0, chunk_idx=0,
                                 n_positions=5)
        bad = np.zeros((5, 20, 8, 8), dtype=np.float32)
        np.savez_compressed(
            chunks_dir / "worker_00_chunk_0001.npz",
            states=bad,
            moves=np.arange(5, dtype=np.int64),
            multipv_indices=np.zeros((5, 8), dtype=np.int64),
            multipv_logprobs=np.zeros((5, 8), dtype=np.float32),
            zs=np.zeros(5, dtype=np.float32),
        )
        out = tmp_path / "merged"
        with pytest.raises(RuntimeError, match="shape/dtype mismatch"):
            _run_merge(tmp_path, out, monkeypatch)


class TestRoundTrip:
    def test_output_loads_with_np_load(self, tmp_path, monkeypatch):
        """Drop-in compatibility: trainer expects np.load() of one file."""
        for pod in range(3):
            for w in range(2):
                _make_chunk(tmp_path, pod=pod, worker=w, chunk_idx=0,
                            n_positions=4)
        out = tmp_path / "merged"
        _run_merge(tmp_path, out, monkeypatch)

        with np.load(out / "data.npz") as d:
            keys = set(d.files)
        # train.py's loader expects exactly these (plus the scalars):
        expected = {"states", "moves", "multipv_indices", "multipv_logprobs",
                    "zs", "K", "temperature_pawns"}
        assert expected.issubset(keys), f"missing keys: {expected - keys}"
