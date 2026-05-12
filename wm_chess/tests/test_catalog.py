"""Tests for the library catalog generator at wm_chess/scripts/catalog.py.

These don't need stockfish or torch — pure file-system + markdown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import catalog  # noqa: E402


def _make_dataset(root: Path, sf: str, depth: int, mpv: int, T,
                  games: int, seed: int, *,
                  complete: bool = True,
                  partial_chunks: int = 0,
                  positions: int = 100):
    """Build a fake library entry on disk for testing."""
    T_str = f"{float(T):g}"
    leaf = (root / "games" / f"sf-{sf}" / f"d{depth}-mpv{mpv}-T{T_str}"
            / f"g{games}-seed{seed}")
    leaf.mkdir(parents=True, exist_ok=True)
    chunks = leaf / "chunks"
    chunks.mkdir(exist_ok=True)
    for i in range(partial_chunks):
        (chunks / f"worker_00_chunk_{i:04d}.npz").write_bytes(b"\0" * 1024)
    if complete:
        (leaf / "data.npz").write_bytes(b"\0" * 4096)
        (leaf / "metadata.json").write_text(json.dumps({
            "sf_version": sf,
            "depth": depth,
            "multipv": mpv,
            "temperature_pawns": float(T),
            "n_positions": positions,
            "wall_seconds": 1234.5,
        }))
    return leaf


class TestDiscoverDatasets:
    def test_empty_library(self, tmp_path):
        assert catalog.discover_datasets(str(tmp_path)) == []

    def test_finds_complete_dataset(self, tmp_path):
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42, positions=243_000)
        entries = catalog.discover_datasets(str(tmp_path))
        assert len(entries) == 1
        e = entries[0]
        assert e["status"] == "complete"
        assert e["sf"] == "18"
        assert e["depth"] == 10
        assert e["multipv"] == 8
        assert e["T"] == 1.0
        assert e["games"] == 4000
        assert e["seed"] == 42
        assert e["positions"] == 243_000

    def test_partial_dataset(self, tmp_path):
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42,
                      complete=False, partial_chunks=3)
        entries = catalog.discover_datasets(str(tmp_path))
        assert len(entries) == 1
        assert entries[0]["status"] == "partial"

    def test_empty_dataset_dir(self, tmp_path):
        # Leaf dir exists but no chunks and no data.npz.
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42,
                      complete=False, partial_chunks=0)
        entries = catalog.discover_datasets(str(tmp_path))
        assert len(entries) == 1
        assert entries[0]["status"] == "empty"

    def test_multiple_datasets_sorted(self, tmp_path):
        _make_dataset(tmp_path, "18", 15, 8, 1.0, 4000, 42)
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42)
        _make_dataset(tmp_path, "18", 10, 8, 0.3, 4000, 42)
        entries = catalog.discover_datasets(str(tmp_path))
        assert len(entries) == 3
        # Sorted by path (deterministic).
        paths = [e["path"] for e in entries]
        assert paths == sorted(paths)

    def test_fractional_temperature(self, tmp_path):
        _make_dataset(tmp_path, "18", 10, 8, 0.3, 1000, 7)
        entries = catalog.discover_datasets(str(tmp_path))
        assert entries[0]["T"] == 0.3

    def test_handles_corrupt_metadata(self, tmp_path):
        leaf = _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42)
        (leaf / "metadata.json").write_text("{ this is not json")
        # Should not raise; falls back to path-parsed info.
        entries = catalog.discover_datasets(str(tmp_path))
        assert entries[0]["status"] == "complete"
        assert entries[0]["positions"] is None  # didn't parse


class TestRenderMarkdown:
    def test_empty_message(self):
        md = catalog.render_markdown([])
        assert "library is empty" in md
        assert "# Game library catalog" in md

    def test_summary_counts(self, tmp_path):
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42)
        _make_dataset(tmp_path, "18", 15, 8, 1.0, 4000, 42,
                      complete=False, partial_chunks=2)
        entries = catalog.discover_datasets(str(tmp_path))
        md = catalog.render_markdown(entries)
        assert "1 complete" in md
        assert "1 partial" in md

    def test_paths_shown_as_code(self, tmp_path):
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42)
        entries = catalog.discover_datasets(str(tmp_path))
        md = catalog.render_markdown(entries)
        assert "`games/sf-18/d10-mpv8-T1/g4000-seed42`" in md


class TestHumanBytes:
    @pytest.mark.parametrize("n,expected_unit", [
        (500, "B"),
        (2048, "KB"),
        (5 * 1024 * 1024, "MB"),
        (3 * 1024 * 1024 * 1024, "GB"),
    ])
    def test_chooses_right_unit(self, n, expected_unit):
        result = catalog._human_bytes(n)
        assert expected_unit in result


class TestEndToEnd:
    def test_main_writes_files(self, tmp_path, monkeypatch):
        _make_dataset(tmp_path, "18", 10, 8, 1.0, 4000, 42)
        # Invoke main() against tmp_path
        monkeypatch.setattr("sys.argv", ["catalog.py", "--root", str(tmp_path), "--quiet"])
        catalog.main()
        assert (tmp_path / "CATALOG.md").exists()
        assert (tmp_path / "catalog.json").exists()
        index = json.loads((tmp_path / "catalog.json").read_text())
        assert "datasets" in index
        assert len(index["datasets"]) == 1
