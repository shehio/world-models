# finalize / merge OOM

Both `finalize_library_path` and `merge_shards.py` materialize all
chunks in RAM (`load` + `np.concatenate`), which OOMs on large pods.
Workaround: streaming merge.

`experiments/distill-soft/src/distill_soft/stockfish_data.py::finalize_library_path`
and `wm_chess/scripts/merge_shards.py` both follow the pattern:

```python
arrays = []
for cf in chunks:
    arrays.append(np.load(cf)[k])
np.concatenate(arrays, axis=0)
```

This holds every chunk in RAM, then `np.concatenate` doubles peak. For
wm-chess chunks (states `(N, 19, 8, 8) float32`, ~30 MB uncompressed
per chunk, ~70x compression ratio), this blows up fast:

- us-east pod-7 (165 chunks): ~5 GB raw → ~11 GB peak → OOMs `m6i.large` (8 GB RAM)
- eu pods (~1300 chunks each): ~35 GB raw → ~70 GB peak → won't fit on any sensible single instance
- Cross-pod merge of 4 EU pods: ~140 GB raw → ~280 GB peak → impossible

The OOM is silent — kernel OOM-killer SIGKILLs Python with no
traceback. In v1 of the May 2026 salvage attempt, pods 0–5 finalized
on `m6i.large` then pod-6 (150 chunks) was killed silently.

**Workaround used (May 2026):** `wm_chess/scripts/merge_chunks.py` — a
streaming script that bypasses both finalize and merge. For each array
key, it opens one zip entry, writes the npy header, then streams each
chunk's `tobytes()` directly into the deflate stream. Peak RAM ~100 MB
regardless of dataset size.

**Why:** Original gen pods ran on `m6i.8xlarge` / `c7i.8xlarge`
(64–128 GB RAM) and would have finalized their own per-pod data
successfully. The bug only surfaces during recovery on smaller
instances. See [`world-models-datagen-state.md`](./world-models-datagen-state.md)
for the broader salvage context.

**How to apply:** If anyone fixes this in-repo, the proper patch is to
make both `finalize_library_path` and `merge_shards.py` write per-key
with a zip-entry-streaming pattern (use
`numpy.lib.format.write_array_header_2_0` +
`zipfile.ZipFile.open(..., "w", force_zip64=True)` + chunked
`tobytes()` writes). Until then, point any large-pod recovery work at
the streaming approach in `merge_chunks.py`.
