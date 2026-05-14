# world-models datagen state

Two-region wm-chess runs, what got collected, salvage path via
finalize+merge.

Two distilled-chess datasets were generated on EKS in May 2026, one
per region:

- **us-east-1** — `s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/`
  - depth-15, 8 pods × 12.5k games target = 100k
  - **Actual collected: ~40k games** (pod-0 and pod-3 crashed early; pods 6/7 over-produced)
  - Cluster died before `--finalize-only` step → only `shards_partial/chunks/` exist, no per-pod `data.npz`, no `merged/`

- **eu-west-1** — `s3://wm-chess-library-594561963943-eu-west-1/d10-mpv8-T1-g250000-20260513T0615Z/`
  - depth-10, 4 pods × 62.5k games target = 250k
  - **Actual collected: ~257k games** (essentially complete; two pods left `data.tmp.npz` mid-write)
  - Same shape — `shards_partial/` only, no finalize ran

**Why:** EKS clusters torn down before the gen Job's finalize step in
either region. The chunks under `shards_partial/` are crash-safe
checkpoints; turning them into per-pod `data.npz` requires
`generate_data.py --finalize-only`, then `merge_shards.py` for the
cross-pod merge. Both buckets are otherwise empty.

**Seed schemes** (needed for finalize, since `library_dataset_path()`
is derived from seed):

- us-east base_seed=42, stride=1000 → pod-N seed = 42 + N*1000
- eu base_seed=10142, stride=1000 → pod-N seed = 10142 + N*1000

**Path-prefix bug:** us-east run prefix says `sf-18` but shard subtrees
are `sf-15.1` — Stockfish version mismatch, naming-only.

**How to apply:** When asked "is the dataset ready?" — the answer is
no until a finalize+merge has been run. The recovery path is: pull
`shards_partial/pod-N/` locally, run finalize per pod with original
seed/depth/temperature/n-games, then merge. Treat us-east and eu as
**two separate models**, not one corpus (different depths).
