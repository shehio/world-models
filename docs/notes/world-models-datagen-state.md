# world-models datagen state

Two-region wm-chess runs, what got collected, salvage path via
finalize+merge.

Two distilled-chess datasets were generated on EKS in May 2026, one
per region:

- **us-east-1** — `s3://wm-chess-library-594561963943/d15-mpv8-T1-g100000-20260512T1844Z/`
  - depth-15, 8 pods × 12.5k games target = 100k target
  - **Final merged: 49,525 games / ~5.07M positions** at `merged/data.npz`
    (about half of target — pods 0 and 3 crashed early; pods 6/7 over-produced
    a bit but didn't close the gap). The S3 prefix still says `g100000`
    because rename-on-S3 is expensive; the merged `metadata.json` carries
    the real `n_games`.

- **us-east-1** — `s3://wm-chess-library-594561963943/d10-mpv8-T1-g250000-20260513T0615Z/`
  - depth-10, 4 pods × 62.5k games target = 250k target
  - **Final merged: 249,913 games / ~30.66M positions** (essentially the
    full target) at `merged/data.npz`

**Why the d15 dataset is half what the prefix implies:** EKS pod
churn during the initial run. Finalize + streaming merge now happen
in the same Job (resilient to spot termination), but the d15 prefix
was named at submission time and stuck. Always trust
`merged/metadata.json` over the prefix.

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
