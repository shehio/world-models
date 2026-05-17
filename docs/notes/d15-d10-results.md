# d15 / d10 latest Elo numbers

distill-soft results as of 2026-05-14: d15 20×256 plateaued ~1807 Elo
at ep 20; d10 5M subset and ablations A/C/E running.

## d15 mpv8 T=1 (20×256, 5M positions)

S3 prefix says `g100000` (100K target) but actual collected was
**49,525 games** (~50K) producing ~5.07M positions.

| epoch | UCI=1350 Elo | W/D/L |
|---:|---:|---:|
| 5 | 1651 | 62 / 35 / 3 |
| 10 | 1862 | 62 / 36 / 2 |
| 15 | 1759 | 91 / 8 / 5 |
| 20 | **1807** | 93 / 8 / 3 |

Loss plateaued ~2.59 by ep 10, top-1 ~0.34. Each new epoch adds ~10-50
Elo within noise, suggesting the recipe ceiling is around 1800.

## d10 mpv8 T=1 (20×256 on 5M subset and full 30M)

S3 prefix says `g250000` (250K target) and actual collected was
**249,913 games** (essentially the full target) → 30.66M positions.

A few aborted runs (npz extraction filled the 150 GB volume; needed
`--max-positions` truncation + 300 GB volume).

`d10-l40s-eu.sh` resumes from
`INIT_FROM_S3=…/distilled_epoch004.pt START_EPOCH=5` on `g6e.xlarge`
in eu-central-1. (Note: `distilled_epoch004.pt` is the 5th completed
epoch — file names are 0-indexed in the training code; display calls
it ep 5.)

## Ablations running at writeup

- **A** (40×256, d15, 5M): tests capacity-limited hypothesis.
- **C** (20×256, d10, FULL 30M): tests dataset-size-limited
  hypothesis.
- **E** (sims=4000 re-eval of d15 ep 20): tests "model is under-read by
  800-sim eval depth."

## Headline interpretation

02c d15 (1807) is now **above** 02b d10 (1185) — the old "negative
result" reversed by stronger teacher + EKS-scale training infra. The
1750-1800 plateau is what A/C/E are designed to probe.

**Why:** keep this around so future me doesn't have to re-derive the
Elo trajectory from S3 every time someone asks "what's the current
number?"

**How to apply:** Before quoting Elo numbers in conversation, verify
with `aws s3 cp s3://wm-chess-library-594561963943/<prefix>/checkpoints/net-NxF/<run>/eval_results-distilled_epochNNN.txt -`
in case more epochs landed. Numbers above are point-in-time; for
*current* state run `aws s3 ls` against the run directory.
