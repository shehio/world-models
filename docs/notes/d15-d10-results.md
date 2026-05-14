# d15 / d10 latest Elo numbers

distill-soft results as of 2026-05-14: d15 20×256 plateaued ~1807 Elo
at ep19; d10 5M subset and ablations A/C/E running.

## d15 mpv8 T=1 g100000 (20×256, 5M positions)

| epoch | UCI=1350 Elo | W/D/L |
|---:|---:|---:|
| 4 | 1701 | 62 / 35 / 3 |
| 9 | 1693 | 62 / 36 / 2 |
| 14 | 1759 | 91 / 8 / 5 |
| 19 | **1807** | 93 / 8 / 3 |

Loss plateaued ~2.59 by ep10, top-1 ~0.34. Each new epoch adds ~10-50
Elo within noise, suggesting the recipe ceiling is around 1800.

## d10 mpv8 T=1 g250000 (20×256, 5M positions)

A few aborted runs (npz extraction filled the 150 GB volume; needed
`--max-positions` truncation + 300 GB volume).

`d10-l40s-eu.sh` resumes from
`INIT_FROM_S3=…/distilled_epoch004.pt START_EPOCH=5` on `g6e.xlarge`
in eu-central-1.

## Ablations running at writeup

- **A** (40×256, d15, 5M): tests capacity-limited hypothesis.
- **C** (20×256, d10, FULL 30M): tests dataset-size-limited
  hypothesis.
- **E** (sims=4000 re-eval of d15 ep19): tests "model is under-read by
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
