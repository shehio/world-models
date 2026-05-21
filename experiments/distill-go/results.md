# 9×9 Go distillation — end-to-end demo result

**Date:** 2026-05-20
**Branch:** experiments/distill-go/

## Headline

Student wins **3 / 10** games against KataGo (`g170e-b20c256x2`) at 30
visits each side.

**Elo diff (student − KataGo@30v) ≈ −147**
95% Wilson CI: [−367, +73]

For a one-day, from-scratch pipeline trained on ~5K KataGo-labeled
positions, a finite-but-not-catastrophic Elo gap to a SOTA teacher at
matched visit budget is the expected shape of result. The CI is wide
because n=10; the headline number should not be over-interpreted.

## Pipeline summary

| stage    | config                                                           | time   | output                                |
|----------|------------------------------------------------------------------|--------|---------------------------------------|
| datagen  | 4 workers × 25 games, 9×9, 100 KataGo visits, K=8 multipv        | 21.6 min | 5,493 positions / `go9_merged.npz`  |
| train    | 4-block × 64-filter ResNet, 15 epochs, batch 128, lr 1e-3 (MPS)  | 11 s   | top1=0.88, val_loss=0.21              |
| eval     | student-MCTS @ 30 sims vs KataGo @ 30 visits, 10 games (5 B + 5 W) | 111 s  | 3 W, 7 L → Elo ≈ −147                |

### Per-game results

| #  | student | ply | B pts | W pts | winner | student won |
|----|---------|-----|-------|-------|--------|-------------|
| 1  | B       | 116 |   2.0 |  84.5 |   W    | ✗           |
| 2  | W       | 109 |  78.0 |   9.5 |   B    | ✗           |
| 3  | B       |  44 |  44.0 |  44.5 |   W    | ✗           |
| 4  | W       |  45 |  44.0 |  44.5 |   W    | ✓           |
| 5  | B       |  92 |  16.0 |  64.5 |   W    | ✗           |
| 6  | W       |  45 |  44.0 |  44.5 |   W    | ✓           |
| 7  | B       |  46 |  44.0 |  44.5 |   W    | ✗           |
| 8  | W       | 119 |  81.0 |   7.5 |   B    | ✗           |
| 9  | B       | 134 |   0.0 |  88.5 |   W    | ✗           |
| 10 | W       |  45 |  44.0 |  44.5 |   W    | ✓           |

### Two distinct game outcomes

There are visibly two game types in the table:

- **Contested (1, 2, 5, 8, 9)**: long games (~100 ply), one side accumulates
  most of the board, large point differential. The student lost all five.
- **Quiet (3, 4, 6, 7, 10)**: ~45 ply, both sides reach the two-pass game
  end without significant captures. Final scores are 44 vs 44.5 — exact
  half-and-half split of the 81 board points, with white winning by komi
  (7.5). The student wins these only when it played white (3 of 5 such
  games).

So student's three wins are all komi-driven on uncontested boards. The
student is not yet *fighting* effectively, but it's also not losing the
games it doesn't fight. Both observations are consistent with a small
network trained on a small dataset at low teacher visits (100): the
softmax-over-visits target is too sharp (H ≈ 0.09 across training, near
one-hot), so the student effectively learned hard targets and now imitates
"play KataGo's top move" greedily without learning subtle tradeoffs.

## What the demo proves

- **Pipeline integrity**: datagen → train → MCTS → eval runs end-to-end on
  a single Mac, no infra dependencies beyond a local KataGo install. The
  on-disk `.npz` chunks are byte-compatible with the chess pipeline; the
  chess `merge_chunks.py` reads them unchanged.
- **Rules engine correctness**: 21 unit tests cover capture (single + group),
  suicide, positional superko, pass / two-pass game end, Tromp-Taylor
  scoring (empty / one-color / split). MCTS smoke tests confirm
  legal-move-only selection.
- **Eval methodology**: head-to-head wall-clock-bounded gauntlet vs a SOTA
  teacher with student/teacher visit budgets equalized — the standard
  AlphaZero-vs-engine comparison shape, just at small scale.

## What the demo does NOT prove

- **Strength scaling**: 5,493 positions is roughly 1% of the volume needed
  to see distillation actually start beating the teacher at low visits
  (compare: chess pipeline uses millions of positions). Conclusions about
  "does Go distillation work at scale" cannot be drawn from this number.
- **Soft policy quality**: at 100 KataGo visits the soft policy targets
  collapsed to near-one-hot (training H(tgt) = 0.09). Real production
  runs should use 400+ visits to get meaningful soft information for the
  student to learn from. The spike's memory note flagged this; the demo
  inherited the same limitation.
- **Architecture comparison**: 4 blocks × 64 filters is small. AlphaGo
  Zero's 19×19 net is 20 × 256, and a faithful 9×9 student is probably
  ~10 × 128.
- **History planes**: the demo uses the spike's 4-plane encoding (current
  board + STM + ones). The 17-plane AlphaGo-Zero-style history encoder is
  in `board.py:board_to_history_planes` but the demo's network was sized
  to 4-plane input. Re-running with `--n-input-planes 17` requires
  re-generating the dataset with `--n-input-planes 17` to match shapes.

## To improve the Elo

In rough order of expected impact:

1. **More data** (1× → 10×): 50K positions instead of 5K. Linear in
   wall-clock — bump `games-per-worker` from 25 to 250. Workers/visits
   unchanged. Expected: dramatically reduce CI, modestly improve mean.
2. **Higher teacher visits** (100 → 400): produces soft policy targets
   that aren't near-one-hot. The student then has gradient signal on the
   "second-best move" choice, not just "top move." 4× the datagen time
   for what may be a 100–200 Elo gain.
3. **Bigger student** (4 × 64 → 10 × 128): more capacity to memorize +
   generalize. ~5× training time, modest Elo gain.
4. **History planes** (4 → 17): standard AGZ architecture. Mostly matters
   for ko-rich situations where the recent past disambiguates.
5. **Self-play fine-tune** after distillation. Outside the scope of the
   distillation pipeline — but the chess pipeline's `experiments/selfplay/`
   does exactly this and the same idea ports to Go.

Phase 2 work (not in this demo): build the `wm-go` Docker image, add an
EKS Job spec, and run the rover (`infra-eks/spot-rover/`) against the
go-9x9 config — that path is sketched in `README.md` under "19×19 scale-up".
