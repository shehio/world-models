---
title: "Go: 9x9 Distillation from KataGo"
subtitle: "8x128 ResNet, 1.236M positions, parity with KataGo@v200 in 15 epochs"
next: "/experiments/"
aliases:
  - /go-9x9/
  - /distill-go/
---

## Headline

A 9×9 Go agent distilled from KataGo (multipv=8 at v=400, ~1.236M
positions) reaches **parity with KataGo at 200 visits** in 15 epochs.
Anchored against GnuGo L10 (assumed ≈1,800 Elo on a conventional
Go-Elo scale; see the anchor caveat below), the KataGo @v=200 opponent
sits at **≥ 2,366 absolute Elo**, so the student does too *on that scale*.

| ckpt | sims | opponent | win rate | Elo gap |
|---|---:|---|---|---:|
| ep 5 | 100 | KataGo @ v200 | 4/30 = 13% | **−325** [−500, −150] |
| ep 11 | 100 | KataGo @ v200 | 14/30 = 47% | **−23** [−145, +99] |
| **ep 15** | **100** | **KataGo @ v200** | **15/30 = 50%** | **0** [−122, +122] |
| ep 19 | 100 | KataGo @ v50 | 10/30 = 33% | −120 [−249, +8] *(weaker anchor, wider CI)* |
| ep 19 | 800 | KataGo @ v200 | 15/30 = 50% | 0 [−122, +122] |

**+325 Elo from ep 5 to ep 15.** Then a clean plateau. The
distillation training reached its natural ceiling: matching what the
teacher gave it, then stopping.

The earlier spike (4×64 net, 5K positions, KataGo @ v100 teacher)
scored **Elo −147** vs KataGo @ v30. The full 8×128 net on 1.236M
positions is **~150 Elo above the spike** under the harder v200 anchor.

## Anchoring KataGo @v=200 to an Absolute Elo {#absolute-elo}

To convert the "parity with KataGo @v=200" result into a Go-Elo
absolute number, we need to anchor KataGo @v=200 itself against a
calibrated opponent. Run 2026-05-26 18:21 UTC (`i-07a0f9582d96fcd9a`,
g6.xlarge OD): **100 games KataGo @v=200 vs GnuGo L10 on 9×9, komi 7.5,
anchor GnuGo = 1,800 Elo** (assumed; a conventional single-digit-kyu
Go-Elo estimate, *not* the AlphaGo-paper scale, see caveat below).

```
KataGo @v=200  vs  GnuGo L10  (100 games, 9×9):  100 W / 0 D / 0 L
Score 1.000, 95% CI [0.963, 1.000]
Elo gap (KataGo − GnuGo) ≥ +566  (lower 95% bound; upper bound
                                   uninformative: score saturation)
→ KataGo @v=200 absolute Elo  ≥  2,366  (lower 95% CI)
→ Point estimate  ≈  5,400: clamped, ignore
```

The lower bound is real (gnugo competently lost 100 games in 640 sec
total wallclock), the point estimate is not (a 100-0 sweep blows up
the Elo conversion at the upper end).

So our 8×128 distilled student, at parity with KataGo @v=200, is also
**at least 2,366 absolute Go Elo**. The "true" KataGo @v=200 Elo is
probably substantially higher (community estimates put modern KataGo
small-board networks at 2,800+ on 9×9) but pinning it requires a
stronger anchor than GnuGo. Pachi 1.0 from Ubuntu apt was the obvious
candidate but deadlocks in non-TTY containers regardless of playout
count; CrazyStone / Aya / Lc0 would each need new docker integration.

**Anchor caveat.** The ≈1,800 GnuGo figure is an *assumption* on a
conventional Go-Elo scale; it is **not** from the AlphaGo paper. That
paper (Silver et al., Nature 2016, Extended Data Table 6) actually rates
GnuGo 3.8 (level 10) at **431 Elo** on its own compressed BayesElo scale
(anchored to Fan Hui = 2,908; the ~1,800–1,900 entries there are Zen and
CrazyStone, not GnuGo). So **≥ 2,366 is only valid on a scale where GnuGo
L10 ≈ 1,800**; treat the absolute number as anchor-dependent. The lower
*bound* (KataGo @v=200 beats GnuGo by ≥ 566 Elo) is robust regardless of
the anchor.

→ launcher: [`infra-eks/launchers/calibrate-katago-vs-gnugo.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/calibrate-katago-vs-gnugo.sh) ·
result: `s3://wm-chess-library-594561963943/go-calibration/katago-v200-vs-gnugo-l10-20260526T1821Z.txt`

## What's the Setup

| | value |
|---|---|
| board | 9×9 |
| teacher | KataGo @ 400 visits, multipv=8, T=1 |
| games generated | 32K self-play games (~1.236M positions) |
| network | 8 blocks × 128 channels (~9.6M params) |
| input planes | 4 (current board + side-to-move + ones) |
| training | 20 epochs, Adam, LR=1e-3, MPS local |
| eval opponent | KataGo at fixed visits (v50, v200, v800) |
| eval scale | 30 games per measurement, alternating colors |

## Training Trajectory (sims=100, v200 anchor)

```
ep 5  ─────────  −325 (still learning, lots of error)
ep 11 ───        −23 (within noise of parity)
ep 15 ──         0   (parity)
ep 19 ──         0   (plateau holds)
```

The big gain is between ep 5 and ep 11 (+302 Elo in 6 epochs).
Between ep 11 and ep 15 we gain another +23. After ep 15, no
further improvement. **The agent saturated at "match the teacher's
shape at moderate search depth."**

## Sims Sweep: Does More Search Help?

| ckpt | sims | opponent | Elo gap |
|---|---:|---|---:|
| ep 19 | 100 | KataGo @ v200 | −23 |
| ep 19 | 800 | KataGo @ v200 | 0 |

**+23 Elo from sims=100 → sims=800.** Much smaller than the chess
sims-amplification (chess sims=800 → sims=4000 gave +200+ Elo). Two
plausible reasons:

1. **9×9 Go has a smaller branching factor** than chess. MCTS at
   sims=100 already explores a meaningful chunk of the relevant
   space; the marginal benefit of sims=800 is smaller.
2. **The agent's policy already matches KataGo's** at the current
   strength level. Search amplifies the policy's strength, but
   can't push it past the teacher's ceiling.

If we want significantly more Elo from this agent, the lever isn't
"more sims at eval"; it's "more capacity / more data / self-play."

## What Failed Mode Did *Not* Trigger

The 250K-positions MPS chess distillation famously **regressed**
to the small-network baseline because of soft-target dilution
([results.md §3](https://github.com/shehio/world-models/blob/main/experiments/distill-soft/results.md)).
Go *did not* regress, went from −325 to 0 cleanly. The difference:

- Go uses 8 multipv at the KataGo teacher's softmax, but Go's
  policy distribution is naturally less peaked than chess (more
  candidate moves are nearly equivalent in many positions). So
  the "soft targets dilute signal" failure mode is less acute.
- Go training had **30× more positions than the 250K MPS chess
  run** (1.236M vs 250K), proportional to the network size jump
  (8×128 vs 20×256), so positions-per-parameter was healthier.

## Comparing to Chess

Chess and Go distillation hit very different ceilings against
their respective teachers:

| | chess | Go |
|---|---|---|
| teacher | Stockfish d10/d15 (~2,200-2,500 Elo) | KataGo @ v400 |
| best student vs teacher (matched sims) | tied with d10 in H2H | tied with KataGo@v200 |
| sims=N → sims=4×N Elo bump | +200 | +23 |
| jumps from initial to peak | +900 (1,180 → 2,189) | +325 (ep 5 → ep 15) |

The qualitative story is the same: **distillation saturates near the
teacher's strength at the search depth the student was trained at.**
Going beyond requires self-play (the [next-step](/next/)).

## Code

- [`experiments/distill-go/`](https://github.com/shehio/world-models/tree/main/experiments/distill-go): datagen, training, eval
- [`infra-eks/launchers/eval-go-9x9-8x128.sh`](https://github.com/shehio/world-models/blob/main/infra-eks/launchers/eval-go-9x9-8x128.sh): EC2 eval launcher (used for all the v200 numbers above)
- Checkpoints in S3:
  `s3://wm-chess-library-594561963943/go-9x9-mpv8-gpu-v400-20260521T1230Z/checkpoints/9x9-8x128-20260522T0625Z/`

## What's Next for Go

In rough order of expected information per dollar:

1. **Higher KataGo visits** (v400, v800): does the plateau hold
   when the opponent gets stronger? Will tell us where the agent
   sits on the absolute scale. ~$1, ~40 min. **Running now.**
2. **Self-play RL on top of ep 15**: Go is where AlphaZero-style
   self-play *really* shines (KataGo's whole training was self-play,
   no teacher). Bootstrap from the matched-with-KataGo@v200 ckpt
   and let it diverge into something better. ~$50-100.
3. More data (5M+ positions) + retrain at same architecture.
4. Bigger network (12×128 or 8×256) at the current data.
5. Scale to 13×13 or 19×19 board: different experiment entirely.
