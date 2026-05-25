---
title: "Vs Leela Chess Zero"
subtitle: "we're running Leela's distill-then-RL recipe — and stopping after the distill"
next: "/next/"
aliases:
  - /leela/
  - /lc0/
---

## The Comparison

| | Leela Chess Zero (Lc0) | this project |
|---|---|---|
| approach | supervised pre-training + tabula-rasa self-play RL (hybrid) | supervised distillation from Stockfish only |
| network | same AlphaZero family: ResNet, 19-plane history. Production T80 = 384×30 blocks (~120M params) | 20×256 (~24M params) or 40×256 (~50M); planned to go bigger |
| training compute | thousands of community GPUs over **months** (millions of GPU-hours) | 1 GPU per run, ~5-100 GPU-hours |
| self-play data | hundreds of billions of positions | **none yet** (planned distill-then-RL still in flight) |
| supervised pretraining data | early Lc0: human / Stockfish games. Modern Lc0 nets re-bootstrap from older Lc0 self-play | **30-46M Stockfish-labeled positions** (d10 mpv=8 T=1) |
| eval MCTS sims | 800 typical / much higher in tournaments | 800 routine / 4,000 on the deep-eval daemon |
| peak Elo | **~3,600+ CCRL** (top engine globally) | **2,189** (d10 30M ep 15 @ sims=4,000 vs UCI=1,800) |

## What's the Same

The recipe shape. **Leela's actual training story is "distill first,
RL second."** Early Lc0 networks were warm-started from supervised
training on human / Stockfish games before self-play kicked in;
modern Lc0 nets re-bootstrap from prior generations' self-play data,
which is itself a form of distillation. Either way: nobody seriously
trains a strong AlphaZero-style chess net from genuinely random
weights anymore — the warm start is too valuable.

That's the recipe we're running here. We just stop after the warm
start.

The architecture family is also the same — ResNet trunk with a policy
head producing 4,672 move logits and a scalar value head. Lc0's
production nets are bigger (some have 30+ blocks at 384 channels),
but the *shape* is identical.

## What's Different

### Stopping point

Lc0 runs distill → self-play → distill → self-play indefinitely. The
self-play half is what closes the gap to engines like Stockfish, and
it requires thousands of GPUs to do meaningfully. This project ran
the distill phase and *plans* a self-play phase ([what's next →](/next/))
but has not run it yet. The 2,189 Elo headline is "what you can get
from distillation alone, on one GPU, for ~16 GPU-hours of training."
Lc0's headline of ~3,600 is "what you can get from years of community
self-play on top of distillation."

### Scale

Lc0 is a community-distributed effort. At peak it had ~3,000
volunteers contributing games. The all-time training compute is
genuinely uncountable — millions of GPU-hours across consumer GPUs,
RTX cards, and donated cloud. This project is one person on AWS for
about a week and a half, ~$1,000 total spend.

### Teacher quality

Lc0's supervised phase used grandmaster games, then Stockfish, then
its own prior generations. Each switch made the targets sharper or
weaker depending on the era. We used Stockfish d10 (~2,200 strength)
or d15 (~2,500), measured the distillation Elo against a calibrated
Stockfish, and found that — at our compute scale — the *teacher's*
strength didn't fully translate. The d15-trained 40×256 net
*tied* the d10-trained 20×256 net at sims=4,000 (head-to-head =
0/104/0, [see experiments](/experiments/#d10-vs-d15)). Either we hit
the data-scale ceiling at 30-46M positions or constant-LR
distillation has a recipe ceiling we haven't broken yet.

### Network size

Lc0's biggest production nets are ~10-30× the parameter count of our
biggest (40×256 = 50M params). Theirs is the right size for billions
of training positions. Ours is the right size for tens of millions.
A 384×30 net on our 46M positions would just memorize.

### Search depth at eval

Lc0 in a real tournament uses tens or hundreds of thousands of sims
per move with parallel branch evaluation across GPU cores. Our
"deep-search" eval uses 4,000 sims on a single GPU — meaningful, but
not in the same regime.

## What this Project Is Actually Testing

How close can you get to a strong chess net with **distillation alone**
on a **single GPU**, on a **fixed budget**?

The honest answer so far: ~2,189 Elo vs UCI=1,800-anchored Stockfish.
Strong amateur strength, well below grandmaster. The ceiling is set
by:

1. **Data scale** — 30-46M positions vs Lc0's hundreds of billions.
2. **Teacher quality** — Stockfish d15 is ~2,500 Elo, plus the
   multipv=8 soft targets dilute the signal somewhat.
3. **Training-recipe sophistication** — constant LR plateaus,
   cosine schedules might help (see the in-flight R1 v2 and R2
   cosine variants).

What this project *deliberately doesn't* try: matching Lc0's headline
strength. To do that you'd need either (a) the self-play phase that's
in the [next steps](/next/), or (b) ~10⁵× the compute budget. Both are
out of scope for "what fits on a single GPU."

## Lessons that Mirror Lc0's Trajectory

- **Bigger network ≠ better at our scale.** Lc0 confirmed this too,
  early on — the 256-channel nets stayed competitive with the
  384-channel nets until self-play data scale caught up. The
  [capacity ablation](/experiments/#capacity) replays this finding
  at smaller compute.
- **Search recovers Elo a weaker prior can't earn on its own.**
  Lc0 nets at low sim count and high sim count are ~hundreds of
  Elo apart — same dynamic we measured (+277 Elo from sims=800
  to sims=4,000 on the d15 5M prior).
- **Distillation gives a usable starting point in hours, not months.**
  We hit 2,189 Elo in ~16 GPU-hours. Lc0's tabula-rasa years would
  not have produced that quickly without warm-starting.

The story is consistent: **the recipe Lc0 actually shipped (distill
+ self-play) is the right one**. We're running the cheap half of it
to see how far that gets, and finding it's further than you'd guess
but still leaves the headline-level gap entirely on the self-play
side.
