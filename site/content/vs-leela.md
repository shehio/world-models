---
title: "Vs Leela Chess Zero"
subtitle: "putting self-play on a distilled prior at ~10⁵× less compute than Leela's all-time training — and finding what breaks"
next: "/next/"
aliases:
  - /leela/
  - /lc0/
---

## The Comparison

| | Leela Chess Zero (Lc0) | this project |
|---|---|---|
| approach | self-play RL (AlphaZero-style), accelerated by supervised warm-starts | supervised distillation from Stockfish + self-play attempt on top |
| network | same AlphaZero family: ResNet, 19-plane history. Production T80 = 384×30 blocks (~120M params) | 20×256 (~24M params) or 40×256 (~50M); same architectural family |
| training compute | thousands of community GPUs over **months** (millions of GPU-hours) | 1 GPU per run, ~5–100 GPU-hours |
| self-play data | hundreds of billions of positions | ~hundreds–thousands of games (six spot-evicted attempts; attempt #7 currently running on OD) |
| supervised pretraining data | early Lc0: human / Stockfish games. Modern Lc0 nets re-bootstrap from older Lc0 self-play | **30–46M Stockfish-labeled positions** (d10/d15 mpv=8 T=1) |
| eval MCTS sims | 800 typical / much higher in tournaments | 800 routine / 4,000 deep-eval / one-off 8,000 |
| peak Elo | **~3,600+ CCRL** (top engine globally) | **2,301** wide-CI (R2 v2 ep 14 sims=4,000 vs UCI=1,800, CI [2,190, 2,601]) · **2,153** tight-CI (ep 4 sims=8,000 vs UCI=2,000, CI [2,084, 2,235]) |

## What's the Same

The shared shape: **a strong prior, then self-play.** Lc0 is
fundamentally a *self-play* (AlphaZero-style) engine — its strength
came from years of community self-play, **not** from distilling a
classical engine. But it leans on warm-starts as accelerants: some
early nets were bootstrapped from human / Stockfish games, and modern
nets re-bootstrap from prior generations' self-play data (a form of
self-distillation). Either way, nobody seriously trains a strong
AlphaZero-style net from genuinely random weights anymore — the warm
start is too valuable.

**Where we differ:** our prior comes from distilling a *classical*
engine (Stockfish), not from self-play. We build that warm start, then
*try* the self-play half on top — and (so far) stop there.

The architecture family is also the same — ResNet trunk with a policy
head producing 4,672 move logits and a scalar value head. Lc0's
production nets are bigger (some have 30+ blocks at 384 channels),
but the *shape* is identical.

## What's Different

### Stopping point — and what we learned by *trying* the self-play half

Lc0 runs self-play → train → self-play indefinitely (each generation
re-bootstrapping from its own prior games). The self-play half is what
closes the gap to engines like Stockfish, and it requires thousands of
GPUs to do meaningfully. The original
write-up of this page said we *planned* a self-play phase but hadn't
run it. We've now run it seven times — six chess attempts on spot
(all evicted before iter 1 finished — see the
[self-play postmortem](/next/#selfplay-postmortem)), one chess
attempt on OD ([currently in flight](/next/)), plus a full 24-hour
Go-side run. The Go run is the cleanest data point: it completed
~63 iterations without infrastructure failures, training loss
dropped 4.71 → 2.50, and a 40-game H2H at iter 42 vs the distilled
prior came in at **21–19–0** — score 0.525, Elo gap +17 with a
±100 Elo CI. **Indistinguishable from no change.**

That null result is actually consistent with the published Lc0
trajectory: the *first* Lc0 self-play iterations took weeks of
community compute before producing detectable Elo gains. We don't
have that compute. The honest headline is that the distilled prior
is the practical ceiling on a single GPU: 2,301 Elo (wide CI) /
2,153 (tight CI) for chess, ≥2,366 Go-scale for the 9×9 Go net.
Lc0's 3,600 is exclusively what years of community self-play on top
of a strong prior buys.

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

## Why Stockfish as Teacher, Not Leela?

Leela is itself an AlphaZero-style network. Distilling from Leela
would mean "make a smaller copy of Leela's network." That's a
legitimate problem (model compression), but a *different* problem
from the one this project is testing. The interesting question here
is: **can the classical alpha-beta family of evaluation be transferred
into the AlphaZero family of architecture?** Distilling from a
classical engine (Stockfish) makes that question well-posed.

Five concrete practical reasons too:

1. **Speed at fixed quality.** Stockfish 17 at depth-15 multipv=8
   labels a position in ~30–80ms on a c7a vCPU. Leela needs a GPU
   inference per node — at the equivalent strength (~thousands of
   sims/move) that's tens of milliseconds per position *per GPU*,
   plus GPU cost. Generating 30-46M labels with Leela would have
   been an order of magnitude more expensive.

2. **MultiPV is native to Stockfish.** Stockfish's `MultiPV N`
   option returns the top-N moves with their centipawn scores per
   position. That's exactly the input our soft-target distillation
   loss wants. Leela's natural output is MCTS visit counts, which
   is a different distribution shape (we'd be approximating the same
   distribution from the wrong-shape source).

3. **CPU-bound, parallelizable, spot-cheap.** Stockfish labelling is
   embarrassingly parallel across positions. Our datagen pods are
   c7a/c7i spot instances on EKS — cheap, abundant, doesn't compete
   with our training GPUs for capacity. Leela-as-labeller would put
   the entire pipeline on GPU.

4. **Deterministic targets.** Stockfish at fixed depth + multipv is
   essentially deterministic — re-running on the same position gives
   the same scores. Leela's MCTS visit counts have small run-to-run
   variation that would add noise to the training signal.

5. **The interesting bug-finding asymmetry.** When the student
   (AlphaZero-architecture ResNet) underperforms the teacher
   (classical Stockfish), the gap is *informative* — it tells us
   whether MCTS-at-inference can recover what the network missed.
   If we'd distilled from Leela, "student underperforms teacher"
   would have no clean explanation: same family, same search
   regime, just a smaller copy. The classical-to-NN transfer
   forces the interesting questions about *how* search and
   evaluation interact.

The version pin (Stockfish 17, released 2024) was incidental — it's
the version `apt install stockfish` ships on Amazon Linux 2023 as of
the datagen runs. Stockfish's strength saturates well above d15 at
modern versions, so the version itself isn't load-bearing for the
results.

## What this Project Is Actually Testing

How close can you get to a strong chess (and Go) net with **distillation
alone — and then *try* the self-play half on a tiny budget** — on a
**single GPU**, on a **fixed budget**?

The honest answer so far:

- **Chess: 2,301 Elo (wide CI) / 2,153 (tight CI)** vs calibrated
  Stockfish opponents. Strong amateur strength, well below grandmaster.
- **Go: ≥ 2,366 (lower-bound)** on a GnuGo-anchored Go-Elo scale (*not*
  the AlphaGo-paper scale), on 9×9. Same competitive-amateur tier,
  anchored to GnuGo L10.
- **Self-play on top: +17 ± 100 Elo over the prior on Go after 4
  hours.** Indistinguishable from no change. Chess attempt #7 in
  flight — same recipe, OD instead of spot, will know within ~24h.

The ceiling on the distillation side is set by:

1. **Data scale** — 30–46M positions vs Lc0's hundreds of billions.
2. **Teacher quality** — Stockfish d15 is ~2,500 Elo, plus the
   multipv=8 soft targets dilute the signal somewhat.
3. **Training-recipe sophistication** — constant LR plateaus,
   cosine LR closed the gap to the teacher (R2 v2 cosine ep 14 = 2,301
   vs R1 v1 constant LR ep 7 = 2,146).

The ceiling on the *self-play* side is set by:

1. **Compute** — Lc0 needed wall-weeks of community GPUs to see real
   gains on top of a distilled prior. Our 24h-budget × 1 GPU is ~10⁴×
   less. The Go null result is the textbook signature.
2. **Algorithmic tricks** — KataGo's paper documents ~9× efficiency
   gains over vanilla AlphaZero through ownership-aux heads, global
   pooling, opponent-policy prediction, and game-specific features.
   *Lc0 also didn't use these* — Lc0's actual recipe is just "good
   MCTS + lots of self-play games," same as ours. They got away with
   it because they had the games.

What this project *deliberately doesn't* try: matching Lc0's headline
strength. To do that you'd need (a) the self-play phase running at
genuine Lc0-scale compute (~$10⁵× our budget), and probably (b) the
KataGo efficiency tricks layered on top.

## Lessons that Mirror — and Extend — Lc0's Trajectory

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
  We hit 2,301 Elo in ~16 GPU-hours. Lc0's tabula-rasa years would
  not have produced that quickly without warm-starting.
- **Self-play improvement on top of a distilled prior is brutally
  compute-sensitive.** This is the new lesson from our seven attempts.
  Lc0 reads in retrospect as "distill + self-play work together" —
  but the 1,000+ Elo gap from distill-only (~2,500) to Lc0-final
  (~3,600) is *entirely* the self-play half, and self-play needs
  ~unbounded compute to deliver. Our 24h budget is too short to
  even *register* an improvement, let alone a meaningful one.
- **Infrastructure is half the battle.** Lc0's volunteer-distributed
  training architecture is itself a research contribution. Our
  six-attempt spot-eviction saga ([postmortem](/next/#selfplay-postmortem))
  is the small-scale analog: the *algorithm* worked from day one,
  but the eviction window in a given AZ was shorter than one iter,
  so iter 1 never completed across six tries before we moved to OD.

The story is consistent: **a strong prior + self-play is the right
recipe, and the gap to Lc0's headline is overwhelmingly the self-play
half running at compute we don't have.
What's interesting at our scale is the *distill ceiling*, which we
get to publish numbers for, and the *failure modes* of single-GPU
self-play, which we can document.**
