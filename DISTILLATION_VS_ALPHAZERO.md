# 02b / 02c vs real AlphaZero — what's the same, what's different

The 02 directory implements AlphaZero faithfully. The **02b** and **02c**
directories are deliberately-non-faithful sibling experiments that share
the *architecture* of 02 but replace its training procedure with
**supervised distillation from Stockfish**. This doc explains the
distinction and the relationship between the three.

## Why have both

AlphaZero's defining feature is **tabula rasa self-play RL**: the
network learns from games against itself, with no external chess
knowledge. That's the science of 02.

02b and 02c ask a different question: *given the same network, how
strong can it get if we hand it good training targets instead of
asking it to discover everything by self-play?* If 02 plateaus at +372
Elo vs random (as v5 did) and 02b reaches real Elo 1185 with the same
network in **one hour of training**, the network was clearly not the
bottleneck for 02 — the self-play loop was. That's a useful thing to
know before scaling up.

## Architecture: identical across all three

```
input: (B, 19, 8, 8) board planes
   │ — piece positions (12 planes: 6 white + 6 black piece types)
   │ — side-to-move (1 plane)
   │ — castling rights (4 planes)
   │ — en passant target (1 plane)
   │ — halfmove clock (1 plane, normalized)
   ▼
Conv 3x3 → BN → ReLU              (input projection)
   ▼
[ResBlock] × N                    (each: conv→BN→ReLU→conv→BN→add→ReLU)
   ▼
   ├──► policy head: 1x1 conv → fc → 4672 logits  (one logit per possible move)
   └──► value head:  1x1 conv → fc → tanh → scalar in [-1, +1]
```

| Project | Blocks × channels | Params |
|---|---|---|
| 02 (v3c, v4, v5) | 10 × 128 | ~3M |
| 02b (d6, d10)    | 10 × 128 | ~3M |
| **02c** (scaled) | **20 × 256** | **~21M** |

The 4672 = 8×8×73 chess move encoding (queen-like × 56 + knight jumps × 8 + underpromotions × 9, keyed by from-square).

## Training: this is the actual difference

|                              | **02 (faithful AlphaZero)** | **02b (hard-target distillation)** | **02c (multipv-soft distillation)** |
|------------------------------|-----------------------------|------------------------------------|-------------------------------------|
| Source of training data      | Self-play games             | Stockfish self-play games          | Stockfish self-play games          |
| Policy target                | MCTS visit-count distribution from the agent's own search | Stockfish's actually-played move (one-hot) | Stockfish's full top-K distribution (softmax over centipawns at temperature T) |
| Value target                 | Game outcome from self-play | Game outcome from Stockfish-vs-Stockfish | Game outcome from Stockfish-vs-Stockfish |
| Loop                         | Policy iteration: net → MCTS → better policy → train → repeat | One-shot supervised. No loop. | One-shot supervised. No loop. |
| External chess knowledge     | None (tabula rasa)          | Stockfish moves used as ground truth | Stockfish move *ranking + scores* used as ground truth |
| Bits per training position   | ~few (whatever MCTS extracts)| log₂(legal_moves) ≈ 5 bits        | log₂(legal_moves) + log of full top-K ordering ≈ 10+ bits |
| Theoretical ceiling          | Unbounded — paper hit ~3500 Elo | ≈ teacher's strength minus search contribution | Same ceiling as 02b; possibly tighter teacher fit at the same data scale |
| Wall-clock cost (this repo)  | 7–20 hours laptop self-play | ~1 hour total: data gen + training | ~3 hours total: data gen + larger-net training + eval |

## Inference: identical across all three

All three projects use the same MCTS at decision time. The trained
network provides the prior over moves (policy head) and the leaf-node
value estimate (value head); the MCTS does the search:

```
   for _ in range(N_sims):
       descend from root by PUCT score until a leaf
       if leaf is terminal: v = ±1 / 0
       else: expand leaf via network forward pass; v = value head
       walk path leaf → root, accumulating v and flipping sign each step

   π = (visit counts at root) / sum(visit counts)
   move = sample π if early in game else argmax π
```

"Sims" in our run logs = N MCTS simulations per move. More sims = better
visit-count distribution = stronger play, at fixed network. v5 saw ~+121
Elo going from 80 → 400 sims; 02c is configured for 800 (baseline) and
1600 (stretch).

## NPZ schema (02c specific)

The distillation dataset for 02c is a single compressed numpy archive
(`.npz`) holding 5 arrays:

```
states            (N, 19, 8, 8)  float32   – board positions, same encoder as 02
moves             (N,)           int64     – Stockfish's actually-played move idx
zs                (N,)           float32   – game outcome from STM POV
multipv_indices   (N, K=8)       int64     – top-K candidate move indices
multipv_logprobs  (N, K=8)       float32   – softmax-logprobs over those K moves
```

Padding for endgame positions with fewer than K legal moves:
`multipv_indices[i, j] = -1` and `multipv_logprobs[i, j] = -inf`. The
training loss masks these out by construction since exp(-inf) = 0.

02b's NPZ omits the last two columns — it has only the played-move
one-hot. That's the entire data-side difference between 02b and 02c.

## Stockfish in the loop

Stockfish 18 (the open-source chess engine) plays two roles:

| Role | Where | How configured |
|---|---|---|
| **Teacher** (Phase 1 data gen) | 02b, 02c | `engine.analyse(board, Limit(depth=D), multipv=K)` at fixed depth (d6/d10/d15). For 02b, K=1; for 02c, K=8. |
| **Opponent** (Phase 3 eval) | 02, 02b, 02c | `UCI_Elo` set to a strength-limited level (1320 on macOS-brew, 1350+ on apt-Ubuntu). Plays at `depth=8`. |

The depth-D teacher uses *full-strength* Stockfish capped at search
depth D. That gives roughly:
- d6 ≈ 1900 teacher Elo
- d10 ≈ 2200 teacher Elo
- d15 ≈ 2500 teacher Elo

The student can never exceed the teacher; in practice it caps around
*teacher − 800–1000 Elo* because Stockfish's strength comes from search
the student can't replicate in one forward pass.

## Pipeline phases (02c)

1. **Phase 1 — data generation.** N Stockfish workers play self-games at depth D with multipv K. Output: one `.npz`.
2. **Phase 2 — supervised training.** One process trains the 20×256 ResNet on the NPZ for E epochs. Loss = soft-CE on the multipv distribution + MSE on game outcome.
3. **Phase 3a — eval at baseline sims.** 120 games of the trained net vs Stockfish UCI_Elo=1350 at 800 MCTS sims/move. Apples-to-apples with 02b.
4. **Phase 3b — stretch eval.** Same 120 games but at 1600 sims/move. Measures the search-budget bonus on top of the trained model.

See [02c-distill-scaled/README.md](./02c-distill-scaled/README.md) for
runbook details and [02b-alphazero-stockfish-distill/README.md](./02b-alphazero-stockfish-distill/README.md)
for the original hard-target distillation results.

## Why the ceiling is *teacher − search*

This is the core theoretical insight behind both 02b and 02c, and it's
worth stating explicitly:

> A teacher like Stockfish doesn't *know* good moves — it *calculates*
> them, evaluating millions of positions ~10 plies deep with an
> alpha-beta search. The student is being asked to reproduce that
> *output* via a single forward pass through a small ResNet. The
> student becomes a chess-grandmaster-impersonator with no actual
> calculation ability.

A 7B-param language model "distilled" from GPT-4 is similarly much
weaker than GPT-4. Lc0's networks alone aren't grandmaster-strength;
Lc0 + MCTS-search is. The search is doing real work that distillation
can't transfer.

For us: 02b's d10 teacher is ~2200 Elo, our distilled student lands
~1185 Elo (gap ~1015 Elo). That gap is what the alpha-beta search of
Stockfish was buying. To close it, you need self-play fine-tuning on
top of the distilled prior — which is exactly what Lc0 did, taking
several wall-weeks on a distributed GPU cluster. That's the Lc0
sequence: **distill first, RL second.** The two halves of the
chess-RL pipeline.
