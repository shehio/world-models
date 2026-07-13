---
title: "Vs MuZero"
subtitle: "MuZero ditches the rules engine and learns the model; we kept the rules"
next: "/next/"
aliases:
  - /muzero/
  - /mz/
---

## The Comparison

| | MuZero (DeepMind 2019) | this project |
|---|---|---|
| Approach | learn a *world model* from observations + self-play RL | use *known rules* + supervised distillation from Stockfish/KataGo |
| Domain | board games **and** Atari, RT-pixels-in, no rules engine | board games only, rules engine is canonical |
| Networks | three: **representation** *h* (obs → latent) + **dynamics** *g* (latent + action → next latent + reward) + **prediction** *f* (latent → policy + value) | one: AlphaZero-family ResNet (state → policy + value) |
| MCTS | runs over the *learned* dynamics function, not the true environment | runs over `python-chess` / `GoBoard`: the real, deterministic env |
| Training compute (board games) | similar order to AlphaZero: thousands of TPUs × days | 1 GPU per training run, ~16 GPU-hours |
| Peak Elo (chess) | ~3,500 (matches AlphaZero) | 2,301 wide-CI / 2,153 tight-CI |
| Rules knowledge | **none**: discovers legality, capture, checkmate from data | full python-chess legality + termination |

## What MuZero Adds Over AlphaZero

MuZero is *AlphaZero plus a learned world model*. AlphaZero needed the
rules engine (chess.Board, GoBoard) at every MCTS node; every
simulated child is a real legal position. MuZero removes that
requirement by training three networks that *together* simulate the
environment:

```
real observation o_t
        │
        ▼
   h_θ(o_t) = s⁰_t        latent state at the root
        │
        ▼   ┌─────────────────────────────────────────────────────┐
   MCTS    │  recursive descent through the LEARNED dynamics g_θ │
            │   g_θ(s_t^k, a_{t+1}) = (s_t^{k+1}, r_t^{k+1})      │
            │   prediction f_θ(s_t^k) = (π_t^k, v_t^k)            │
            └─────────────────────────────────────────────────────┘
        │
        ▼
   visit-count π over root actions → move played
```

The dynamics function `g_θ` is trained to predict what happens *only
on the trajectories MCTS visits*, not the full state space. That means
MuZero never has to learn "what's a legal move in chess" in general,
only enough to evaluate the moves its own search considers. That's
much less than learning chess rules ab initio, but it's enough to
match AlphaZero's strength on chess + shogi + Go and **also** play
Atari (where rules are unknown).

## What This Project Has That MuZero Doesn't Care About

We assume the rules. `wm_chess.board.encode_board` reads
`python-chess`'s legality directly, MCTS only descends legal moves,
game termination is determined by `board.is_game_over()`. That's a
massive assumption MuZero deliberately drops, but for our problem
(can a small ResNet distilled from Stockfish play strong chess?) it's
the right assumption. Re-learning chess rules from scratch would be
~10⁶ extra positions of training signal that we don't have, on top of
the actual hard problem (learning to *play well*).

A cleaner way to say it: **the world-model-learning question is
genuinely orthogonal to the question this project asks.** AlphaZero
and MuZero hit the same Elo on chess; the difference is *what they
needed to see* to get there.

## Why Our Network Is "AlphaZero" Not "MuZero"

Three structural reasons:

1. **We feed in real board state, not pixel observations.** The
   19-plane encoding (`encode_board`) is a hand-crafted representation
   of piece positions, castling rights, en-passant, and the side to
   move. MuZero's representation network `h_θ` is *learned* from raw
   inputs (e.g. screen pixels for Atari). For board games MuZero
   *also* uses the same hand-crafted planes, so on chess specifically
   MuZero's representation network is doing little work.
2. **MCTS calls the real env.** Every descent in our MCTS calls
   `board.push(move)` on a python-chess Board. MuZero would call
   `g_θ(s, a)` on the latent. We benefit from the real env being
   exact and free; MuZero benefits from generalizing past known-env
   problems.
3. **We have a teacher.** Stockfish labels positions for us. MuZero
   has no teacher; it learns from MCTS visit counts on its own
   self-play games (same as AlphaZero). The teacher means we don't
   need MuZero's world model to deliver a strong move-distribution
   target; Stockfish already gives us one for free.

## Where MuZero Would Help (and we ignored it)

MuZero's auxiliary losses anticipated a lot of what KataGo later
formalized. The reward-prediction head in `g_θ` is morally the same
shape as KataGo's score head. The recurrent latent unrolling forces
the network to keep enough internal state to plan multi-step, which
helps even with known-env board games because it forces the encoder
to make planning-useful features explicit.

We don't have any of that. Our network outputs a per-position policy
and a single scalar value; that's it. KataGo's paper documented ~9×
efficiency gains over vanilla AlphaZero from adding aux heads of
exactly this flavor ([see vs-Lc0 page](/vs-leela/)).
[We've plumbed the architecture](https://github.com/shehio/world-models/blob/main/experiments/distill-go/src/distill_go/network.py)
for KataGo-style aux heads on the Go side but haven't enabled them in
a real run yet.

## Is "Distill from Stockfish" Actually MuZero-Adjacent?

Sort of. Both AlphaZero/MuZero ultimately train the value head on
game outcomes (the `z` target) and the policy head on improved
distributions (`π` from MCTS). When we use Stockfish multipv as the
policy target, we're using a *much stronger* `π` than MCTS would
produce at the same network strength; Stockfish at depth 15 sees
much further than 800 MCTS sims of our smallish ResNet. So our policy
training signal is structurally similar but bandwidth-rich.

What we **don't** get from Stockfish-distillation: experience with
the network's *own* search distribution. MCTS's blind spots (where
priors disagree with deep MCTS) are exactly the positions where
self-play would help most, and exactly the positions Stockfish's
labels can't directly inform. That's the residual gap between
distill ceiling (2,300-ish) and Lc0's 3,600+. The self-play attempts
documented on the [what's next](/next/#selfplay-postmortem) page
target that gap.

## In One Sentence

**MuZero kept AlphaZero's MCTS-at-inference and *learned* the rules
engine; we kept AlphaZero's MCTS-at-inference and *used* the rules
engine plus a Stockfish teacher.** Both are valid moves; ours is the
right one for "single GPU + want a strong chess net fast."
