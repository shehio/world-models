"""AlphaZero-style self-play loop for 9x9 Go, starting from a distilled prior.

Mirrors the chess `experiments/selfplay/scripts/selfplay_loop_mp.py` structure
but specialized for Go (board, MCTS, network all from distill_go.*):

Outer loop, one iteration:
  1. K worker processes each play `games_per_worker` self-play games with
     Playout Cap Randomization (KataGo trick). Workers checkpoint to a
     shared file location; main process collects samples via mp.Queue.
  2. Main process adds the iteration's samples as one shard to the
     ShardedReplayBuffer.
  3. Main process runs `train_steps` SGD steps on (state, π, z) triples
     sampled from the buffer.
  4. Quick eval vs a random-policy opponent (cheap sanity, 10–20 games).
  5. Save ckpt, optionally sync to S3.

PCR (Playout Cap Randomization):
  - With prob p_full: use sims_full sims + Dirichlet noise + RECORD this
    move's policy target.
  - Otherwise: use sims_reduced sims, NO noise, DO NOT record. Move still
    advances the game.
  - z = game outcome is written into every RECORDED position from that
    position's side-to-move POV.

Usage:
    uv run python scripts/selfplay_loop.py \\
        --workers 4 --games-per-worker 4 \\
        --pcr-sims-full 200 --pcr-sims-reduced 50 --pcr-p-full 0.25 \\
        --train-steps 100 --lr 1e-5 \\
        --time-budget 3600 \\
        --n-blocks 8 --n-filters 128 --n-input-planes 4 \\
        --resume s3://wm-chess-library-.../checkpoints/.../distilled_epoch014.pt \\
        --ckpt-dir /work/checkpoints_selfplay \\
        --s3-ckpt-base s3://.../selfplay/9x9-8x128-<run-id>/
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go.board import BLACK, WHITE, EMPTY, GoBoard, board_to_planes, board_to_history_planes
from distill_go.config import GoConfig
from distill_go.mcts import run_mcts, select_move, visits_to_distribution
from distill_go.network import AlphaZeroGoNet, get_device


# ----------------------------------------------------------------------------
# Replay buffer — copied from experiments/selfplay/src/selfplay/replay.py
# (game-agnostic, so no reason to depend on the chess workspace member).
# ----------------------------------------------------------------------------

from collections import deque


class ShardedReplayBuffer:
    """Sliding-window-by-iteration buffer. Each shard = one iter's samples."""

    def __init__(self, max_shards: int = 20):
        self.shards: deque = deque(maxlen=max_shards)
        self._flat_cache: list | None = None

    def add_iteration(self, samples_iter) -> None:
        shard = list(samples_iter)
        self.shards.append(shard)
        self._flat_cache = None

    def _flat(self):
        if self._flat_cache is None:
            self._flat_cache = [s for shard in self.shards for s in shard]
        return self._flat_cache

    def sample(self, batch_size: int, device: torch.device):
        """Return a dict of stacked tensors for one training batch.

        Keys: state, pi, z (always); ownership, score, opp_pi (extras for
        KataGo aux heads). Older samples that were 3-tuples are upcast to
        6-tuples with zero-filled aux fields, so a buffer mixing old +
        new shards still works.
        """
        flat = self._flat()
        if len(flat) == 0:
            raise ValueError("ShardedReplayBuffer.sample called on empty buffer — "
                             "the caller must `if len(buffer) >= batch_size` first")
        idxs = np.random.randint(0, len(flat), size=batch_size)
        batch = [flat[i] for i in idxs]

        # Probe sample width — backward-compat with 3-tuple samples.
        sample_len = len(batch[0])
        if sample_len < 6:
            # Promote a 3-tuple to a 6-tuple with zero aux targets.
            S = batch[0][0].shape[-1]            # state (..., S, S)
            policy_dim = batch[0][1].shape[0]    # pi shape
            zero_own = np.zeros((S, S), dtype=np.int8)
            zero_opp = np.zeros(policy_dim, dtype=np.float32)
            batch = [b + (zero_own, 0.0, zero_opp) for b in batch]

        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        owns = torch.from_numpy(np.stack([b[3] for b in batch])).to(device).long()  # (B, S, S) int64 class labels
        scores = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=device)
        opp_pis = torch.from_numpy(np.stack([b[5] for b in batch])).to(device)
        return {
            "state": states, "pi": pis, "z": zs,
            "ownership": owns, "score": scores, "opp_pi": opp_pis,
        }

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    @property
    def n_shards(self) -> int:
        return len(self.shards)


# ----------------------------------------------------------------------------
# Self-play with PCR
# ----------------------------------------------------------------------------

def _outcome_black_pov(board: GoBoard) -> float:
    """Game result from BLACK's POV: +1 if black wins, -1 if white wins, 0 if no winner.

    BLACK-POV (not white) because BLACK plays first in Go — matches the
    sample-z assignment below where `was_black ? z : -z` converts to
    side-to-move POV.
    """
    w = board.winner()
    if w == BLACK:
        return 1.0
    if w is None:
        return 0.0
    return -1.0


def play_game_pcr(
    network,
    cfg: GoConfig,
    device: torch.device,
    sims_full: int,
    sims_reduced: int,
    p_full: float,
    max_moves: int,
):
    """One self-play game with PCR.

    Returns (samples, z_black_pov, ply, stats), where each sample is a
    6-tuple:
      (state, pi, z, ownership, score, opp_pi)

      state      — (C, S, S) float32, board-encoded planes
      pi         — (S²+1,) float32, MCTS visit-count distribution (the
                   policy improvement target)
      z          — float, game outcome from STM's POV (+1 win, -1 loss, 0 draw)
      ownership  — (S, S) int8, STM-relative final ownership labels
                   (0=empty/dame, 1=own, 2=opp). All zeros if game ended
                   by max_moves cutoff (no reliable scoring).
      score      — float, STM-relative score margin (own_pts − opp_pts).
                   0.0 if game ended by max_moves cutoff.
      opp_pi     — (S²+1,) float32, one-hot at the move the opponent
                   actually played 1 ply after this position. All zeros
                   if no such move exists (game ended before opp moved).

    Aux targets (ownership / score / opp_pi) are filled regardless of
    whether the network has aux heads enabled — train_step only computes
    aux losses if the corresponding network flag is on.
    """
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    # Per-recorded-position scratch: state, pi, was_black_to_move, ply_at_record
    recorded: list[dict] = []
    # Every move played, ply-indexed, for opp_pi lookup.
    moves_played: dict[int, int] = {}
    ply = 0
    full_count = reduced_count = 0
    game_history: list[GoBoard] = []

    while not board.is_game_over and ply < max_moves:
        is_full = random.random() < p_full
        sims = sims_full if is_full else sims_reduced
        add_root_noise = is_full
        if is_full:
            full_count += 1
        else:
            reduced_count += 1

        visits = run_mcts(
            board,
            network,
            num_sims=sims,
            c_puct=cfg.c_puct,
            add_root_noise=add_root_noise,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_eps=cfg.dirichlet_eps,
            device=device,
            n_input_planes=cfg.n_input_planes,
            game_history=game_history,
        )

        if is_full:
            pi = visits_to_distribution(visits, cfg)
            full_history = game_history + [board.copy()]
            if cfg.n_input_planes == 17:
                state = board_to_history_planes(full_history, n_history=8)
            else:
                state = board_to_planes(board)
            recorded.append({
                "state": state,
                "pi": pi,
                "was_black": board.to_move == BLACK,
                "ply": ply,
            })

        temp = 1.0 if ply < cfg.temp_moves else 0.0
        move = select_move(visits, temperature=temp)
        moves_played[ply] = move
        game_history.append(board.copy())
        if cfg.n_input_planes == 17 and len(game_history) > 8:
            game_history = game_history[-8:]
        board.play(move)
        ply += 1

    z_black = _outcome_black_pov(board)
    game_ended_naturally = board.is_game_over

    # Aux targets: only meaningful if the game ended naturally (two passes).
    # Otherwise (max_moves cutoff), zero them out — torn games confuse the score head.
    if game_ended_naturally:
        own_abs = board.ownership_map()                  # (S, S) int8, BLACK / WHITE / EMPTY
        bp_pts, wp_pts = board.tromp_taylor_score()
        score_black_pov = bp_pts - wp_pts                # signed margin
    else:
        own_abs = np.zeros((cfg.board_size, cfg.board_size), dtype=np.int8)
        score_black_pov = 0.0

    S = cfg.board_size
    policy_dim = cfg.policy_size
    samples: list[tuple] = []
    for rec in recorded:
        was_black = rec["was_black"]
        z = z_black if was_black else -z_black

        # STM-relative ownership map: 1 = own, 2 = opp, 0 = empty/dame.
        if game_ended_naturally:
            stm_color = BLACK if was_black else WHITE
            opp_color = WHITE if was_black else BLACK
            own_stm = np.zeros((S, S), dtype=np.int8)
            own_stm[own_abs == stm_color] = 1
            own_stm[own_abs == opp_color] = 2
            score = float(score_black_pov if was_black else -score_black_pov)
        else:
            own_stm = np.zeros((S, S), dtype=np.int8)
            score = 0.0

        # Opp-policy target: one-hot at the move that was played 1 ply
        # after this recorded position (= the opponent's response).
        opp_ply = rec["ply"] + 1
        opp_pi = np.zeros(policy_dim, dtype=np.float32)
        if opp_ply in moves_played:
            opp_pi[moves_played[opp_ply]] = 1.0
        # else: opp didn't move (recorded position was the last ply before
        # the game ended), leave all-zeros — train_step's masked CE skips
        # samples whose target sums to 0.

        samples.append((rec["state"], rec["pi"], z, own_stm, score, opp_pi))

    stats = {
        "full_moves": full_count,
        "reduced_moves": reduced_count,
        "recorded_targets": len(samples),
        "game_ended_naturally": game_ended_naturally,
    }
    return samples, z_black, ply, stats


# ----------------------------------------------------------------------------
# Worker process — loads net, plays games, queues samples back to main
# ----------------------------------------------------------------------------

def _worker(args, ckpt_path: str, out_queue, worker_id: int) -> None:
    """Run `games_per_worker` games, push each game's samples to out_queue.

    Always sends a final {"done": True} so the main process never hangs
    on queue.get() — even on exception (the finally block ensures it).
    """
    # Seed RNG distinctly per worker so PCR coin-flips + MCTS Dirichlet
    # noise are independent across workers. Without this, every worker
    # starts from Python's default seed (process start time) and games
    # can correlate within an iter.
    seed = (int(time.time() * 1000) ^ (worker_id * 2654435761)) & 0x7FFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        cfg = GoConfig(
            board_size=args.board_size,
            n_input_planes=args.n_input_planes,
            n_res_blocks=args.n_blocks,
            n_filters=args.n_filters,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            temp_moves=args.temp_moves,
            c_puct=args.c_puct,
            # Forward aux head flags so worker's net matches main's net
            # — otherwise state_dict load fails with "unexpected keys".
            use_global_pool=args.use_global_pool,
            use_aux_ownership=args.use_aux_ownership,
            use_aux_opp_policy=args.use_aux_opp_policy,
        )
        device = torch.device(args.worker_device)
        net = AlphaZeroGoNet(cfg).to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        net.eval()

        for g in range(args.games_per_worker):
            samples, z, ply, stats = play_game_pcr(
                net, cfg, device,
                sims_full=args.pcr_sims_full,
                sims_reduced=args.pcr_sims_reduced,
                p_full=args.pcr_p_full,
                max_moves=args.max_moves,
            )
            out_queue.put({
                "worker": worker_id,
                "game": g,
                "samples": samples,
                "z_black": z,
                "ply": ply,
                "stats": stats,
            })
    except Exception as e:
        # Surface the error so the main process can log it, but still
        # signal done so it doesn't deadlock waiting forever.
        import traceback
        out_queue.put({
            "worker": worker_id,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        })
    finally:
        out_queue.put({"worker": worker_id, "done": True})


# ----------------------------------------------------------------------------
# Train step (generic AZ loss — same as chess)
# ----------------------------------------------------------------------------

def train_step(network, optimizer, batch) -> dict[str, float]:
    """One SGD step with KataGo-style aux losses when the network has them.

    `batch` is a dict from ShardedReplayBuffer.sample(): state, pi, z,
    ownership, score, opp_pi. The aux fields are always present; the
    losses are only added when the corresponding cfg flag is on.

    Loss = policy_CE + value_MSE
         + cfg.aux_ownership_weight  · ownership_CE   (if use_aux_ownership)
         + cfg.aux_score_weight      · score_MSE      (if use_aux_ownership;
                                                       score is part of the
                                                       same head package)
         + cfg.aux_opp_policy_weight · opp_policy_CE  (if use_aux_opp_policy)

    The KataGo paper's default weights (1.5 / 0.02 / 0.15) live in GoConfig.
    """
    cfg = network.cfg
    states = batch["state"]
    network.train()

    # forward_aux computes all heads present on the net.
    out = network.forward_aux(states)
    policy_logits = out["policy"]
    value_pred = out["value"]

    log_probs = F.log_softmax(policy_logits, dim=-1)
    policy_loss = -(batch["pi"] * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, batch["z"])
    loss = policy_loss + value_loss

    stats: dict[str, float] = {}
    stats["policy_loss"] = float(policy_loss.item())
    stats["value_loss"] = float(value_loss.item())

    # Ownership + score aux losses (shared head package).
    if cfg.use_aux_ownership and "ownership" in out:
        # ownership target: (B, S, S) int64 class labels in {0, 1, 2}.
        # ownership logits: (B, 3, S, S).
        own_loss = F.cross_entropy(out["ownership"], batch["ownership"])
        # score head: scalar margin. Targets can be tens of points wide;
        # scale loss down to keep its gradient comparable to value MSE.
        score_loss = F.mse_loss(out["score"], batch["score"])
        loss = (loss
                + cfg.aux_ownership_weight * own_loss
                + cfg.aux_score_weight * score_loss)
        stats["ownership_loss"] = float(own_loss.item())
        stats["score_loss"] = float(score_loss.item())

    # Opp-policy aux loss.
    if cfg.use_aux_opp_policy and "opp_policy" in out:
        # opp_pi target: (B, S²+1) one-hot (or all-zeros if no opp move).
        # Mask out the zero-target rows so they don't contribute a loss
        # (cross-entropy on a zero-sum target is undefined / NaN).
        opp_log_probs = F.log_softmax(out["opp_policy"], dim=-1)
        target_sums = batch["opp_pi"].sum(dim=-1)            # (B,) — 1.0 or 0.0
        mask = (target_sums > 0).float()
        per_sample = -(batch["opp_pi"] * opp_log_probs).sum(dim=-1)  # (B,)
        denom = mask.sum().clamp(min=1.0)
        opp_loss = (per_sample * mask).sum() / denom
        loss = loss + cfg.aux_opp_policy_weight * opp_loss
        stats["opp_policy_loss"] = float(opp_loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    stats["loss"] = float(loss.item())
    return stats


# ----------------------------------------------------------------------------
# Eval vs random (cheap sanity check)
# ----------------------------------------------------------------------------

def eval_vs_random(network, cfg: GoConfig, device: torch.device, games: int,
                   sims: int = 50) -> dict:
    wins = draws = losses = 0
    for g in range(games):
        board = GoBoard(size=cfg.board_size, komi=cfg.komi)
        net_color = BLACK if g % 2 == 0 else 3 - BLACK
        history: list[GoBoard] = []
        ply = 0
        while not board.is_game_over and ply < cfg.max_plies:
            if board.to_move == net_color:
                visits = run_mcts(board, network, num_sims=sims,
                                  c_puct=cfg.c_puct, add_root_noise=False,
                                  device=device, n_input_planes=cfg.n_input_planes,
                                  game_history=history)
                # Defensive: select_move raises on empty visits dict (would
                # happen only if MCTS expanded zero children, which means no
                # legal moves at the root — fall through to pass).
                if visits:
                    move = select_move(visits, temperature=0.0)
                else:
                    move = cfg.pass_move
            else:
                legal = [i for i, x in enumerate(board.legal_mask()) if x]
                move = random.choice(legal) if legal else cfg.pass_move
            history.append(board.copy())
            if cfg.n_input_planes == 17 and len(history) > 8:
                history = history[-8:]
            board.play(move)
            ply += 1
        w = board.winner()
        if w == net_color:
            wins += 1
        elif w is None:
            draws += 1
        else:
            losses += 1
    score = (wins + 0.5 * draws) / games if games else 0.0
    return {"games": games, "wins": wins, "draws": draws, "losses": losses, "score": score}


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--board-size", type=int, default=9)
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--n-filters", type=int, default=128)
    p.add_argument("--n-input-planes", type=int, default=4)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--dirichlet-alpha", type=float, default=0.15)
    p.add_argument("--dirichlet-eps", type=float, default=0.25)
    p.add_argument("--temp-moves", type=int, default=30)
    p.add_argument("--max-moves", type=int, default=200)

    # KataGo aux head flags (off by default to preserve compat with the
    # prior; turn on to enable the efficiency tricks from the KataGo paper).
    p.add_argument("--use-global-pool", action="store_true",
                   help="Use KataGo global-pooling residual blocks (~1.60× speedup).")
    p.add_argument("--use-aux-ownership", action="store_true",
                   help="Aux ownership + score heads (~1.65× speedup).")
    p.add_argument("--use-aux-opp-policy", action="store_true",
                   help="Aux opponent-policy head (~1.30× speedup).")

    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--games-per-worker", type=int, default=4)
    p.add_argument("--worker-device", default="cpu",
                   help="Device for self-play workers (cpu/cuda). CPU is usually fine on 9x9.")

    p.add_argument("--pcr-sims-full", type=int, default=200)
    p.add_argument("--pcr-sims-reduced", type=int, default=50)
    p.add_argument("--pcr-p-full", type=float, default=0.25)

    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-device", default=None,
                   help="Device for the trainer (auto-detected if not set).")

    p.add_argument("--replay-shards", type=int, default=20)

    p.add_argument("--time-budget", type=float, default=3600.0)
    p.add_argument("--max-iters", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-sims", type=int, default=50)

    p.add_argument("--resume", type=str, default=None,
                   help="Path or s3:// uri to a prior .pt checkpoint to start from.")
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--s3-ckpt-base", type=str, default=None,
                   help="If set, sync per-iter ckpts to this S3 prefix.")

    args = p.parse_args()

    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg = GoConfig(
        board_size=args.board_size,
        komi=args.komi,
        n_input_planes=args.n_input_planes,
        n_res_blocks=args.n_blocks,
        n_filters=args.n_filters,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        temp_moves=args.temp_moves,
        max_plies=args.max_moves,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_global_pool=args.use_global_pool,
        use_aux_ownership=args.use_aux_ownership,
        use_aux_opp_policy=args.use_aux_opp_policy,
    )

    train_device = torch.device(args.train_device) if args.train_device else get_device()
    print(f"train device: {train_device}", flush=True)

    # Load (or initialize) prior network.
    net = AlphaZeroGoNet(cfg).to(train_device)
    if args.resume:
        local = args.resume
        if local.startswith("s3://"):
            local = str(args.ckpt_dir / "prior.pt")
            subprocess.run(["aws", "s3", "cp", args.resume, local, "--no-progress"], check=True)
        state = torch.load(local, map_location=train_device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        print(f"resumed prior from {args.resume}", flush=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    buffer = ShardedReplayBuffer(max_shards=args.replay_shards)

    # Save the prior so workers can load it on iter 0 (and we have a baseline ckpt on disk).
    prior_path = args.ckpt_dir / "current.pt"
    torch.save(net.state_dict(), prior_path)
    if args.s3_ckpt_base:
        subprocess.run(["aws", "s3", "cp", str(prior_path),
                        f"{args.s3_ckpt_base.rstrip('/')}/prior.pt", "--no-progress"], check=False)

    # Write run metadata for reproducibility.
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cfg": asdict(cfg),
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.ckpt_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    if args.s3_ckpt_base:
        subprocess.run(["aws", "s3", "cp", str(args.ckpt_dir / "run_metadata.json"),
                        f"{args.s3_ckpt_base.rstrip('/')}/run_metadata.json", "--no-progress"], check=False)

    print(f"PCR ON: full={args.pcr_sims_full} reduced={args.pcr_sims_reduced} p_full={args.pcr_p_full}", flush=True)
    print(f"time budget: {args.time_budget}s  max iters: {args.max_iters}", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")

    total_games = 0
    for iter_idx in range(args.max_iters):
        if time.time() - t0 > args.time_budget:
            print(f"time budget reached after iter {iter_idx}", flush=True)
            break

        # ---- self-play (one iteration) ----
        iter_t0 = time.time()
        queue = ctx.Queue(maxsize=args.workers * args.games_per_worker + args.workers)
        procs = [
            ctx.Process(target=_worker, args=(args, str(prior_path), queue, w),
                        daemon=False)
            for w in range(args.workers)
        ]
        for p_ in procs:
            p_.start()

        iter_samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        done_workers = 0
        n_games = 0
        n_positions = 0
        n_errors = 0
        # Per-game timeout: even if a worker hangs, the main loop won't.
        # An iter at sims_full=200 typically takes seconds, so 1800s is
        # ~30 min of slack — well above expected per-iter wallclock.
        queue_timeout = max(args.pcr_sims_full * 1.5, 1800)
        while done_workers < args.workers:
            try:
                msg = queue.get(timeout=queue_timeout)
            except Exception:
                print(f"  queue.get timed out after {queue_timeout}s — "
                      f"{done_workers}/{args.workers} workers done; aborting iter",
                      flush=True)
                break
            if msg.get("done"):
                done_workers += 1
                continue
            if "error" in msg:
                n_errors += 1
                print(f"  worker {msg['worker']} error: {msg['error']}", flush=True)
                continue
            iter_samples.extend(msg["samples"])
            n_games += 1
            n_positions += len(msg["samples"])
        for p_ in procs:
            # Don't block forever on a wedged child.
            p_.join(timeout=30)
            if p_.is_alive():
                p_.terminate()
                p_.join(timeout=5)
        if n_errors > 0:
            print(f"  iter had {n_errors} worker errors (continuing with what landed)",
                  flush=True)

        iter_wall = time.time() - iter_t0
        total_games += n_games
        buffer.add_iteration(iter_samples)
        print(f"iter {iter_idx}: {n_games} games ({n_positions} pos) in {iter_wall:.1f}s  "
              f"buffer={len(buffer)}/{buffer.n_shards} shards  total_games={total_games}",
              flush=True)

        # ---- train ----
        if len(buffer) >= args.batch_size:
            train_t0 = time.time()
            losses: dict[str, float] = {}
            for step in range(args.train_steps):
                batch = buffer.sample(args.batch_size, train_device)
                stats = train_step(net, optimizer, batch)
                for k, v in stats.items():
                    losses[k] = losses.get(k, 0.0) + v
            for k in losses:
                losses[k] /= max(args.train_steps, 1)
            train_wall = time.time() - train_t0
            # Print main losses always; aux losses only when present.
            parts = [
                f"loss={losses.get('loss', 0):.3f}",
                f"pol={losses.get('policy_loss', 0):.3f}",
                f"val={losses.get('value_loss', 0):.3f}",
            ]
            for k_label in [("ownership_loss", "own"), ("score_loss", "scr"),
                             ("opp_policy_loss", "opp")]:
                key, lbl = k_label
                if key in losses:
                    parts.append(f"{lbl}={losses[key]:.3f}")
            print(f"  train ({args.train_steps} steps, {train_wall:.1f}s): " + " ".join(parts),
                  flush=True)
        else:
            print(f"  skipping train (buffer={len(buffer)} < batch={args.batch_size})", flush=True)

        # ---- eval vs random (cheap sanity) ----
        if iter_idx % args.eval_every == 0:
            eval_t0 = time.time()
            r = eval_vs_random(net, cfg, train_device, games=args.eval_games,
                                sims=args.eval_sims)
            print(f"  vs random ({time.time()-eval_t0:.1f}s): {r}", flush=True)

        # ---- save ckpt ----
        iter_path = args.ckpt_dir / f"net_iter{iter_idx:03d}.pt"
        torch.save(net.state_dict(), iter_path)
        torch.save(net.state_dict(), prior_path)  # update "current" pointer for next iter's workers

        if args.s3_ckpt_base:
            base = args.s3_ckpt_base.rstrip("/")
            for src, key in [
                (iter_path, f"{base}/net_iter{iter_idx:03d}.pt"),
                (prior_path, f"{base}/current.pt"),
                (args.ckpt_dir / "selfplay_log.txt", f"{base}/selfplay_log.txt"),  # may not exist
            ]:
                if src.exists():
                    subprocess.run(["aws", "s3", "cp", str(src), key, "--no-progress"], check=False)

        elapsed_min = (time.time() - t0) / 60
        pct = (time.time() - t0) / args.time_budget * 100
        print(f"  total elapsed: {elapsed_min:.1f} min  ({pct:.0f}% of budget)  "
              f"lr={args.lr:.2e}", flush=True)

    print(f"done. {total_games} self-play games, {len(buffer)} buffer positions, "
          f"{(time.time()-t0)/60:.1f} min wallclock", flush=True)


if __name__ == "__main__":
    main()
