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
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_go.board import BLACK, GoBoard
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
        flat = self._flat()
        idxs = np.random.randint(0, len(flat), size=batch_size)
        batch = [flat[i] for i in idxs]
        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        pis = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        zs = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=device)
        return states, pis, zs

    def __len__(self) -> int:
        return sum(len(s) for s in self.shards)

    @property
    def n_shards(self) -> int:
        return len(self.shards)


# ----------------------------------------------------------------------------
# Self-play with PCR
# ----------------------------------------------------------------------------

def _outcome_white_pov(board: GoBoard) -> float:
    """Game result from BLACK POV (chess convention): +1 if black wins, -1 if white wins, 0 if no winner."""
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
) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float, int, dict]:
    """One self-play game with PCR. Returns (samples, z_black_pov, ply, stats)."""
    board = GoBoard(size=cfg.board_size, komi=cfg.komi)
    # (state, pi, was_black_to_move)
    recorded: list[tuple[np.ndarray, np.ndarray, bool]] = []
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
            # Encode current state — same as mcts._encode_state
            from distill_go.board import board_to_planes, board_to_history_planes
            full_history = game_history + [board.copy()]
            if cfg.n_input_planes == 17:
                state = board_to_history_planes(full_history, n_history=8)
            else:
                state = board_to_planes(board)
            recorded.append((state, pi, board.to_move == BLACK))

        temp = 1.0 if ply < cfg.temp_moves else 0.0
        move = select_move(visits, temperature=temp)
        game_history.append(board.copy())
        # Cap history to keep memory bounded.
        if cfg.n_input_planes == 17 and len(game_history) > 8:
            game_history = game_history[-8:]
        board.play(move)
        ply += 1

    z_black = _outcome_white_pov(board)
    samples = [(s, p, z_black if was_black else -z_black) for s, p, was_black in recorded]
    stats = {
        "full_moves": full_count,
        "reduced_moves": reduced_count,
        "recorded_targets": len(samples),
    }
    return samples, z_black, ply, stats


# ----------------------------------------------------------------------------
# Worker process — loads net, plays games, queues samples back to main
# ----------------------------------------------------------------------------

def _worker(args, ckpt_path: str, out_queue, worker_id: int) -> None:
    """Run `games_per_worker` games, push each game's samples to out_queue."""
    cfg = GoConfig(
        board_size=args.board_size,
        n_input_planes=args.n_input_planes,
        n_res_blocks=args.n_blocks,
        n_filters=args.n_filters,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        temp_moves=args.temp_moves,
        c_puct=args.c_puct,
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
    out_queue.put({"worker": worker_id, "done": True})


# ----------------------------------------------------------------------------
# Train step (generic AZ loss — same as chess)
# ----------------------------------------------------------------------------

def train_step(network, optimizer, batch) -> dict[str, float]:
    states, pi_targets, z_targets = batch
    network.train()
    logits, value_pred = network(states)
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi_targets * log_probs).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred, z_targets)
    loss = policy_loss + value_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item())}


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
                move = select_move(visits, temperature=0.0)
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
        while done_workers < args.workers:
            msg = queue.get()
            if msg.get("done"):
                done_workers += 1
                continue
            iter_samples.extend(msg["samples"])
            n_games += 1
            n_positions += len(msg["samples"])
        for p_ in procs:
            p_.join()

        iter_wall = time.time() - iter_t0
        total_games += n_games
        buffer.add_iteration(iter_samples)
        print(f"iter {iter_idx}: {n_games} games ({n_positions} pos) in {iter_wall:.1f}s  "
              f"buffer={len(buffer)}/{buffer.n_shards} shards  total_games={total_games}",
              flush=True)

        # ---- train ----
        if len(buffer) >= args.batch_size:
            train_t0 = time.time()
            losses = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}
            for step in range(args.train_steps):
                batch = buffer.sample(args.batch_size, train_device)
                stats = train_step(net, optimizer, batch)
                for k in losses:
                    losses[k] += stats[k]
            for k in losses:
                losses[k] /= max(args.train_steps, 1)
            train_wall = time.time() - train_t0
            print(f"  train ({args.train_steps} steps, {train_wall:.1f}s): "
                  f"loss={losses['loss']:.3f} pol={losses['policy_loss']:.3f} val={losses['value_loss']:.3f}",
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
