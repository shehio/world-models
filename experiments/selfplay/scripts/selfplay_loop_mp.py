"""Multi-process AlphaZero self-play loop.

Architecture (synchronous actor-learner):

    each iter:
        1. trainer writes the current network to checkpoints/current.pt
        2. a Pool of N worker processes is dispatched:
             - each worker loads checkpoints/current.pt
             - each worker plays `games_per_worker` games (sequentially within itself)
             - each worker returns its sample list
        3. trainer aggregates samples from all workers into the replay buffer
        4. trainer runs `train-steps` SGD steps
        5. trainer evaluates and saves an iter checkpoint
        6. repeat

Why synchronous vs async:
  - Simpler. No worker-divergence issues, no replay-corruption races.
  - Workers idle slightly while trainer trains, but training is fast (~30s)
    relative to self-play (~minutes). Acceptable cost.

Why a Pool (vs forked-on-demand):
  - Workers persist across iters; the per-process startup cost (5-10s of
    PyTorch + chess imports) is paid once.
  - We use 'spawn' start method (the macOS default) so each worker is a
    clean subprocess, not a fork. Safer for PyTorch.

Per-worker setup:
  - torch.set_num_threads(1) so 6 workers don't each spawn 6 threads
    and contend for cores.
  - Each worker pins to CPU (no MPS/CUDA inside workers — the network
    is small enough that batch-1 inference is faster on CPU anyway).

Run from project root:
    uv run python scripts/selfplay_loop_mp.py \
        --workers 6 --games-per-worker 4 --sims 400 --batch-size 8 \
        --n-blocks 10 --n-filters 128 --train-steps 300 --time-budget 28800
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

# Make src/ importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# IMPORTANT: set this BEFORE importing torch, so workers inherit it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.multiprocessing as mp

from wm_chess.arena import (
    network_policy,
    play_match,
    random_policy,
    stockfish_engine,
    stockfish_policy,
)
from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet
from selfplay.ckpt import load_net_state_dict
from selfplay.replay import ShardedReplayBuffer
from selfplay.selfplay import play_game, play_game_pcr
from selfplay.train import train_step

# Single source of truth for hyperparameter defaults — see wm_chess/config.py.
_DEFAULTS = Config()


def _load_net_weights(path: str, map_location):
    """Load just the network state_dict from either format.

    Legacy / distilled-supervised checkpoints: a raw state_dict.
    Selfplay checkpoints: {"net": state_dict, "opt": ..., "iter": ..., "hour": ...}
    """
    return load_net_state_dict(torch.load(path, map_location=map_location))


# Logistic Elo model: a 400-point gap == a 10x odds ratio (FIDE/Elo convention).
_ELO_SCALE = 400.0
# Clamp the score into (0, 1) so the logit stays finite at a clean sweep/shutout.
_SCORE_CLAMP_EPS = 1e-4


def _elo_from_score(score: float, anchor_elo: int) -> float:
    """Convert a match score vs a fixed-Elo opponent into an absolute Elo estimate."""
    s = min(max(score, _SCORE_CLAMP_EPS), 1.0 - _SCORE_CLAMP_EPS)
    return anchor_elo - _ELO_SCALE * math.log10(1.0 / s - 1.0)


def match_worker(args: tuple) -> dict:
    """Worker: load two nets from disk, play n_games (alternating colors), return A's W/D/L."""
    ckpt_a, ckpt_b, n_games, cfg, sims, batch_size, n_history, seed = args
    torch.set_num_threads(1)
    import random as _r
    import numpy as _np
    torch.manual_seed(seed)
    _np.random.seed(seed)
    _r.seed(seed)
    dev = torch.device("cpu")
    na = AlphaZeroNet(cfg); na.load_state_dict(_load_net_weights(ckpt_a, "cpu")); na.eval()
    nb = AlphaZeroNet(cfg); nb.load_state_dict(_load_net_weights(ckpt_b, "cpu")); nb.eval()
    pa = network_policy(na, cfg, dev, sims=sims, batch_size=batch_size,
                        n_history=n_history, add_root_noise=True)
    pb = network_policy(nb, cfg, dev, sims=sims, batch_size=batch_size,
                        n_history=n_history, add_root_noise=True)
    return play_match(pa, pb, n_games=n_games, max_plies=cfg.max_plies)


def sf_match_worker(args: tuple) -> dict:
    """Worker: load a net + open its own Stockfish, play n_games, return the net's W/D/L."""
    ckpt, elo, n_games, cfg, sims, depth, batch_size, n_history, seed = args
    torch.set_num_threads(1)
    import random as _r
    import numpy as _np
    torch.manual_seed(seed)
    _np.random.seed(seed)
    _r.seed(seed)
    dev = torch.device("cpu")
    net = AlphaZeroNet(cfg); net.load_state_dict(_load_net_weights(ckpt, "cpu")); net.eval()
    pol = network_policy(net, cfg, dev, sims=sims, batch_size=batch_size, n_history=n_history)
    with stockfish_engine(elo=elo) as eng:
        return play_match(pol, stockfish_policy(eng, depth=depth),
                          n_games=n_games, max_plies=cfg.max_plies)


def _split_games(n_games: int, workers: int) -> list:
    """Spread n_games across workers as evenly as possible."""
    base, rem = divmod(n_games, workers)
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _agg_match(results: list) -> dict:
    """Sum per-worker W/D/L into one match result (A's POV)."""
    w = sum(r["wins"] for r in results)
    d = sum(r["draws"] for r in results)
    l = sum(r["losses"] for r in results)
    g = w + d + l
    return {"games": g, "wins": w, "draws": d, "losses": l,
            "score": (w + 0.5 * d) / g if g else 0.0}


def parallel_match(pool, ckpt_a: str, ckpt_b: str, n_games: int, workers: int,
                   cfg, sims: int, batch_size: int, n_history: int) -> dict:
    """A-vs-B match fanned out across the worker pool (vs. one game at a time in the trainer)."""
    args = [(ckpt_a, ckpt_b, c, cfg, sims, batch_size, n_history, 1000 + i)
            for i, c in enumerate(_split_games(n_games, workers)) if c > 0]
    return _agg_match(pool.map(match_worker, args))


def parallel_sf_eval(pool, ckpt: str, elos, n_games: int, workers: int, cfg,
                     sims: int, depth: int, batch_size: int, n_history: int) -> None:
    """In-loop Stockfish-panel yardstick, fanned out across the worker pool.

    Same anchor method as the definitive distill-soft eval (fewer games, noisier).
    Skips silently if stockfish isn't on PATH so the local CPU smoke stays runnable.
    """
    if shutil.which("stockfish") is None:
        print("  [sf-eval] stockfish not on PATH — skipping yardstick", flush=True)
        return
    for elo in elos:
        args = [(ckpt, elo, c, cfg, sims, depth, batch_size, n_history, 2000 + i)
                for i, c in enumerate(_split_games(n_games, workers)) if c > 0]
        try:
            stats = _agg_match(pool.map(sf_match_worker, args))
        except Exception as e:  # a missing/flaky engine shouldn't kill training
            print(f"  [sf-eval] vs SF {elo} failed: {e}", flush=True)
            continue
        est = _elo_from_score(stats["score"], elo)
        print(f"  [sf-eval] vs SF {elo}: score={stats['score']:.3f} "
              f"W{stats['wins']} D{stats['draws']} L{stats['losses']} → ~{est:.0f} Elo",
              flush=True)


def selfplay_worker(args: tuple) -> list:
    """Worker process: load latest checkpoint, play games_per_worker games, return samples."""
    (worker_id, ckpt_path, games_per_worker, cfg, sims, batch_size,
     use_pcr, sims_full, sims_reduced, p_full, n_history) = args

    # Inside the worker, force single-threaded torch so N workers don't
    # collectively oversubscribe CPU cores.
    torch.set_num_threads(1)

    network = AlphaZeroNet(cfg)
    network.load_state_dict(_load_net_weights(ckpt_path, "cpu"))
    network.eval()

    device = torch.device("cpu")
    cfg_local = replace(cfg, sims_train=sims)

    results = []
    for g in range(games_per_worker):
        if use_pcr:
            samples, z, ply, pcr_stats = play_game_pcr(
                network, cfg_local, device,
                sims_full=sims_full, sims_reduced=sims_reduced,
                p_full=p_full, batch_size=batch_size,
                n_history=n_history,
            )
        else:
            pcr_stats = None
            samples, z, ply = play_game(network, cfg_local, device,
                                        sims=sims, batch_size=batch_size,
                                        n_history=n_history)
        results.append({
            "worker_id": worker_id,
            "game_idx": g,
            "samples": samples,
            "z": z,
            "ply": ply,
            "pcr_stats": pcr_stats,
        })
    return results


def main():
    p = argparse.ArgumentParser()
    # Loop / time budget
    p.add_argument("--time-budget", type=int, default=28800,
                   help="seconds; quit after this many seconds. 28800 = 8h.")
    p.add_argument("--max-iters", type=int, default=10000,
                   help="hard cap on iters as backup to time budget.")
    # Self-play
    p.add_argument("--workers", type=int, default=6, help="# worker processes")
    p.add_argument("--games-per-worker", type=int, default=4,
                   help="# games each worker plays per iter")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-plies", type=int, default=200)
    # Playout Cap Randomization (KataGo). When --pcr is set, --sims is ignored
    # and per-move sims are drawn from {sims_full with prob p_full, sims_reduced
    # with prob 1-p_full}. Only full-sim moves contribute training samples.
    p.add_argument("--pcr", action="store_true",
                   help="enable Playout Cap Randomization (KataGo)")
    p.add_argument("--pcr-sims-full", type=int, default=400,
                   help="MCTS sims on the 'full' fraction of moves (target-recording)")
    p.add_argument("--pcr-sims-reduced", type=int, default=80,
                   help="MCTS sims on the 'reduced' fraction (play only, no target)")
    p.add_argument("--pcr-p-full", type=float, default=0.25,
                   help="probability a move uses full sims (KataGo default 0.25)")
    # Network
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--n-filters", type=int, default=128)
    # Training
    p.add_argument("--train-steps", type=int, default=40,
                   help="SGD steps per iter. KataGo target ~4 samples-seen / new sample.")
    # LR default lowered from 1e-3 → 1e-5 after the first 12h run regressed
    # ~700 Elo. The loop initializes from a *pretrained* distilled prior, so
    # Adam at 1e-3 immediately sprints out of the Stockfish-mimic basin.
    # KataGo's continual-RL recipe is 1e-4 → 1e-5; this is at the gentle end
    # of that range. AlphaZero's 1e-2 figure was SGD+momentum from scratch,
    # not Adam from a pretrained net — not comparable.
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--replay-shards", type=int, default=20,
                   help="how many recent iters to keep in the sharded replay (suragnair default 20)")
    # Optimizer (paper-faithful options for AlphaZero)
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                   help="adam (educational default) or sgd+momentum (AZ paper)")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="momentum for SGD (ignored if --optimizer adam)")
    p.add_argument("--lr-decay-iters", type=str, default="",
                   help="comma-separated iter boundaries to decay LR. e.g. '20,40' decays at iter 20 and 40. AZ paper used step decay.")
    p.add_argument("--lr-decay-factor", type=float, default=0.1,
                   help="multiplicative factor at each LR decay boundary (AZ paper used 0.1)")
    p.add_argument("--lr-schedule", choices=["step", "cosine"], default="step",
                   help="step (uses --lr-decay-iters/factor) or cosine (smooth from --lr to --lr-min over --max-iters).")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="floor LR for cosine schedule (paper trick: KataGo uses ~1e-5)")
    # 8-step history (paper-faithful input encoding)
    p.add_argument("--n-history", type=int, default=1,
                   help="positions stacked in input encoding. 1 = legacy 19-plane; 8 = AZ-paper 103-plane.")
    # Trainer device
    p.add_argument("--train-device", choices=["cpu", "mps", "auto"], default="auto",
                   help="device for the trainer's SGD step. 'auto' picks MPS if available.")
    # Hourly model dumps (for evolution tracking)
    p.add_argument("--hourly-dump", action="store_true",
                   help="save net_hour_NNN.pt every wall-clock hour for later eval-vs-time analysis.")
    # Resume bookkeeping: pick up cosine-LR progress, iter filenames, and hourly
    # numbering where the previous run left off. The model weights resume via
    # --resume; these flags restore the loop's index state.
    p.add_argument("--start-iter", type=int, default=0,
                   help="logical iter number for the first iteration of this run. Used for cosine LR progress and iter checkpoint filenames.")
    p.add_argument("--start-hour", type=int, default=0,
                   help="starting index for net_hour_NNN.pt filenames when resuming an hourly-dump run.")
    # Eval
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-sims", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=2)
    # Arena gating (AlphaGo Zero evaluator). Defaults live in wm_chess/config.py.
    # gate-every=0 disables it (workers self-play from the latest net, no gate).
    p.add_argument("--gate-every", type=int, default=_DEFAULTS.gate_every,
                   help="run a candidate-vs-champion gate every N iters; 0 = no gating.")
    p.add_argument("--gate-games", type=int, default=_DEFAULTS.gate_games,
                   help="games in the gating match")
    p.add_argument("--gate-sims", type=int, default=_DEFAULTS.gate_sims,
                   help="MCTS sims per move during the gating match")
    p.add_argument("--gate-threshold", type=float, default=_DEFAULTS.gate_threshold,
                   help="promote the candidate only if its score vs champion >= this")
    # KL-anchor to the distilled teacher (trust region for warm-started RL).
    p.add_argument("--kl-beta", type=float, default=_DEFAULTS.kl_beta,
                   help="weight on KL(policy||teacher); 0 = off")
    p.add_argument("--teacher-ckpt", default="",
                   help="checkpoint for the frozen KL-anchor teacher (defaults to --resume weights)")
    # In-loop Stockfish yardstick (runs on each promotion when gating is on).
    p.add_argument("--sf-eval-elos", default="",
                   help="comma-separated Stockfish UCI Elos to eval promoted champions vs; empty = off")
    p.add_argument("--sf-eval-games", type=int, default=_DEFAULTS.sf_eval_games)
    p.add_argument("--sf-eval-sims", type=int, default=_DEFAULTS.sf_eval_sims)
    p.add_argument("--sf-eval-depth", type=int, default=_DEFAULTS.sf_eval_depth)
    # I/O
    p.add_argument("--ckpt-dir", default="checkpoints_mp")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    # Use 'spawn' for cross-platform compat and to avoid forking PyTorch state.
    mp.set_start_method("spawn", force=True)

    # n_input_planes depends on --n-history: 1 → 19, 8 → 103.
    if args.n_history > 1:
        from wm_chess.board import N_HISTORY, N_PIECE_PLANES_PER_BOARD, N_META_PLANES
        n_input_planes = args.n_history * N_PIECE_PLANES_PER_BOARD + N_META_PLANES
    else:
        n_input_planes = 19

    cfg = replace(Config(),
                  sims_train=args.sims, sims_eval=args.eval_sims,
                  max_plies=args.max_plies,
                  n_input_planes=n_input_planes,
                  n_res_blocks=args.n_blocks, n_filters=args.n_filters,
                  lr=args.lr)  # training batch size comes from Config.batch_size

    os.makedirs(args.ckpt_dir, exist_ok=True)
    current_ckpt = os.path.join(args.ckpt_dir, "current.pt")
    state_ckpt = os.path.join(args.ckpt_dir, "state.pt")  # full {net,opt,iter,hour} for resume

    # --- Trainer-side setup ---
    if args.train_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.train_device)
    print(f"trainer device: {device}", flush=True)
    network = AlphaZeroNet(cfg).to(device)

    # Load network weights from --resume. The blob may be legacy (raw state_dict)
    # or new format ({"net", "opt", "iter", "hour"}). We extract the net here and
    # defer optimizer restoration until after the optimizer object exists.
    resume_blob = None
    if args.resume:
        blob = torch.load(args.resume, map_location=device)
        if isinstance(blob, dict) and "net" in blob and isinstance(blob["net"], dict):
            network.load_state_dict(blob["net"])
            resume_blob = blob
            has_opt = "opt" in blob
            print(f"resumed from {args.resume} (full state: opt={'yes' if has_opt else 'no'}, "
                  f"iter={blob.get('iter', '-')}, hour={blob.get('hour', '-')})", flush=True)
        else:
            network.load_state_dict(blob)
            print(f"resumed from {args.resume} (legacy weights-only)", flush=True)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=cfg.lr,
            momentum=args.momentum,
            weight_decay=cfg.weight_decay,
        )
        print(f"optimizer: SGD lr={cfg.lr} momentum={args.momentum} weight_decay={cfg.weight_decay}", flush=True)
    else:
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        print(f"optimizer: Adam lr={cfg.lr} weight_decay={cfg.weight_decay}", flush=True)

    # Restore optimizer state and auto-fill start-iter/start-hour from the checkpoint
    # if the CLI didn't override them.
    if resume_blob is not None:
        if "opt" in resume_blob:
            try:
                optimizer.load_state_dict(resume_blob["opt"])
                print(f"  restored optimizer state (momentum buffers preserved)", flush=True)
            except Exception as e:
                print(f"  WARN: could not restore optimizer state ({e}); momentum will re-warm", flush=True)
        if args.start_iter == 0 and "iter" in resume_blob:
            args.start_iter = int(resume_blob["iter"]) + 1
            print(f"  auto-set --start-iter {args.start_iter} from checkpoint", flush=True)
        if args.start_hour == 0 and "hour" in resume_blob:
            args.start_hour = int(resume_blob["hour"])
            print(f"  auto-set --start-hour {args.start_hour} from checkpoint", flush=True)

    # Parse LR decay boundaries — list of iters at which to multiply lr by lr_decay_factor.
    lr_decay_iters = (
        sorted(int(x) for x in args.lr_decay_iters.split(",") if x.strip())
        if args.lr_decay_iters else []
    )
    if args.lr_schedule == "cosine":
        print(f"LR cosine schedule: {args.lr:.2e} → {args.lr_min:.2e} over {args.max_iters} iters", flush=True)
    elif lr_decay_iters:
        print(f"LR step-decay schedule: at iters {lr_decay_iters}, lr *= {args.lr_decay_factor}", flush=True)
    # Sliding-window-by-iteration buffer. We DROP the previous run's replay
    # entirely because it's poisoned with iters 0-5 near-random play.
    replay = ShardedReplayBuffer(max_shards=args.replay_shards)

    n_params = sum(p.numel() for p in network.parameters())
    print(f"network: {args.n_blocks} blocks × {args.n_filters} ch = {n_params:,} params",
          flush=True)
    print(f"workers: {args.workers}  games/worker: {args.games_per_worker}  "
          f"sims: {args.sims}  batch K: {args.batch_size}",
          flush=True)
    if args.pcr:
        print(f"PCR ON: full={args.pcr_sims_full} reduced={args.pcr_sims_reduced} "
              f"p_full={args.pcr_p_full}", flush=True)
    print(f"time budget: {args.time_budget}s  max iters: {args.max_iters}",
          flush=True)

    # KL-anchor teacher: a frozen snapshot of the distilled prior. Built only
    # when --kl-beta > 0. Defaults to --teacher-ckpt, else the weights we just
    # loaded (fine for a fresh run that starts from the prior).
    teacher_net = None
    if args.kl_beta > 0.0:
        teacher_net = AlphaZeroNet(cfg).to(device)
        t_path = args.teacher_ckpt or args.resume
        if t_path:
            teacher_net.load_state_dict(_load_net_weights(t_path, device))
        else:
            teacher_net.load_state_dict(network.state_dict())
        teacher_net.eval()
        for pr in teacher_net.parameters():
            pr.requires_grad_(False)
        print(f"KL-anchor ON: beta={args.kl_beta} teacher={t_path or 'initial weights'}",
              flush=True)

    # Arena gating: workers self-play from the frozen champion; the trainer
    # improves a candidate (`network`); every --gate-every iters the candidate
    # must beat the champion to be promoted. gate-every=0 disables all of this.
    gating = args.gate_every > 0
    champion_net = None
    champion_ckpt = os.path.join(args.ckpt_dir, "champion.pt")
    sf_eval_elos = [int(x) for x in args.sf_eval_elos.split(",") if x.strip()]
    if gating:
        champion_net = AlphaZeroNet(cfg).to(device)
        champion_net.load_state_dict(network.state_dict())
        champion_net.eval()
        torch.save({k: v.cpu() for k, v in champion_net.state_dict().items()}, champion_ckpt)
        print(f"GATING ON: every {args.gate_every} iters, promote if candidate scores "
              f">= {args.gate_threshold:.0%} over {args.gate_games} games @ sims={args.gate_sims}",
              flush=True)

    # Save initial checkpoint so workers have something to load on iter 0.
    torch.save(network.state_dict(), current_ckpt)
    # Workers self-play from the champion when gating, else from the latest net.
    worker_ckpt = champion_ckpt if gating else current_ckpt

    # Launch persistent worker pool. Workers inherit nothing important
    # because we use spawn; they import alphazero fresh.
    pool = mp.Pool(args.workers)

    # --- Main loop ---
    start = time.time()
    total_games = 0
    next_hourly_save = start + 3600  # save first hourly checkpoint at t = +1h
    hourly_idx = args.start_hour
    if args.start_iter:
        print(f"resuming at logical iter {args.start_iter} (cosine progress {args.start_iter / max(1, args.max_iters):.3f})",
              flush=True)

    try:
        for offset in range(args.max_iters - args.start_iter):
            it = args.start_iter + offset
            elapsed = time.time() - start
            if elapsed >= args.time_budget:
                print(f"time budget reached at iter {it} ({elapsed:.0f}s)", flush=True)
                break

            # Apply LR schedule.
            if args.lr_schedule == "cosine":
                # Cosine from args.lr → args.lr_min over the full max_iters horizon.
                progress = min(it / max(1, args.max_iters), 1.0)
                cur_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = cur_lr
            elif it in lr_decay_iters:
                for pg in optimizer.param_groups:
                    pg["lr"] *= args.lr_decay_factor
                print(f"  LR decayed to {optimizer.param_groups[0]['lr']:.2e} at iter {it}", flush=True)

            # Step 1: write the latest candidate for workers — but only when NOT
            # gating. Under gating, workers self-play from the frozen champion
            # (champion_ckpt), which changes only on promotion.
            if not gating:
                torch.save({k: v.cpu() for k, v in network.state_dict().items()}, current_ckpt)

            # Step 2: dispatch games_per_worker games to each of N workers.
            t0 = time.time()
            worker_args = [
                (i, worker_ckpt, args.games_per_worker,
                 cfg, args.sims, args.batch_size,
                 args.pcr, args.pcr_sims_full, args.pcr_sims_reduced, args.pcr_p_full,
                 args.n_history)
                for i in range(args.workers)
            ]
            all_results = pool.map(selfplay_worker, worker_args)
            sp_dt = time.time() - t0

            # Step 3: aggregate ALL of this iter's samples into ONE shard.
            # That way, when shard count exceeds max_shards, this whole
            # iter's data drops as a unit — keeping the buffer fresh.
            iter_samples = []
            iter_games = 0
            for results in all_results:
                for r in results:
                    iter_samples.extend(r["samples"])
                    total_games += 1
                    iter_games += 1
            replay.add_iteration(iter_samples)
            print(f"iter {it}: {iter_games} games ({len(iter_samples)} pos) "
                  f"in {sp_dt:.1f}s  "
                  f"buffer={len(replay)}/{replay.n_shards} shards  "
                  f"total_games={total_games}",
                  flush=True)

            # Step 4: train.
            if len(replay) >= cfg.batch_size:
                t0 = time.time()
                loss_acc = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "kl": 0.0}
                for _ in range(args.train_steps):
                    batch = replay.sample(cfg.batch_size, device)
                    losses = train_step(network, optimizer, batch,
                                        teacher_net=teacher_net, kl_beta=args.kl_beta)
                    for k, v in losses.items():
                        loss_acc[k] += v
                tr_dt = time.time() - t0
                n = args.train_steps
                print(f"  train ({n} steps, {tr_dt:.1f}s): "
                      f"loss={loss_acc['loss']/n:.3f} "
                      f"pol={loss_acc['policy_loss']/n:.3f} "
                      f"val={loss_acc['value_loss']/n:.3f} "
                      f"kl={loss_acc['kl']/n:.3f}",
                      flush=True)

            # Save full state (net + optimizer + counters) for crash-safe resume.
            # Written every iter so a restart loses at most one iter of progress.
            torch.save({
                "net": {k: v.cpu() for k, v in network.state_dict().items()},
                "opt": optimizer.state_dict(),
                "iter": it,
                "hour": hourly_idx,
            }, state_ckpt)

            # Step 5: gating (if on) else the original vs-random eval. The gate
            # match + Stockfish eval fan out across the worker pool (Step 2's
            # processes) — playing one game at a time in the trainer wasted 14 cores.
            if gating and (it + 1) % args.gate_every == 0:
                t0 = time.time()
                candidate_ckpt = os.path.join(args.ckpt_dir, "candidate.pt")
                torch.save({k: v.cpu() for k, v in network.state_dict().items()}, candidate_ckpt)
                gstats = parallel_match(pool, candidate_ckpt, champion_ckpt,
                                        args.gate_games, args.workers, cfg,
                                        args.gate_sims, args.batch_size, args.n_history)
                g_dt = time.time() - t0
                if gstats["score"] >= args.gate_threshold:
                    champion_net.load_state_dict(network.state_dict())
                    cpu_state = {k: v.cpu() for k, v in champion_net.state_dict().items()}
                    torch.save(cpu_state, champion_ckpt)
                    ckpt_path = os.path.join(args.ckpt_dir, f"net_iter{it:03d}.pt")
                    torch.save(cpu_state, ckpt_path)
                    print(f"  GATE iter {it} ({g_dt:.1f}s): candidate {gstats['score']:.3f} "
                          f"(W{gstats['wins']} D{gstats['draws']} L{gstats['losses']}) "
                          f">= {args.gate_threshold:.0%} → PROMOTED, saved {ckpt_path}", flush=True)
                    if sf_eval_elos:
                        parallel_sf_eval(pool, champion_ckpt, sf_eval_elos,
                                         args.sf_eval_games, args.workers, cfg,
                                         args.sf_eval_sims, args.sf_eval_depth,
                                         args.batch_size, args.n_history)
                else:
                    # Reject: roll the candidate back to the champion. Optimizer
                    # momentum re-warms against the reverted weights (acceptable).
                    network.load_state_dict(champion_net.state_dict())
                    print(f"  GATE iter {it} ({g_dt:.1f}s): candidate {gstats['score']:.3f} "
                          f"(W{gstats['wins']} D{gstats['draws']} L{gstats['losses']}) "
                          f"< {args.gate_threshold:.0%} → REJECTED, reverted to champion", flush=True)
            elif not gating and (it + 1) % args.eval_every == 0:
                t0 = time.time()
                net_pol = network_policy(network, cfg, device,
                                         sims=args.eval_sims,
                                         batch_size=args.batch_size,
                                         n_history=args.n_history)
                stats = play_match(net_pol, random_policy,
                                   n_games=args.eval_games,
                                   max_plies=args.max_plies)
                ev_dt = time.time() - t0
                print(f"  vs random ({ev_dt:.1f}s): {stats}", flush=True)

                ckpt_path = os.path.join(args.ckpt_dir, f"net_iter{it:03d}.pt")
                torch.save({k: v.cpu() for k, v in network.state_dict().items()}, ckpt_path)
                print(f"  saved {ckpt_path}", flush=True)

            # Step 6: hourly checkpoint dump (for evolution tracking across the whole run).
            if args.hourly_dump and time.time() >= next_hourly_save:
                hourly_path = os.path.join(args.ckpt_dir, f"net_hour_{hourly_idx:03d}.pt")
                torch.save({k: v.cpu() for k, v in network.state_dict().items()}, hourly_path)
                print(f"  saved {hourly_path} (hourly @ iter {it})", flush=True)
                hourly_idx += 1
                next_hourly_save += 3600  # next at +1h from this one (not from now)

            print(f"  total elapsed: {(time.time()-start)/60:.1f} min  "
                  f"({(time.time()-start)/args.time_budget*100:.0f}% of budget)  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}",
                  flush=True)
    finally:
        # Always shut workers down cleanly.
        pool.close()
        pool.join()
        # Final checkpoint: full state so future runs can resume the optimizer.
        final_path = os.path.join(args.ckpt_dir, "final.pt")
        torch.save({
            "net": {k: v.cpu() for k, v in network.state_dict().items()},
            "opt": optimizer.state_dict(),
            "iter": locals().get("it", args.start_iter),
            "hour": hourly_idx,
        }, final_path)
        print(f"saved final: {final_path}", flush=True)
        print(f"total wall: {(time.time()-start)/60:.1f} min  "
              f"total games: {total_games}",
              flush=True)


if __name__ == "__main__":
    main()
