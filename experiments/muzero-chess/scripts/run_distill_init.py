"""Cloud entry point for distill-init MuZero.

Loads the distilled teacher → factors into frozen h + f → generates teacher
self-play transitions → trains ONLY the dynamics g → periodically evals the
composed (h, g, f) net vs Stockfish and checkpoints + S3-syncs.

Success criterion: the composed net's Elo at sims=800 lands within ~1 sigma
of the teacher's known 2,055 @ sims=800 — i.e. the learned dynamics preserves
teacher strength under MCTS.

Invoked by infra-eks/entrypoint-muzero-distill-init.sh; hparams from env.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time

import torch

from muzero_chess.config import distill_init_config
from muzero_chess.distill_dynamics import (
    TransitionBuffer,
    distill_step,
    generate_transitions,
)
from muzero_chess.eval import evaluate_vs_stockfish
from muzero_chess.networks import DynamicsNet
from muzero_chess.teacher import (
    DistillInitMuZeroNet,
    TeacherPrediction,
    TeacherRepresentation,
    load_teacher,
)


def _env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def _s3_sync(local_dir: str, s3_uri: str) -> None:
    try:
        subprocess.run(["aws", "s3", "sync", local_dir, s3_uri, "--no-progress"],
                       check=False, timeout=300)
    except subprocess.TimeoutExpired:
        print(f"[s3] sync timed out: {local_dir} -> {s3_uri}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-ckpt", default=_env("TEACHER_CKPT", "/work/teacher.pt"))
    p.add_argument("--teacher-ckpt-s3", default=_env("TEACHER_CKPT_S3", ""))
    p.add_argument("--teacher-n-blocks", type=int, default=_env("TEACHER_N_BLOCKS", 20, int))
    p.add_argument("--teacher-n-filters", type=int, default=_env("TEACHER_N_FILTERS", 256, int))
    p.add_argument("--sims", type=int, default=_env("SIMS", 800, int))
    p.add_argument("--mcts-batch-size", type=int, default=_env("MCTS_BATCH_SIZE", 8, int))
    p.add_argument("--mcts-top-k", type=int, default=_env("MCTS_TOP_K", 32, int))
    p.add_argument("--unroll", type=int, default=_env("UNROLL", 5, int))
    p.add_argument("--n-games", type=int, default=_env("N_GAMES", 2000, int),
                   help="teacher self-play games for the transition dataset")
    p.add_argument("--rounds", type=int, default=_env("ROUNDS", 10, int))
    p.add_argument("--train-steps-per-round", type=int, default=_env("TRAIN_STEPS_PER_ROUND", 500, int))
    p.add_argument("--batch-size", type=int, default=_env("BATCH_SIZE", 256, int))
    p.add_argument("--lr", type=float, default=_env("LR", 1e-3, float))
    p.add_argument("--epsilon-random", type=float, default=_env("EPSILON_RANDOM", 0.1, float))
    p.add_argument("--eval-games", type=int, default=_env("EVAL_GAMES", 30, int))
    p.add_argument("--eval-elos", default=_env("EVAL_ELOS", "1350,1800"))
    p.add_argument("--ckpt-dir", default=_env("CKPT_DIR", "/work/checkpoints_distill_init"))
    p.add_argument("--s3-base", default=_env("S3_BASE", ""))
    p.add_argument("--device", default=_env("DEVICE", ""))
    args = p.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"[distill-init] device={device}", flush=True)

    if args.teacher_ckpt_s3:
        print(f"[teacher] downloading {args.teacher_ckpt_s3}", flush=True)
        subprocess.run(["aws", "s3", "cp", args.teacher_ckpt_s3, args.teacher_ckpt,
                        "--no-progress"], check=True)

    cfg = distill_init_config(
        latent_channels=args.teacher_n_filters,
        dyn_n_filters=args.teacher_n_filters,
        teacher_n_blocks=args.teacher_n_blocks,
        teacher_n_filters=args.teacher_n_filters,
        num_simulations=args.sims,
        mcts_batch_size=args.mcts_batch_size,
        mcts_top_k=args.mcts_top_k,
        num_unroll_steps=args.unroll,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon_random=args.epsilon_random,
    )
    print(f"[config] {cfg}", flush=True)

    teacher = load_teacher(args.teacher_ckpt, n_blocks=args.teacher_n_blocks,
                           n_filters=args.teacher_n_filters, map_location=device).to(device)
    composed = DistillInitMuZeroNet(
        TeacherRepresentation(teacher), DynamicsNet(cfg), TeacherPrediction(teacher),
    ).to(device)
    opt = torch.optim.Adam(composed.g.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Sanity: factoring is exact at startup (catches a wrong teacher shape early).
    with torch.no_grad():
        o = torch.randn(2, 19, 8, 8, device=device)
        tp, tv = teacher(o)
        fp, fv = composed.f(composed.h(o))
        assert torch.allclose(tp, fp, atol=1e-4), "teacher factoring mismatch!"
    print("[distill-init] factoring verified: f(h(obs)) == teacher(obs)", flush=True)

    print(f"[data] generating {args.n_games} teacher self-play games...", flush=True)
    t0 = time.time()
    games = generate_transitions(teacher, cfg, args.n_games, device=device, seed=0)
    buf = TransitionBuffer(K=cfg.num_unroll_steps)
    buf.add(games)
    print(f"[data] {len(buf)} games, {buf.n_windows} K-step windows "
          f"({time.time() - t0:.0f}s)", flush=True)

    eval_elos = [int(x) for x in args.eval_elos.split(",") if x]
    history: dict = {"rounds": [], "evals": []}
    rng = random.Random(0)

    for rnd in range(args.rounds):
        t0 = time.time()
        agg = {"loss": 0.0, "latent_loss": 0.0, "pred_loss": 0.0, "reward_loss": 0.0}
        for _ in range(args.train_steps_per_round):
            batch = buf.sample(cfg.batch_size, rng=rng)
            batch = {k: v.to(device) for k, v in batch.items()}
            stats = distill_step(composed, opt, batch, cfg)
            for k in agg:
                agg[k] += stats[k]
        for k in agg:
            agg[k] /= args.train_steps_per_round
        train_dt = time.time() - t0
        history["rounds"].append({"round": rnd, **agg, "train_time": train_dt})
        print(f"[round {rnd:2d}] loss={agg['loss']:.4f} latent={agg['latent_loss']:.4f} "
              f"pred={agg['pred_loss']:.4f} reward={agg['reward_loss']:.4f} ({train_dt:.0f}s)",
              flush=True)

        for elo in eval_elos:
            te = time.time()
            res = evaluate_vs_stockfish(composed, cfg, device, stockfish_elo=elo,
                                        n_games=args.eval_games, sims=cfg.num_simulations)
            row = {"round": rnd, "stockfish_elo": elo, **res, "eval_time": time.time() - te}
            history["evals"].append(row)
            print(f"[round {rnd:2d} eval] vs SF UCI={elo} sims={cfg.num_simulations} "
                  f"score={res['score']:.3f} W={res['wins']} D={res['draws']} "
                  f"L={res['losses']} ({row['eval_time']:.0f}s)", flush=True)

        torch.save({"round": rnd, "g": composed.g.state_dict(),
                    "config": cfg.__dict__, "history": history},
                   os.path.join(args.ckpt_dir, "current.pt"))
        with open(os.path.join(args.ckpt_dir, "history.json"), "w") as fh:
            json.dump(history, fh, default=float)
        if args.s3_base:
            _s3_sync(args.ckpt_dir, args.s3_base)

    print("[distill-init] done.", flush=True)


if __name__ == "__main__":
    main()
