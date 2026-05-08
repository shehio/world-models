"""Collect random-policy rollouts from CarRacing.

We need data to train V (and indirectly M and C). Pure random actions
are surprisingly fine for V because the racing track + car visually
covers most of the state space the agent will ever see — the agent
mostly drives around on grass and asphalt regardless of skill.

Each rollout is saved as a separate compressed `.npz` so we can grow the
dataset incrementally and shuffle/restart cheaply.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from tqdm.auto import tqdm

from world_models.config import CFG
from world_models.env import make_env, preprocess
from world_models.utils import seed_everything


def random_action(rng: np.random.Generator) -> np.ndarray:
    """Slightly-biased random action: more gas than brake to keep the car moving."""
    steer = rng.uniform(-1.0, 1.0)
    gas = rng.uniform(0.0, 1.0) ** 0.7  # bias toward higher gas
    brake = rng.uniform(0.0, 1.0) ** 3.0  # bias toward less brake
    return np.array([steer, gas, brake], dtype=np.float32)


def collect_one(env, episode_idx: int, max_steps: int, rng: np.random.Generator) -> dict:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    frames, actions, rewards = [], [], []
    # Use the same action for k steps to encourage non-trivial trajectories.
    repeat_steps = 4
    cur_action = random_action(rng)
    for step in range(max_steps):
        if step % repeat_steps == 0:
            cur_action = random_action(rng)
        frames.append(preprocess(obs))
        actions.append(cur_action)
        obs, reward, terminated, truncated, _ = env.step(cur_action)
        rewards.append(reward)
        if terminated or truncated:
            break
    return {
        "frames": np.stack(frames).astype(np.float32),       # (T, 64, 64, 3)
        "actions": np.stack(actions).astype(np.float32),     # (T, 3)
        "rewards": np.array(rewards, dtype=np.float32),      # (T,)
        "episode": episode_idx,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=CFG.n_rollouts)
    parser.add_argument("--max-steps", type=int, default=CFG.rollout_max_steps)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args()

    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    env = make_env()
    print(f"Collecting {args.rollouts} rollouts of up to {args.max_steps} steps each...")
    t0 = time.time()

    total_frames = 0
    for ep in tqdm(range(args.rollouts), desc="rollouts"):
        rollout = collect_one(env, ep, args.max_steps, rng)
        out = CFG.rollouts_dir / f"rollout_{ep:04d}.npz"
        np.savez_compressed(
            out,
            frames=rollout["frames"],
            actions=rollout["actions"],
            rewards=rollout["rewards"],
        )
        total_frames += rollout["frames"].shape[0]

    env.close()
    print(f"Saved {total_frames} frames in {time.time()-t0:.1f}s -> {CFG.rollouts_dir}")


if __name__ == "__main__":
    main()
