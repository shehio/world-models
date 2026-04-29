"""Gymnasium environment factory and frame preprocessing.

CarRacing-v3 returns 96x96x3 uint8 frames with a small HUD bar at the
bottom. We resize to 64x64x3 and convert to float32 in [0, 1] — which is
what every other module in the project consumes.
"""
from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np

from world_models.config import CFG


def make_env(render_mode: str | None = None) -> gym.Env:
    """Build the env with continuous controls (the canonical World Models setup)."""
    return gym.make(CFG.env_id, render_mode=render_mode, continuous=True)


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert a raw env frame to (H, W, C) float32 in [0, 1] at CFG.image_size."""
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.shape[0] != CFG.image_size or frame.shape[1] != CFG.image_size:
        frame = cv2.resize(
            frame, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_AREA
        )
    return frame.astype(np.float32) / 255.0


def to_chw(batch_hwc: np.ndarray) -> np.ndarray:
    """(N,H,W,C) -> (N,C,H,W) for PyTorch."""
    return np.transpose(batch_hwc, (0, 3, 1, 2))
