"""Quick benchmark: CPU vs MPS for AlphaZeroNet at the sizes we actually use."""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from wm_chess.config import Config
from wm_chess.network import AlphaZeroNet


def bench(device_str: str, batch: int, n_planes: int, n_iters: int = 50):
    cfg = replace(Config(), n_res_blocks=10, n_filters=128, n_input_planes=n_planes)
    device = torch.device(device_str)
    net = AlphaZeroNet(cfg).to(device).eval()

    x = torch.randn(batch, n_planes, 8, 8, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(5):
            _ = net(x)
    if device_str == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = net(x)
    if device_str == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return dt / n_iters * 1000


def bench_train(device_str: str, batch: int, n_planes: int, n_iters: int = 30):
    cfg = replace(Config(), n_res_blocks=10, n_filters=128, n_input_planes=n_planes)
    device = torch.device(device_str)
    net = AlphaZeroNet(cfg).to(device).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.02, momentum=0.9)

    x = torch.randn(batch, n_planes, 8, 8, device=device)
    pi = torch.randn(batch, 4672, device=device).softmax(-1)
    z = torch.randn(batch, device=device)

    for _ in range(3):  # warmup
        opt.zero_grad()
        logits, v = net(x)
        loss = (-pi * torch.log_softmax(logits, dim=-1)).sum(-1).mean() + (v - z).pow(2).mean()
        loss.backward()
        opt.step()
    if device_str == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    for _ in range(n_iters):
        opt.zero_grad()
        logits, v = net(x)
        loss = (-pi * torch.log_softmax(logits, dim=-1)).sum(-1).mean() + (v - z).pow(2).mean()
        loss.backward()
        opt.step()
    if device_str == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    return dt / n_iters * 1000


print(f"PyTorch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print()

for n_planes in (19, 103):
    print(f"==== n_planes = {n_planes} ====")
    print(f"{'mode':<28}{'CPU (ms)':>12}{'MPS (ms)':>12}{'speedup':>10}")
    for batch, label in [(1, "fwd batch=1 (sim leaf)"),
                         (8, "fwd batch=8 (batched MCTS)"),
                         (256, "train batch=256")]:
        if "train" in label:
            cpu = bench_train("cpu", batch, n_planes)
            mps = bench_train("mps", batch, n_planes)
        else:
            cpu = bench("cpu", batch, n_planes)
            mps = bench("mps", batch, n_planes)
        spd = cpu / mps
        print(f"{label:<28}{cpu:>12.2f}{mps:>12.2f}{spd:>9.2f}x")
    print()
