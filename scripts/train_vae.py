"""Train the VAE on collected frames."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from world_models.config import CFG
from world_models.data import FrameDataset
from world_models.utils import pick_device, seed_everything
from world_models.vae import VAE, vae_loss


def save_recon_grid(model: VAE, batch: torch.Tensor, path) -> None:
    import torchvision

    model.eval()
    with torch.no_grad():
        out = model(batch)
        grid = torch.cat([batch[:8].cpu(), out.recon[:8].cpu()], dim=0)  # 8 originals + 8 recons
    torchvision.utils.save_image(grid, path, nrow=8)
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.vae_epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.vae_batch_size)
    parser.add_argument("--lr", type=float, default=CFG.vae_lr)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    print(f"device={device}")

    ds = FrameDataset(CFG.rollouts_dir)
    print(f"frames: {len(ds)}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = VAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    fixed = ds.frames[:16].to(device)  # for the recon-grid checkpoint

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        recon_sum = kl_sum = n = 0.0
        pbar = tqdm(loader, desc=f"vae epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            batch = batch.to(device)
            out = model(batch)
            losses = vae_loss(out, batch, kl_weight=CFG.vae_kl_weight)
            opt.zero_grad()
            losses["loss"].backward()
            opt.step()
            recon_sum += losses["recon"].item() * batch.size(0)
            kl_sum += losses["kl"].item() * batch.size(0)
            n += batch.size(0)
            pbar.set_postfix(recon=f"{recon_sum/n:.1f}", kl=f"{kl_sum/n:.2f}")

        save_recon_grid(model, fixed, CFG.ckpt_dir / f"vae_recon_epoch{epoch+1}.png")
        torch.save({"state_dict": model.state_dict(), "z_dim": CFG.z_dim}, CFG.vae_path)

    print(f"VAE training finished in {time.time()-t0:.1f}s -> {CFG.vae_path}")


if __name__ == "__main__":
    main()
