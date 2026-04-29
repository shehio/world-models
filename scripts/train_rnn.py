"""Train the MDN-RNN on encoded sequences."""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from world_models.config import CFG
from world_models.data import LatentSequenceDataset
from world_models.mdn_rnn import MDNRNN, mdn_loss
from world_models.utils import pick_device, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=CFG.rnn_epochs)
    parser.add_argument("--batch-size", type=int, default=CFG.rnn_batch_size)
    parser.add_argument("--lr", type=float, default=CFG.rnn_lr)
    parser.add_argument("--seq-len", type=int, default=CFG.rnn_seq_len)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    print(f"device={device}")

    ds = LatentSequenceDataset(CFG.latents_dir, args.seq_len)
    print(f"sequences: {len(ds)}  (seq_len={args.seq_len})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = MDNRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = n = 0.0
        pbar = tqdm(loader, desc=f"rnn epoch {epoch+1}/{args.epochs}")
        for z_in, a_in, z_tgt in pbar:
            z_in, a_in, z_tgt = z_in.to(device), a_in.to(device), z_tgt.to(device)
            (logit_pi, mu, log_sigma), _ = model(z_in, a_in)
            loss = mdn_loss(logit_pi, mu, log_sigma, z_tgt)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * z_in.size(0)
            n += z_in.size(0)
            pbar.set_postfix(nll=f"{loss_sum/n:.3f}")

        torch.save(
            {"state_dict": model.state_dict(),
             "z_dim": CFG.z_dim, "hidden": CFG.rnn_hidden, "K": CFG.mdn_components},
            CFG.rnn_path,
        )
    print(f"RNN training finished in {time.time()-t0:.1f}s -> {CFG.rnn_path}")


if __name__ == "__main__":
    main()
