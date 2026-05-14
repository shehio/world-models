# Engineering notes

Distilled lessons from operating the wm-chess pipeline — the kinds of
things that aren't obvious from the code or git history but matter the
next time something goes sideways.

Each note follows the same shape: what's going on, **Why:** the
constraint or incident that drove it, **How to apply:** when it kicks
in next time.

## Index

### Project state
- [`world-models-datagen-state.md`](./world-models-datagen-state.md) —
  the two-region EKS datagen runs (us-east d15, eu-west d10), what
  got collected vs lost, the salvage path through finalize + merge.
- [`d15-d10-results.md`](./d15-d10-results.md) — Elo numbers and
  status of the current ablations (A: 40×256, C: full 30M, E: sims=4000).
- [`migration-state.md`](./migration-state.md) — `MIGRATION_PLAN.md`
  was substantially done; what was left and is now finished.

### Training infrastructure
- [`training-infra-layout.md`](./training-infra-layout.md) — the GPU
  EKS pipeline (image, clusters, jobs, indexed checkpoints) and the
  region-specific gotchas (us-east on-demand, eu-west spot-only).
- [`multi-region-capacity.md`](./multi-region-capacity.md) —
  us-east-1 + eu-central-1 as quota-pool partners; cross-region ECR
  pull and S3 read just work.
- [`bare-ec2-launchers.md`](./bare-ec2-launchers.md) — the
  `infra-eks/launchers/` pattern for one-shot self-terminating EC2s
  when EKS can't get capacity.

### Operational gotchas
- [`eks-gpu-taint-breaks-coredns.md`](./eks-gpu-taint-breaks-coredns.md) —
  don't add `nvidia.com/gpu:NoSchedule` to single-node EKS clusters.
- [`finalize-merge-oom.md`](./finalize-merge-oom.md) — both
  `finalize_library_path` and `merge_shards.py` materialize all chunks
  in RAM and OOM on large pods. Workaround: streaming `merge_chunks.py`.
- [`training-dataloader-memmap.md`](./training-dataloader-memmap.md) —
  `MultipvDataset` uses memory-mapped `.npy` files because merged
  datasets don't fit in any sensible GPU instance's RAM.

These notes started life as the assistant's auto-memory; copied here so
they're discoverable in the repo and don't drift away from the code
they describe.
