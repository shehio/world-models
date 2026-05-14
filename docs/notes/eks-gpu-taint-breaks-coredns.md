# EKS GPU taint breaks CoreDNS

Tainting GPU nodes with `nvidia.com/gpu:NoSchedule` breaks CoreDNS in
single-node EKS clusters — the only node is tainted and CoreDNS can't
tolerate it.

The naive eksctl GPU spec adds a taint:

```yaml
taints:
  - key: nvidia.com/gpu
    value: "true"
    effect: NoSchedule
```

In a **single-node cluster** (typical for one-job training runs), this
breaks the cluster: the only node is tainted, EKS-managed system pods
(CoreDNS, metrics-server) don't tolerate `nvidia.com/gpu`, so they
stay `Pending` forever. With CoreDNS unscheduled:

- Pods can't resolve DNS (e.g. `sts.<region>.amazonaws.com`).
- `aws s3 cp` from the training pod fails with `Could not connect to the endpoint URL`.
- The training Job hits `BackoffLimitExceeded`.

This took ~45 min to diagnose on the first training rollout because
the symptom (no STS connectivity) looked like a VPC/IGW/NAT problem,
not a DNS problem.

**Fix (May 2026):** removed the taint from both
`cluster-train-{us,eu}.yaml`. Single-purpose clusters don't need the
taint — the Job's `nodeSelector role=gpu-trainer` is what targets the
GPU node, and there are no other workloads competing for it. For
multi-tenant clusters where the taint actually matters, the right fix
is to add tolerations to the EKS-managed CoreDNS/metrics-server
deployments instead.

**Why:** Default eksctl GPU specs (and most online tutorials) include
the taint because they assume multi-tenant clusters where the taint
usefully blocks random pods from the expensive GPU node. In our setup
the assumption doesn't hold.

**How to apply:** For new single-purpose GPU training clusters, **omit
the taint** from the eksctl spec. If you do need it (multi-tenant),
patch the CoreDNS deployment to tolerate it before any training Job
submits, or use a non-GPU nodegroup alongside.
