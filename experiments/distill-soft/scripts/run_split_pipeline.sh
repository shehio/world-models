#!/usr/bin/env bash
# DEPRECATED — the two-box split pipeline (single CPU box for datagen,
# single GPU box for training) was replaced by the EKS-native pipeline
# in `infra-eks/`. The `infra/` Terraform module it depended on has
# been removed.
#
# What this script used to do
# ---------------------------
#   Phase 1: terraform apply CPU box → rsync code → run generate_data
#            → rsync data back → terraform destroy
#   Phase 2: terraform apply GPU box → rsync (code + data) → train.py
#            → rsync ckpts back → terraform destroy
#   Phase 3: eval on GPU box (still applied from phase 2)
#
# Where each phase lives now
# --------------------------
#   Phase 1: infra-eks/k8s/job-gen.yaml + job-merge.yaml
#            (parallel across N pods; spot-reclamation-safe via
#            shards_partial/chunks/)
#   Phase 2: TODO (GPU EKS nodegroup; see infra-eks/README.md, the
#            "GPU training" section once that lands)
#   Phase 3: same as Phase 2 — runs as a second Job on the GPU
#            nodegroup
#
# See infra-eks/README.md for the current end-to-end workflow.

cat >&2 <<'EOF'

  run_split_pipeline.sh has been removed.

  The Terraform module (infra/) it provisioned no longer exists in
  this repo. For an end-to-end run on AWS, use the EKS pipeline:

      cd infra-eks
      ./run.sh cluster create
      ./run.sh docker build && ./run.sh docker push
      ./run.sh gen      # parallel datagen
      ./run.sh merge    # cross-pod merge

  Training on GPU under the same cluster is documented in
  infra-eks/README.md (or coming soon if not yet there).

EOF
exit 1
