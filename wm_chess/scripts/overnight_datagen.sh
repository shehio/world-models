#!/usr/bin/env bash
# DEPRECATED — the terraform-based AWS pipeline this script drove was
# replaced by the EKS-native datagen pipeline in `infra-eks/`. The
# `infra/` Terraform module it depended on has been removed.
#
# Replacement
# -----------
# What this script used to do                    | Where it lives now
# -----------------------------------------------|--------------------------
# Drive N depth/T/multipv configs sequentially   | One Indexed Job per
# on a single spot EC2, idempotently             | config; bumped via
#                                                | `infra-eks/run.sh gen`
# rsync code up + library tree back              | S3 (entrypoint.sh syncs
#                                                | shards_partial then
#                                                | shards/, no rsync)
# `terraform apply` / `destroy`                  | `eksctl create/delete
#                                                | cluster -f cluster.yaml`
#                                                | (or scale the nodegroup
#                                                | to 0 between runs)
# Crash-safe checkpoints                         | Same (chunks/) — the
#                                                | partial-sync uploader in
#                                                | infra-eks/entrypoint.sh
#                                                | pushes them to S3 every
#                                                | $PARTIAL_SYNC_INTERVAL
# Cross-pod merge                                | infra-eks/k8s/job-
#                                                | merge.yaml (or the
#                                                | streaming
#                                                | wm_chess/scripts/
#                                                | merge_chunks.py for
#                                                | salvage runs)
#
# See infra-eks/README.md for the new workflow.

cat >&2 <<'EOF'

  overnight_datagen.sh has been removed.

  The terraform-based pipeline it drove (infra/) no longer exists in
  this repo. Use the EKS-native equivalent under infra-eks/:

      cd infra-eks
      ./run.sh cluster create           # one-time per region
      ./run.sh docker build && push     # one-time / on code change
      ./run.sh gen                      # submits the parallel Indexed Job
      ./run.sh merge                    # cross-pod merge after gen completes

  Configuration moved from .tsv files + KEY_NAME env to the env vars
  loaded from .env (see .env.example) and k8s manifests in
  infra-eks/k8s/.

EOF
exit 1
