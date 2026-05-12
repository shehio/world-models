#!/usr/bin/env bash
# First-boot bootstrap for the 02c experiment box.
# Runs as root via cloud-init. Idempotent; safe to re-run.
#
# Installs:
#   - stockfish (apt)
#   - uv (modern Python package manager)
#   - basic monitoring tools
#
# Does NOT clone the code — push it up with the rsync command in
# `terraform output rsync_command`.

set -eux

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y stockfish htop tmux jq rsync

# uv: installed for the ubuntu user so they don't need sudo.
sudo -u ubuntu bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
# Make uv discoverable in non-login shells.
echo 'export PATH="$HOME/.local/bin:$PATH"' | sudo -u ubuntu tee -a /home/ubuntu/.bashrc

# Quality of life: pre-stage a workspace dir owned by ubuntu.
mkdir -p /home/ubuntu/work
chown -R ubuntu:ubuntu /home/ubuntu/work

# Marker so we can verify the bootstrap completed via SSH.
date -u +"%Y-%m-%dT%H:%M:%SZ bootstrap complete (project=${project_tag})" \
  | tee /home/ubuntu/BOOTSTRAP_COMPLETE
chown ubuntu:ubuntu /home/ubuntu/BOOTSTRAP_COMPLETE
