# spot-rover

Scan AWS regions for available spot capacity, score the options, and deploy
an EKS-based datagen job to the winner. One-shot MVP.

## Why this exists

On 2026-05-20 the us-east-1 spot pool for c7a/c7i collapsed for ~30 minutes —
pods reclaimed within 30s of spawning, the Job controller getting stuck after
each reclaim wave. Failover to eu-central-1 was manual: pick a region by feel,
copy `cluster-gen-d15-250k.yaml` → `-eu.yaml`, hand-edit, `eksctl create
cluster`, `kubectl apply`. Took an hour of intervention.

The rover turns that into a CLI invocation. For Go datagen (which the
project plan calls out as "harder than chess" — more positions per game,
longer per-position eval at the visit counts that produce useful soft
policy), this matters more, not less.

## How it scores

For each `(region, AZ, instance-type)` tuple it can sample, the rover
computes four factors in [0, 1]:

- **cost** — spot $/vCPU vs. the cheapest seen.
- **stability** — inverse of 7-day spot-price stddev (high variance ≈
  hot pool ≈ likely reclaims).
- **discount** — spot vs. on-demand price ratio; near-parity = pool is hot.
- **interrupt** — Spot Advisor band (`<5%` … `>20%`) if reachable.

Composite = product of the four. The product (not sum) means a single
bad factor kills the candidate, which matches how spot interruption
actually hurts: one reclaim wave can erase hours of work.

## Usage

```bash
# Install once
uv sync --project infra-eks/spot-rover

# Dry-run — capacity report only, no rendering, no AWS provisioning.
uv run --project infra-eks/spot-rover \
    python infra-eks/spot-rover/scripts/rove.py \
    --config infra-eks/spot-rover/configs/chess-d15-250k.yaml \
    --report-only

# Render the eksctl cluster YAML + Job YAML for the chosen region, print
# the eksctl/kubectl commands without running them.
uv run --project infra-eks/spot-rover \
    python infra-eks/spot-rover/scripts/rove.py \
    --config infra-eks/spot-rover/configs/chess-d15-250k.yaml

# Same but actually provision (cluster create takes ~15 min).
uv run --project infra-eks/spot-rover \
    python infra-eks/spot-rover/scripts/rove.py \
    --config infra-eks/spot-rover/configs/chess-d15-250k.yaml \
    --execute
```

The default is dry-run because creating a cluster commits real spend
(~$0.10/hr for the control plane plus instance cost). The output prints
exactly which `eksctl create cluster -f …` and `kubectl apply -f …`
commands would run, so you can review before flipping `--execute`.

## State

On `--execute` success the rover writes a record to
`s3://<state_bucket>/<state_prefix>/state.json`:

```json
{
  "version": 1,
  "updated_at": "2026-05-20T22:00:00Z",
  "deployments": [
    {
      "deployment_id": "wm-chess-d15-250k-euc1",
      "workload": "wm-chess-d15-250k",
      "region": "eu-central-1",
      "cluster_name": "wm-chess-d15-250k-euc1",
      "job_name": "wm-chess-d15-250k",
      "instance_type": "c7a.8xlarge",
      "primary_az": "eu-central-1a",
      "fallback_instance_types": ["c7i.8xlarge", "c6i.8xlarge"],
      "created_at": "2026-05-20T22:00:00Z",
      "status": "active",
      "notes": ""
    }
  ]
}
```

That object is the input the phase-2 continuous daemon will read to know
which clusters to monitor. For the MVP it's a deployment log.

## Layout

```
spot-rover/
  src/spot_rover/
    capacity.py   — boto3 spot-price-history + Spot Advisor fetch
    score.py      — rank candidates by composite score
    template.py   — render eksctl ClusterConfig + envsubst the Job YAML
    provision.py  — drive eksctl + kubectl (dry-run by default)
    state.py      — S3 deployment registry
  scripts/rove.py — main CLI
  configs/
    chess-d15-250k.yaml — workload config for chess datagen
    go-9x9.yaml         — placeholder for the Go pipeline once wm-go image exists
  templates/
    cluster.yaml.tmpl   — eksctl ClusterConfig with rover placeholders
  tests/
    test_score.py       — scoring unit tests, no AWS
    test_template.py    — template rendering, no AWS
```

## What's NOT in the MVP

- **Continuous monitoring.** The rover runs once and exits. If your
  cluster's spot pool collapses, you re-run the rover (it'll pick a
  different region this time) and tear down the old cluster manually.
  Phase-2 daemon would watch the deployed fleet and re-roll automatically.
- **Cross-region failover within a single job.** Re-rolling means killing
  the job and starting fresh in a new region; the chess pipeline's
  S3-partial-sync design means no data loss, but each pod restarts.
- **Live-test fulfillment confirmation.** The score uses historical
  signals (spot price + interruption band) as proxies. Adding a small
  canary fleet request as a final confirmation step would raise
  confidence at the cost of ~30s of wall-clock per region scored.
- **GPU instance scoring.** The on-demand price table includes g5/g6
  entries but the scoring math doesn't differentiate CPU-bound from
  GPU-bound workloads — for production GPU datagen, weight the cost
  factor by `$/GPU-hr` instead of `$/vCPU-hr`.

## Tests

```bash
uv run --project infra-eks/spot-rover pytest infra-eks/spot-rover/tests/
```

Two test files, no AWS calls. Scoring + template rendering only — the
boto3 paths are exercised by actually running the CLI against AWS.
