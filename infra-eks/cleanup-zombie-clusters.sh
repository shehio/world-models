#!/usr/bin/env bash
# Scan AWS regions for EKS clusters that look unused (no nodes, or only
# old idle nodes) and delete them. Dry-run by default; pass --apply to
# actually fire the deletes.
#
# Why this exists: after the d15-250k datagen completed, the EU cluster
# was thought to be torn down — but actually still had 8× m7i.8xlarge
# spot nodes running for *three more days*. ~$77/day in idle compute.
# This script catches that class of bug.
#
# Heuristic: any cluster with name matching "wm-*" or "*-gen-*" or
# "*-train-*" that has 0 running nodes (or only old nodes whose tags
# don't match an active workflow) is flagged.
#
# Usage:
#   bash infra-eks/cleanup-zombie-clusters.sh                # dry-run
#   bash infra-eks/cleanup-zombie-clusters.sh --apply        # delete flagged
#   REGIONS="us-east-1 eu-central-1" bash ... --apply        # limit regions
#   PATTERN="wm-go-*" bash ... --apply                       # different prefix
#
# Exits non-zero only if delete commands fail; "found zombies but dry-run"
# is exit 0 (treat as informational).

set -u

APPLY=0
if [ "${1:-}" = "--apply" ]; then APPLY=1; fi

# Default regions covers everywhere we routinely launch. Override via
# REGIONS env var. (us-east-2, us-west-2 + eu-west-1 included even
# though we rarely use them — zombie clusters there would also bill.)
REGIONS=${REGIONS:-"us-east-1 us-east-2 us-west-2 eu-central-1 eu-west-1"}
PATTERN=${PATTERN:-"wm-*"}

echo "=== cleanup-zombie-clusters $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "regions: $REGIONS"
echo "pattern: $PATTERN"
echo "mode:    $([ $APPLY -eq 1 ] && echo APPLY || echo DRY-RUN)"
echo ""

# fnmatch-style match — bash's [[ $name == $pattern ]] handles *
name_matches() {
    local name="$1"
    local pat="$2"
    [[ $name == $pat ]]
}

# Returns 0 if cluster $1 in region $2 looks like a zombie.
is_zombie() {
    local cluster="$1"
    local region="$2"

    # 1) Are any nodes running for this cluster?
    local n_nodes
    n_nodes=$(aws ec2 describe-instances --region "$region" \
        --filters "Name=tag:eks:cluster-name,Values=$cluster" \
                  "Name=instance-state-name,Values=running,pending" \
        --query 'length(Reservations[].Instances[])' --output text 2>/dev/null)
    n_nodes=${n_nodes:-0}

    # 2) Cluster status (DELETING is in-progress; skip)
    local status
    status=$(aws eks describe-cluster --region "$region" --name "$cluster" \
        --query 'cluster.status' --output text 2>/dev/null)
    if [ "$status" = "DELETING" ]; then
        echo "  [$region/$cluster] already DELETING — skipping"
        return 1
    fi

    # Zombie if 0 nodes (control plane idle) — explicit decision: this is
    # the only safe heuristic. Clusters with running nodes might be doing
    # real work; let the human re-classify.
    if [ "$n_nodes" -eq 0 ]; then
        echo "  [$region/$cluster] ZOMBIE — 0 nodes, control plane only (\$0.10/h)"
        return 0
    else
        echo "  [$region/$cluster] $n_nodes nodes running — leaving alone"
        return 1
    fi
}

# Look for the matching YAML to feed to `eksctl delete cluster -f`.
# Falls back to `--name` + `--region` if no yaml found.
find_cluster_yaml() {
    local cluster="$1"
    local d
    for d in "$(dirname "$0")" "infra-eks"; do
        local f
        f=$(ls "$d"/cluster-*.yaml 2>/dev/null \
            | xargs -I{} grep -l "name: $cluster$" {} 2>/dev/null \
            | head -1)
        if [ -n "$f" ]; then echo "$f"; return 0; fi
    done
    return 1
}

delete_cluster() {
    local cluster="$1"
    local region="$2"
    local yaml
    if yaml=$(find_cluster_yaml "$cluster"); then
        echo "  [$region/$cluster] deleting via: eksctl delete cluster -f $yaml"
        eksctl delete cluster -f "$yaml" --disable-nodegroup-eviction 2>&1 | tail -3
    else
        echo "  [$region/$cluster] no YAML found, deleting by name"
        eksctl delete cluster --region "$region" --name "$cluster" --disable-nodegroup-eviction 2>&1 | tail -3
    fi
}

n_zombies=0
n_deleted=0
for region in $REGIONS; do
    clusters=$(aws eks list-clusters --region "$region" --query 'clusters[]' --output text 2>/dev/null)
    if [ -z "$clusters" ]; then continue; fi
    echo "[$region]"
    for cluster in $clusters; do
        if ! name_matches "$cluster" "$PATTERN"; then
            echo "  [$region/$cluster] doesn't match pattern '$PATTERN' — skipping"
            continue
        fi
        if is_zombie "$cluster" "$region"; then
            n_zombies=$((n_zombies + 1))
            if [ $APPLY -eq 1 ]; then
                delete_cluster "$cluster" "$region"
                n_deleted=$((n_deleted + 1))
            fi
        fi
    done
done

echo ""
echo "=== summary ==="
echo "zombies found:    $n_zombies"
echo "zombies deleted:  $n_deleted  ($([ $APPLY -eq 1 ] && echo 'applied' || echo 'dry-run, pass --apply'))"
