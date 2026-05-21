"""Template-rendering tests. No AWS needed."""
from __future__ import annotations

from pathlib import Path

from spot_rover.capacity import CapacityProbe
from spot_rover.score import RankedCandidate
from spot_rover.template import cluster_name, render_cluster, render_job


def _ranked(region: str, az: str, itype: str, score: float = 0.5) -> RankedCandidate:
    p = CapacityProbe(
        region=region, az=az, instance_type=itype,
        spot_price=0.30, on_demand_price=1.50,
        price_stddev_7d=0.01, interrupt_band="<5%", vcpus=32,
    )
    return RankedCandidate(
        probe=p, cost_score=1.0, stability_score=1.0,
        discount_score=1.0, interrupt_score=1.0,
    )


def test_cluster_name_includes_region_suffix():
    assert cluster_name("wm-chess-d15-250k", "us-east-1") == "wm-chess-d15-250k-use1"
    assert cluster_name("wm-chess-d15-250k", "eu-central-1") == "wm-chess-d15-250k-euc1"


def test_cluster_name_fallback_for_unknown_region():
    # ap-south-1 isn't in our hardcoded suffix table → strip dashes
    assert cluster_name("wm-go", "ap-south-1") == "wm-go-apsouth1"


def test_render_includes_winner_first_then_fallbacks():
    winner = _ranked("us-east-1", "us-east-1a", "c7a.8xlarge")
    sib1 = _ranked("us-east-1", "us-east-1b", "c7i.8xlarge")
    sib2 = _ranked("us-east-1", "us-east-1a", "c6i.8xlarge")
    other_region = _ranked("eu-central-1", "eu-central-1a", "c7a.8xlarge")

    out = render_cluster(
        winner, [winner, sib1, sib2, other_region],
        workload="wm-chess-d15-250k", desired_capacity=8,
    )
    yaml = out.cluster_yaml
    assert "wm-chess-d15-250k-use1" in yaml
    assert "region: us-east-1" in yaml
    assert "desiredCapacity: 8" in yaml
    # winner must appear before siblings in the list
    pos_winner = yaml.find("c7a.8xlarge")
    pos_sib1 = yaml.find("c7i.8xlarge")
    pos_sib2 = yaml.find("c6i.8xlarge")
    assert 0 <= pos_winner < pos_sib1
    assert pos_winner < pos_sib2
    # other-region candidate must NOT be included
    assert out.fallback_instance_types == ["c7i.8xlarge", "c6i.8xlarge"]


def test_render_job_substitutes_placeholders(tmp_path: Path):
    tmpl = tmp_path / "job.yaml"
    tmpl.write_text(
        "image: ${ECR_URI}:latest\n"
        "S3_REGION=${S3_REGION}\n"
        "S3_BUCKET=${S3_BUCKET}\n"
        "S3_PREFIX=${S3_PREFIX}\n"
    )
    out = render_job(
        tmpl, ecr_uri="123.dkr.ecr.us-east-1.amazonaws.com/wm-chess",
        s3_region="us-east-1", s3_bucket="my-bucket", s3_prefix="my/prefix",
    )
    assert "${" not in out
    assert "my-bucket" in out
    assert "us-east-1" in out


def test_render_job_overrides():
    """Workload-specific overrides patch additional placeholders."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("image: ${ECR_URI}\nseed: ${BASE_SEED}\n")
        path = Path(f.name)
    try:
        out = render_job(
            path, ecr_uri="123/wm", s3_region="us-east-1",
            s3_bucket="b", s3_prefix="p",
            overrides={"BASE_SEED": "99042"},
        )
        assert "99042" in out
    finally:
        path.unlink()
