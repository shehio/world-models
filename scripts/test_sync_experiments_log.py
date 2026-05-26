"""Tests for sync_experiments_log.py's transform pipeline.

The script's transform is pure-functional given input text → output text,
so we can exercise the failure modes that have bitten us (or could):

  - Frontmatter stripping handles missing / malformed YAML headers
  - Hugo block-attribute anchors ({#id}) become <a id="id"></a>
  - Hugo shortcodes ({{< chart-foo >}}) become a one-line pointer
  - Internal Hugo links (/experiments/#search) become absolute github.io URLs
  - The full source file at site/content/experiments.md round-trips with
    no Hugo bits left in the output (this is what CI's --check enforces)

Run from repo root:
    python -m pytest scripts/test_sync_experiments_log.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sync_experiments_log import (
    SITE_BASE,
    SOURCE,
    rewrite_anchors,
    rewrite_internal_links,
    strip_frontmatter,
    strip_shortcodes,
    transform,
)


def test_strip_frontmatter_removes_yaml_block():
    text = "---\ntitle: foo\nnext: /bar/\n---\n\nbody\n"
    assert strip_frontmatter(text) == "body\n"


def test_strip_frontmatter_passes_through_no_header():
    text = "no frontmatter here\nbody\n"
    assert strip_frontmatter(text) == text


def test_strip_frontmatter_handles_unterminated_header():
    text = "---\nstart but no end\n\nbody\n"
    assert strip_frontmatter(text) == text  # left alone, safer than truncating


def test_rewrite_anchors_converts_block_attribute():
    text = "## Heading {#capacity}\n"
    out = rewrite_anchors(text)
    assert out == '## Heading <a id="capacity"></a>\n'


def test_rewrite_anchors_leaves_non_anchor_braces_alone():
    text = "## Heading {not-an-anchor}\n"
    assert rewrite_anchors(text) == text


def test_strip_shortcodes_replaces_chart_shortcode_with_pointer():
    text = "before\n{{< chart-capacity >}}\nafter\n"
    out = strip_shortcodes(text)
    assert "{{<" not in out
    assert ">}}" not in out
    assert "chart-capacity" in out


def test_strip_shortcodes_handles_multiple_in_same_text():
    text = "{{< chart-search >}} and {{< chart-data-scale >}}"
    out = strip_shortcodes(text)
    assert "chart-search" in out
    assert "chart-data-scale" in out
    assert "{{<" not in out


def test_rewrite_internal_links_promotes_relative_to_github_io():
    text = "see [the search section](/experiments/#search) for details"
    out = rewrite_internal_links(text)
    assert f"]({SITE_BASE}/experiments/#search)" in out


def test_rewrite_internal_links_leaves_external_links_alone():
    text = "[paper](https://arxiv.org/abs/1712.01815)"
    assert rewrite_internal_links(text) == text


def test_rewrite_internal_links_leaves_fragment_only_links_alone():
    text = "[same page](#peak-elo)"
    assert rewrite_internal_links(text) == text


def test_rewrite_internal_links_leaves_relative_path_links_alone():
    text = "[file](./README.md)"
    assert rewrite_internal_links(text) == text


def test_transform_strips_all_hugo_artifacts_from_real_source():
    """End-to-end: transformed real source has no Hugo block-attrs, no
    shortcodes, no bare-/-prefixed markdown links."""
    src_text = SOURCE.read_text(encoding="utf-8")
    out = transform(src_text)
    assert "{{<" not in out, "found surviving Hugo shortcode"
    assert ">}}" not in out, "found surviving Hugo shortcode close"
    # Bare `{#anchor}` blocks should all be converted to <a id=...> tags.
    assert not re.search(r"\{#[A-Za-z0-9_-]+\}", out), \
        "found surviving Hugo block-attribute anchor"
    # Frontmatter should be gone.
    assert not out.startswith("---")
    # No `](/...)` markdown links left (rewrite_internal_links should
    # promote them all to absolute URLs).
    bare_internal = re.findall(r"\]\((/[A-Za-z][^)]*)\)", out)
    assert not bare_internal, f"unrewritten internal links: {bare_internal[:3]}"


def test_transform_preserves_external_github_urls():
    src_text = SOURCE.read_text(encoding="utf-8")
    out = transform(src_text)
    # The source has many launcher/source links pointing at github.com — they
    # should survive the transform untouched.
    assert "https://github.com/shehio/world-models" in out


def test_transform_emits_autogen_header():
    out = transform("---\ntitle: x\n---\n\nbody\n")
    assert out.startswith("<!-- AUTO-GENERATED"), \
        "transform output must lead with an auto-generated marker so " \
        "humans don't edit it by hand"
