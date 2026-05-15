"""Unit tests for the internal-link checker.

Each test builds a tiny fake `public/` directory in `tmp_path` and runs
the checker against it. Keeps the tests fast and the failure modes
explicit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_links import (
    check_site,
    extract_hrefs,
    is_internal,
    resolve,
)


# ─── extract_hrefs ──────────────────────────────────────────────────────────

def test_extract_hrefs_double_quotes():
    html = '<a href="/x/">x</a><a href="/y/">y</a>'
    assert extract_hrefs(html) == ["/x/", "/y/"]


def test_extract_hrefs_single_quotes():
    html = "<a href='/x/'>x</a>"
    assert extract_hrefs(html) == ["/x/"]


def test_extract_hrefs_minified_unquoted():
    """hugo --minify strips quotes from URLs that don't need them."""
    html = '<a href=/findings/>find</a><a href=https://github.com/x>g</a>'
    assert extract_hrefs(html) == ["/findings/", "https://github.com/x"]


def test_extract_hrefs_dedups_preserving_order():
    html = '<a href="/a/">1</a><a href="/b/">2</a><a href="/a/">3</a>'
    assert extract_hrefs(html) == ["/a/", "/b/"]


def test_extract_hrefs_empty_input():
    assert extract_hrefs("") == []
    assert extract_hrefs("<html><body><p>no links</p></body></html>") == []


# ─── is_internal ────────────────────────────────────────────────────────────

def test_is_internal_relative_paths():
    for url in ["/findings/", "/css/main.css", "findings/", "../foo/"]:
        assert is_internal(url, "shehio.github.io") is True, url


def test_is_internal_same_host_https():
    assert is_internal("https://shehio.github.io/world-models/findings/",
                       "shehio.github.io") is True


def test_is_internal_other_host():
    assert is_internal("https://github.com/shehio/world-models",
                       "shehio.github.io") is False


def test_is_internal_fragments_and_protocols():
    for url in ["#section", "mailto:a@b.c", "tel:+1234", "javascript:void(0)",
                "data:text/html,foo"]:
        assert is_internal(url, "shehio.github.io") is False, url


def test_is_internal_with_no_base_host():
    """When no base host is set, every absolute http(s) is external."""
    assert is_internal("https://shehio.github.io/x/", None) is False
    assert is_internal("/x/", None) is True


# ─── resolve ────────────────────────────────────────────────────────────────

@pytest.fixture
def fake_site(tmp_path: Path) -> Path:
    """A tiny public/ tree with the file layout Hugo would produce."""
    pub = tmp_path / "public"
    (pub / "css").mkdir(parents=True)
    (pub / "findings").mkdir()
    (pub / "method").mkdir()
    (pub / "baseline").mkdir()  # alias redirect dir
    (pub / "index.html").write_text("ROOT")
    (pub / "css" / "main.css").write_text("/* css */")
    (pub / "findings" / "index.html").write_text("FINDINGS")
    (pub / "method" / "index.html").write_text("METHOD")
    (pub / "baseline" / "index.html").write_text("ALIAS REDIRECT")
    (pub / "404.html").write_text("NOT FOUND")
    return pub


def test_resolve_root_path(fake_site):
    assert resolve("/", fake_site, "").name == "index.html"


def test_resolve_directory_url_returns_index(fake_site):
    """A URL ending in `/` resolves to <path>/index.html — the Hugo
    pattern."""
    got = resolve("/findings/", fake_site, "")
    assert got is not None
    assert got.parent.name == "findings"
    assert got.name == "index.html"


def test_resolve_alias_redirect_directory(fake_site):
    """An alias under /baseline/ should resolve — Hugo writes an
    index.html redirect there."""
    assert resolve("/baseline/", fake_site, "") is not None


def test_resolve_asset_file(fake_site):
    """A non-directory URL (a stylesheet) resolves directly."""
    got = resolve("/css/main.css", fake_site, "")
    assert got is not None
    assert got.name == "main.css"


def test_resolve_strips_base_path_prefix(fake_site):
    """Hugo with canonifyURLs emits the full
    https://shehio.github.io/world-models/findings/. After is_internal
    accepts it, resolve must strip the /world-models prefix and find
    /findings/."""
    assert resolve("/world-models/findings/", fake_site, "/world-models") is not None
    assert resolve("/world-models/", fake_site, "/world-models") is not None


def test_resolve_missing_returns_none(fake_site):
    assert resolve("/does-not-exist/", fake_site, "") is None
    assert resolve("/no-such-asset.css", fake_site, "") is None


def test_resolve_strips_fragment_and_query(fake_site):
    """An in-page anchor on a real page should still resolve."""
    assert resolve("/findings/#capacity", fake_site, "") is not None
    assert resolve("/findings/?ref=home", fake_site, "") is not None


# ─── check_site (end-to-end on a fake site) ─────────────────────────────────

def _write_page(pub: Path, path: str, body: str) -> None:
    p = pub / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"<!DOCTYPE html><html><body>{body}</body></html>")


def test_check_site_no_broken_links(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    _write_page(pub, "index.html",
                '<a href="/findings/">f</a><a href="/method/">m</a>')
    _write_page(pub, "findings/index.html", '<a href="/method/">m</a>')
    _write_page(pub, "method/index.html", '<a href="/findings/">f</a>')
    broken = check_site(pub, base_host=None, base_path_prefix="")
    assert broken == {}


def test_check_site_finds_broken_link(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    _write_page(pub, "index.html",
                '<a href="/exists/">ok</a><a href="/gone/">404</a>')
    _write_page(pub, "exists/index.html", "OK")
    broken = check_site(pub, base_host=None, base_path_prefix="")
    assert list(broken.values()) == [["/gone/"]]
    [src] = broken
    assert src.name == "index.html"


def test_check_site_ignores_external_and_anchors(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    _write_page(
        pub, "index.html",
        '<a href="https://github.com/x">gh</a>'
        '<a href="#top">top</a>'
        '<a href="mailto:a@b.c">m</a>'
        '<a href="/findings/">f</a>'
    )
    _write_page(pub, "findings/index.html", "F")
    assert check_site(pub, base_host=None, base_path_prefix="") == {}


def test_check_site_treats_same_host_as_internal(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    _write_page(
        pub, "index.html",
        '<a href="https://shehio.github.io/world-models/findings/">F</a>'
        '<a href="https://shehio.github.io/world-models/gone/">G</a>'
    )
    _write_page(pub, "findings/index.html", "F")
    broken = check_site(pub,
                        base_host="shehio.github.io",
                        base_path_prefix="/world-models")
    assert list(broken.values()) == [
        ["https://shehio.github.io/world-models/gone/"]
    ]


def test_check_site_reports_multiple_files(tmp_path):
    pub = tmp_path / "public"
    pub.mkdir()
    _write_page(pub, "a/index.html", '<a href="/gone1/">x</a>')
    _write_page(pub, "b/index.html", '<a href="/gone2/">y</a>')
    broken = check_site(pub, base_host=None, base_path_prefix="")
    rels = sorted((src.parent.name, urls) for src, urls in broken.items())
    assert rels == [("a", ["/gone1/"]), ("b", ["/gone2/"])]
