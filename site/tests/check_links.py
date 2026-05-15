#!/usr/bin/env python3
"""Static internal-link checker for the built Hugo site.

Walks every .html file under the given public directory, extracts every
`href`, and verifies that every internal target resolves to a file on
disk. External links (http://, https:// to anything other than our own
baseURL host), fragment-only links (#section), and mailto:/tel: are
skipped.

Exits 0 on success, 1 on any broken link, with a per-source-file report
of which targets failed. Designed to be the last guardrail before
`actions/upload-pages-artifact` — if this passes, the deploy has no
internal 404s.

Usage:
    python site/tests/check_links.py site/public
    python site/tests/check_links.py site/public --base-host shehio.github.io
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


HREF_RE = re.compile(r'''href\s*=\s*["']?([^\s"'>]+)''')


def extract_hrefs(html: str) -> list[str]:
    """All href targets in an HTML string, in source order, deduplicated
    by first appearance."""
    seen: dict[str, None] = {}
    for m in HREF_RE.finditer(html):
        url = m.group(1).strip()
        if url:
            seen.setdefault(url, None)
    return list(seen.keys())


def is_internal(url: str, base_host: str | None) -> bool:
    """True if `url` should be checked against the local public/ tree.

    Internal:
      - relative paths (`/findings/`, `findings/`, `css/main.css`)
      - absolute URLs whose host matches `base_host`

    Not internal (skipped):
      - other http/https hosts
      - fragments-only (`#anchor`)
      - mailto:, tel:, javascript:, data:
    """
    if not url or url.startswith("#"):
        return False
    parsed = urlparse(url)
    if parsed.scheme in {"mailto", "tel", "javascript", "data"}:
        return False
    if parsed.scheme in {"http", "https"}:
        if base_host and parsed.netloc == base_host:
            return True
        return False
    # No scheme: relative or absolute path → internal.
    return True


def resolve(url: str, public: Path, base_path_prefix: str) -> Path | None:
    """Map an internal URL to a candidate file on disk under `public/`.
    Returns the resolved Path if found, otherwise None."""
    parsed = urlparse(url)
    path = parsed.path
    # Strip the baseURL path prefix (e.g. "/world-models") if present so
    # we can map absolute Pages URLs back to public/ entries.
    if base_path_prefix and path.startswith(base_path_prefix):
        path = path[len(base_path_prefix):]
    if not path or path == "/":
        cand = public / "index.html"
        return cand if cand.exists() else None
    # Strip leading / so the path is relative to public/.
    p = path.lstrip("/")
    # Order: a real file first, then <dir>/index.html. Using `is_file()`
    # (not `exists()`) for the first probe so that a directory like
    # `public/findings/` doesn't resolve before its `index.html`.
    direct = public / p
    if direct.is_file():
        return direct
    index = direct / "index.html"
    if index.is_file():
        return index
    return None


def check_site(public: Path, base_host: str | None,
               base_path_prefix: str) -> dict[Path, list[str]]:
    """Return a {source_html: [broken_url, ...]} dict. Empty dict means
    no broken links."""
    broken: dict[Path, list[str]] = {}
    for html_file in sorted(public.rglob("*.html")):
        text = html_file.read_text(encoding="utf-8")
        misses: list[str] = []
        for url in extract_hrefs(text):
            if not is_internal(url, base_host):
                continue
            if resolve(url, public, base_path_prefix) is None:
                misses.append(url)
        if misses:
            broken[html_file] = misses
    return broken


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("public", type=Path,
                    help="Path to the built Hugo public/ directory.")
    ap.add_argument("--base-host", default="shehio.github.io",
                    help="Hostname treated as 'internal' for the link "
                         "check. Defaults to shehio.github.io.")
    ap.add_argument("--base-path-prefix", default="/world-models",
                    help="Path prefix on the baseURL (e.g. /world-models "
                         "for a project Pages site). Stripped before "
                         "looking up files. Empty for a root site.")
    args = ap.parse_args()

    if not args.public.is_dir():
        print(f"error: public dir not found: {args.public}", file=sys.stderr)
        return 2

    broken = check_site(args.public, args.base_host, args.base_path_prefix)
    if not broken:
        files = sum(1 for _ in args.public.rglob("*.html"))
        print(f"ok: no broken internal links across {files} HTML files.")
        return 0

    print("BROKEN INTERNAL LINKS:", file=sys.stderr)
    total = 0
    for src, urls in broken.items():
        rel = src.relative_to(args.public)
        print(f"  {rel}", file=sys.stderr)
        for u in urls:
            print(f"      -> {u}", file=sys.stderr)
            total += 1
    print(f"\n{total} broken link(s) in {len(broken)} file(s).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
