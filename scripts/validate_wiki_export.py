#!/usr/bin/env python3
"""Validate relative Markdown links in a GitHub Wiki export directory."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# [label](href) — href without scheme, optional #anchor
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def resolve_href(current_md: Path, href: str, export_root: Path) -> Path | None:
    """Return expected .md path for href, or None if external/anchor-only."""
    if not href or href.startswith(("#", "http://", "https://", "mailto:")):
        return None
    if "/" not in href and href.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")):
        return None
    path_part, _sep, _frag = href.partition("#")
    if not path_part:
        return None
    target = (current_md.parent / path_part).resolve()
    try:
        target.relative_to(export_root.resolve())
    except ValueError:
        return None
    if target.suffix.lower() == ".md":
        return target
    # GitHub wiki pages are Foo.md or foo/bar.md
    md_candidate = Path(str(target) + ".md")
    if md_candidate.is_file():
        return md_candidate
    if target.is_file():
        return target
    return md_candidate


def validate_export(export_root: Path) -> list[str]:
    export_root = export_root.resolve()
    errors: list[str] = []
    for md in sorted(export_root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for m in LINK_RE.finditer(text):
            href = m.group(1).strip()
            resolved = resolve_href(md, href, export_root)
            if resolved is None:
                continue
            if not resolved.is_file():
                rel = md.relative_to(export_root)
                errors.append(f"{rel}: broken link [{href}] -> {resolved.relative_to(export_root)}")
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description="Validate relative links in wiki export tree.")
    p.add_argument("export_dir", type=Path, nargs="?", default=None, help="Export directory (default: ./wiki_export)")
    args = p.parse_args()
    root = args.export_dir or Path(__file__).resolve().parent.parent / "wiki_export"
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    err = validate_export(root)
    if err:
        print(f"BROKEN LINKS ({len(err)}):")
        for e in err:
            print(f"  {e}")
        return 1
    print(f"All relative links resolve under {root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
