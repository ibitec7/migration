#!/usr/bin/env python3
"""Export wiki/ to GitHub Wiki–compatible Markdown (relative links, Home.md, _Sidebar.md)."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Allow `python scripts/sync_github_wiki.py` from repo root
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from wiki_common import (  # noqa: E402
    collect_pages,
    page_title,
    relative_wiki_href,
    strip_frontmatter,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
WIKI_DIR = REPO_ROOT / "wiki"


def split_wikilink_inner(raw: str) -> tuple[str, str | None]:
    """Match validate_wiki_links key extraction; return (target_key, display or None)."""
    parts = re.split(r"\\?\|", raw)
    target = parts[0].strip()
    if len(parts) > 1:
        display = "|".join(p.replace("\\|", "|") for p in parts[1:]).strip()
        return target, display or None
    return target, None


def transform_wikilinks(
    text: str,
    *,
    current_source: Path,
    lookup: dict[str, Path],
    wiki_root: Path,
    title_cache: dict[Path, str],
) -> str:
    """Replace [[wikilinks]] with relative Markdown links; skip fenced code blocks."""

    def repl_one(raw: str) -> str:
        key, display_override = split_wikilink_inner(raw)
        lk = key.lower()
        if lk not in lookup:
            raise ValueError(f"Unresolved wikilink [[{raw}]] in {current_source}")
        target_path = lookup[lk]
        if display_override:
            label = display_override
        else:
            if target_path not in title_cache:
                title_cache[target_path] = page_title(target_path)
            label = title_cache[target_path]
        href = relative_wiki_href(current_source, target_path, wiki_root)
        return f"[{label}]({href})"

    def replace_in_line(line: str) -> str:
        def sub(m: re.Match[str]) -> str:
            return repl_one(m.group(1))

        # Preserve inline `code` (same idea as extract_wikilinks)
        chunks = re.split(r"(`[^`]*`)", line)
        out: list[str] = []
        for chunk in chunks:
            if len(chunk) >= 2 and chunk[0] == "`" and chunk[-1] == "`":
                out.append(chunk)
            else:
                out.append(re.sub(r"\[\[([^\]]+)\]\]", sub, chunk))
        return "".join(out)

    out: list[str] = []
    in_fence = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        out.append(replace_in_line(line))
    return "".join(out)


def strip_obsidian_tip_and_optional(text: str) -> str:
    """Remove Obsidian-only tip blockquote and Optional Enhancements section from index copy."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping_optional = False
    for line in lines:
        if line.startswith("## Optional Enhancements"):
            skipping_optional = True
            continue
        if skipping_optional:
            continue
        if line.startswith("> **Tip**:") or (out and out[-1].startswith("> ") and line.startswith("> ")):
            # Drop blockquote: skip consecutive > lines
            continue
        if line.startswith("> "):
            continue
        out.append(line)
    return "".join(out)


def build_sidebar_body(index_after_home_transform: str) -> str:
    """Sidebar: same nav as index but without intro paragraph and pipeline code block."""
    text = index_after_home_transform
    # Drop first heading and welcome paragraph (until first --- or ##)
    m = re.search(r"^## ", text, flags=re.MULTILINE)
    if m:
        text = text[m.start() :]
    # Remove fenced code block under Pipeline
    text = re.sub(
        r"\n```[^\n]*\n[^`]*```\s*\n",
        "\n",
        text,
        count=1,
    )
    return "# Wiki navigation\n\n" + text.strip() + "\n"


def export_wiki(out_dir: Path, *, write_sidebar: bool) -> None:
    if not WIKI_DIR.is_dir():
        raise SystemExit(f"ERROR: wiki directory not found at {WIKI_DIR}")

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    lookup = collect_pages(WIKI_DIR)
    title_cache: dict[Path, str] = {}

    # First pass: copy all files with transforms (resolve index → Home)
    for md in sorted(WIKI_DIR.rglob("*.md")):
        if ".obsidian" in md.parts:
            continue
        rel = md.relative_to(WIKI_DIR)
        raw = md.read_text(encoding="utf-8")
        body = strip_frontmatter(raw)

        if md.name.lower() == "index.md":
            source_for_links = md
            body = strip_obsidian_tip_and_optional(body)
        else:
            source_for_links = md

        transformed = transform_wikilinks(
            body,
            current_source=source_for_links,
            lookup=lookup,
            wiki_root=WIKI_DIR,
            title_cache=title_cache,
        )

        if md.name.lower() == "index.md":
            dest = out_dir / "Home.md"
        else:
            dest = out_dir / rel

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(transformed, encoding="utf-8")

    if write_sidebar:
        home_path = out_dir / "Home.md"
        home_text = home_path.read_text(encoding="utf-8")
        sidebar = build_sidebar_body(home_text)
        (out_dir / "_Sidebar.md").write_text(sidebar, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Export wiki/ to GitHub Wiki Markdown under OUT_DIR.",
        epilog=(
            "Prerequisites: enable Wiki on GitHub (Settings → General → Features). "
            "Clone the wiki repo with: git clone https://github.com/OWNER/REPO.wiki.git "
            "then copy OUT_DIR contents into that clone and push."
        ),
    )
    p.add_argument(
        "out_dir",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "wiki_export",
        help="Output directory (default: ./wiki_export)",
    )
    p.add_argument(
        "--no-sidebar",
        action="store_true",
        help="Do not write _Sidebar.md",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="After export, run validate_wiki_export.py on OUT_DIR",
    )
    args = p.parse_args()
    try:
        export_wiki(args.out_dir, write_sidebar=not args.no_sidebar)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Exported GitHub Wiki files to {args.out_dir.resolve()}")
    if args.validate:
        from validate_wiki_export import validate_export

        err = validate_export(args.out_dir)
        if err:
            print(f"ERROR: {len(err)} broken link(s) in export:", file=sys.stderr)
            for e in err:
                print(f"  {e}", file=sys.stderr)
            return 1
        print("Export link check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
