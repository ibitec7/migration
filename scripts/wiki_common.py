"""Shared helpers for Obsidian-style wiki/ and GitHub Wiki export."""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "collect_pages",
    "extract_wikilinks",
    "strip_frontmatter",
    "page_title",
    "relative_wiki_href",
]


def collect_pages(wiki_dir: Path) -> dict[str, Path]:
    """Map lowercase title / alias / filename stem → markdown path."""
    lookup: dict[str, Path] = {}
    for md in wiki_dir.rglob("*.md"):
        if ".obsidian" in md.parts:
            continue
        stem = md.stem.lower()
        lookup[stem] = md
        text = md.read_text(encoding="utf-8")
        in_frontmatter = False
        for line in text.splitlines():
            if line.strip() == "---":
                if not in_frontmatter:
                    in_frontmatter = True
                    continue
                break
            if in_frontmatter:
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip("\"'")
                    lookup[title.lower()] = md
                elif line.startswith("aliases:"):
                    aliases_str = line.split(":", 1)[1].strip()
                    if aliases_str.startswith("["):
                        aliases_str = aliases_str.strip("[]")
                        for alias in aliases_str.split(","):
                            a = alias.strip().strip("\"'")
                            if a:
                                lookup[a.lower()] = md
    return lookup


def extract_wikilinks(text: str) -> list[str]:
    """Extract [[target]] wikilink keys (target part only, before display pipe)."""
    cleaned = re.sub(r"`[^`]+`", "", text)
    targets: list[str] = []
    for m in re.finditer(r"\[\[([^\]]+)\]\]", cleaned):
        raw = m.group(1)
        target = re.split(r"\\?\|", raw)[0].strip()
        if target:
            targets.append(target)
    return targets


def strip_frontmatter(text: str) -> str:
    """Remove leading YAML frontmatter delimited by --- lines."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[i + 1 :]).lstrip("\n")
    return text


def page_title(path: Path) -> str:
    """Prefer `title:` from frontmatter; else Title-Case the filename stem."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        in_fm = False
        for line in text.splitlines():
            if line.strip() == "---":
                if not in_fm:
                    in_fm = True
                    continue
                break
            if in_fm and line.startswith("title:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    return path.stem.replace("-", " ").title()


def relative_wiki_href(from_md: Path, to_md: Path, wiki_root: Path) -> str:
    """Relative href from one wiki page to another (no .md), for GitHub Wiki."""
    import os

    wiki_root = wiki_root.resolve()
    from_md = from_md.resolve()
    to_md = to_md.resolve()

    if from_md.name.lower() == "index.md":
        from_base = wiki_root
    else:
        from_base = from_md.parent

    # index.md is exported as Home.md; wiki links must target Home, not index
    if to_md.name.lower() == "index.md":
        dest = wiki_root / "Home"
    else:
        dest = (wiki_root / to_md.relative_to(wiki_root)).with_suffix("")

    rel = Path(os.path.relpath(dest, from_base)).as_posix()
    if rel == ".":
        return ""
    return rel
