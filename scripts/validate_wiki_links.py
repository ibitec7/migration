#!/usr/bin/env python3
"""Validate all [[wikilinks]] in the wiki/ directory resolve to existing pages."""

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from wiki_common import collect_pages, extract_wikilinks  # noqa: E402

WIKI_DIR = Path(__file__).resolve().parent.parent / "wiki"


def main() -> int:
    if not WIKI_DIR.is_dir():
        print(f"ERROR: wiki directory not found at {WIKI_DIR}")
        return 1

    lookup = collect_pages(WIKI_DIR)
    print(f"Indexed {len(lookup)} title/alias/filename entries from wiki pages.\n")

    broken: list[tuple[str, str, str]] = []
    total_links = 0

    for md in sorted(WIKI_DIR.rglob("*.md")):
        if ".obsidian" in md.parts:
            continue
        text = md.read_text(encoding="utf-8")
        links = extract_wikilinks(text)
        total_links += len(links)
        for link in links:
            key = link.lower()
            if key not in lookup:
                rel = md.relative_to(WIKI_DIR)
                broken.append((str(rel), link, key))

    n_pages = len([p for p in WIKI_DIR.rglob("*.md") if ".obsidian" not in p.parts])
    print(f"Scanned {total_links} wikilinks across {n_pages} files.\n")

    if broken:
        print(f"BROKEN LINKS ({len(broken)}):")
        for source, link, _key in broken:
            print(f"  {source}  →  [[{link}]]")
        return 1
    print("All wikilinks resolve successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
