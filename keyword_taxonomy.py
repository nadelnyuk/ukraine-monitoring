"""
keyword_taxonomy.py — Parse keyword taxonomy markdown into matching rules.

The markdown is structured as:

    ## CATEGORY N: Name
    ### N.M Subcategory Name
    **Russian (RU)**
    phrase1, phrase2, phrase3, ...
    **Ukrainian (UA)**
    phrase1, phrase2, phrase3, ...

Call `load_taxonomy(path)` to get a list of SubcategoryEntry dicts.

Each entry contains a combined phrase list (RU + UA, lowercased, NFC-normalised).
Phrases are used as exact substrings — if ANY phrase from a subcategory appears
in a post's cleaned text, that post counts as a match for that subcategory.

To update matching rules: edit the markdown file, then rerun
`python process_topic_mentions.py --full`.

AUDITING A MATCH
----------------
To find all posts on a date that matched a specific phrase:

    python3 -c "
    import json, re, html, unicodedata
    date = '2026-04-15'
    phrase = 'нелегітимний президент'
    path = '/Volumes/Transcend/.../posts_by_day/' + date + '.json'
    count = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            text = re.sub(r'<[^>]+>', ' ', d.get('text', ''))
            text = html.unescape(text)
            text = unicodedata.normalize('NFC', text).lower()
            if phrase in text:
                count += 1
                print(text[:300])
                print('---')
    print('Total:', count)
    "
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TypedDict


class SubcategoryEntry(TypedDict):
    category_id: str      # e.g. "1"
    category: str         # e.g. "Political"
    subcategory_id: str   # e.g. "1.1"
    subcategory: str      # e.g. "Presidential legitimacy under martial law"
    phrases: list[str]    # combined RU + UA phrases, lowercased, NFC-normalised


# ── Regexes for recognising markdown structure ─────────────────────────────────

# Matches: ## CATEGORY 1: Political
_RE_CATEGORY = re.compile(r"^##\s+CATEGORY\s+(\d+):\s+(.+)$")

# Matches: ### 1.1 Presidential legitimacy under martial law
_RE_SUBCATEGORY = re.compile(r"^###\s+(\d+\.\d+)\s+(.+)$")

# Matches: **Russian (RU)** or **Ukrainian (UA)**
_RE_LANG_HEADER = re.compile(r"^\*\*(?:Russian|Ukrainian)\s*\((?:RU|UA)\)\*\*$")


def _normalise(raw: str) -> str:
    """Lowercase, NFC-normalise, strip whitespace and markdown escape characters.

    Markdown uses backslash to escape special chars (e.g. \\#hashtag → #hashtag).
    We strip the leading backslash so phrases match real post text.
    """
    text = unicodedata.normalize("NFC", raw.strip())
    text = text.lstrip("\\")   # remove markdown escape: \\#tag → #tag
    return text.lower()


def load_taxonomy(md_path: Path) -> list[SubcategoryEntry]:
    """Parse keyword taxonomy markdown into a list of SubcategoryEntry dicts.

    Phrases from the Russian (RU) and Ukrainian (UA) blocks are merged into a
    single list per subcategory. Duplicates are silently dropped.

    Args:
        md_path: Path to the .md taxonomy file.

    Returns:
        List of SubcategoryEntry dicts, one per subcategory, in document order.

    Raises:
        FileNotFoundError: If md_path does not exist.
        ValueError: If no subcategories were parsed (likely a format problem).
    """
    entries: list[SubcategoryEntry] = []
    current_cat_id = ""
    current_cat = ""
    current_entry: SubcategoryEntry | None = None
    collecting = False   # True once we are inside a RU/UA phrase block

    with open(md_path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n").strip()

            # ── Category heading  (## CATEGORY N: Name) ──────────────────
            m = _RE_CATEGORY.match(line)
            if m:
                current_cat_id = m.group(1)
                current_cat = m.group(2).strip()
                collecting = False
                continue

            # ── Subcategory heading  (### N.M Title) ─────────────────────
            m = _RE_SUBCATEGORY.match(line)
            if m:
                current_entry = {
                    "category_id": current_cat_id,
                    "category": current_cat,
                    "subcategory_id": m.group(1),
                    "subcategory": m.group(2).strip(),
                    "phrases": [],
                }
                entries.append(current_entry)
                collecting = False
                continue

            # ── Language block header  (**Russian (RU)** / **Ukrainian (UA)**) ──
            if _RE_LANG_HEADER.match(line):
                collecting = True
                continue

            # ── Phrase line: collect comma-separated phrases ───────────────
            if collecting and current_entry is not None and line:
                # Rebuild dedup set from what's already collected (handles
                # the second language block being added to the same entry).
                seen = set(current_entry["phrases"])
                for raw_phrase in line.split(","):
                    phrase = _normalise(raw_phrase)
                    # Reject empty strings and single-char noise
                    if len(phrase) >= 2 and phrase not in seen:
                        current_entry["phrases"].append(phrase)
                        seen.add(phrase)

    if not entries:
        raise ValueError(
            f"No subcategories parsed from {md_path}. "
            "Check that the file uses the expected markdown structure."
        )

    return entries


def taxonomy_summary(entries: list[SubcategoryEntry]) -> str:
    """Return a human-readable summary suitable for a log message.

    Example output::

        1: Political
          1.1  Presidential legitimacy under martial law     — 30 phrases
          1.2  Authoritarian consolidation                   — 26 phrases
        ...
        Total subcategories: 19 | total phrases: 540
    """
    lines: list[str] = []
    current_cat = ""
    for e in entries:
        if e["category"] != current_cat:
            current_cat = e["category"]
            lines.append(f"  {e['category_id']}: {e['category']}")
        lines.append(
            f"    {e['subcategory_id']:<6}  {e['subcategory']:<50}  "
            f"— {len(e['phrases'])} phrases"
        )
    total_phrases = sum(len(e["phrases"]) for e in entries)
    lines.append(
        f"  Total subcategories: {len(entries)} | total phrases: {total_phrases}"
    )
    return "\n".join(lines)


# ── Quick self-check (run this file directly to verify parsing) ────────────────

if __name__ == "__main__":
    import sys

    md_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "Monitoring Keywords by Topic.md"
    )

    print(f"Parsing: {md_path}\n")
    entries = load_taxonomy(md_path)
    print(taxonomy_summary(entries))

    # Show a sample of phrases from the first 3 subcategories
    print("\nSample phrases (first 3 subcategories):")
    for entry in entries[:3]:
        print(f"\n  [{entry['subcategory_id']}] {entry['subcategory']}")
        for phrase in entry["phrases"][:6]:
            print(f"    • {phrase!r}")
        if len(entry["phrases"]) > 6:
            print(f"    … and {len(entry['phrases']) - 6} more")
