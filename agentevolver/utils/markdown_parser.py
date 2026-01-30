from __future__ import annotations

"""
Simple markdown section parser used by CMT memory modules.

Current usage patterns:
- The LLM is instructed to output several sections, each starting with a
  markdown level-1 heading, e.g.:

    # current step
    ...
    # previous instruction code
    ...

- Callers pass `expected_sections` such as:
    ["current step", "previous instruction code",
     "relevant environment feedback", "next-step instruction code"]

This utility extracts those sections by header name (case-insensitive),
falls back to a default placeholder for missing sections, and returns
two booleans:
  - `find_everything`: True  iff all expected sections are found and
    contain non-placeholder content.
  - `find_nothing`:    True  iff none of the expected sections are found.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class MarkdownSectionParseResult:
    sections: Dict[str, str]
    find_everything: bool
    find_nothing: bool


def _normalize_section_name(name: str) -> str:
    """Normalize a section name for matching."""
    return " ".join(name.strip().lower().split())


def read_markdown_and_extract_sections(
    markdown_text: str,
    expected_sections: Iterable[str],
    default_placeholder: str = "",
) -> Tuple[Dict[str, str], bool, bool]:
    """
    Parse markdown text and extract content under given headings.

    Args:
        markdown_text: Full markdown string from the LLM.
        expected_sections: Iterable of expected section names. Matching is
            case-insensitive and whitespace-insensitive. The keys in the
            returned dict will use the *original* strings from
            `expected_sections`.
        default_placeholder: Text used when a section is missing or empty.

    Returns:
        (sections_dict, find_everything, find_nothing)
    """
    # Preprocess expected section names
    expected_list: List[str] = list(expected_sections)
    normalized_to_original: Dict[str, str] = {
        _normalize_section_name(name): name for name in expected_list
    }
    normalized_expected = set(normalized_to_original.keys())

    # Storage for collected lines per normalized section name
    collected: Dict[str, List[str]] = {norm: [] for norm in normalized_expected}

    current_section_norm: str | None = None

    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip("\n")

        # Detect heading lines that start a new section.
        # We focus on level-1 headings (`# xxx`), which is what the prompts use.
        stripped = line.lstrip()
        if stripped.startswith("#"):
            # Count leading '#' then a space, e.g. "# title" / "## title"
            hash_part, _, title_part = stripped.partition(" ")
            if hash_part and hash_part.replace("#", "") == "":
                # We have something like "# title" or "## title"
                section_name_norm = _normalize_section_name(title_part)
                if section_name_norm in normalized_expected:
                    current_section_norm = section_name_norm
                    # Start a new section; do not carry over previous content.
                    collected[current_section_norm] = []
                    continue  # Do not record the heading line itself

        # Normal content line: if we are inside a known section, record it.
        if current_section_norm is not None:
            collected[current_section_norm].append(line)

    # Build final sections dict using original expected names as keys
    sections: Dict[str, str] = {}
    any_found = False
    all_found_and_non_placeholder = True

    for expected_name in expected_list:
        norm = _normalize_section_name(expected_name)
        lines = collected.get(norm, [])
        if lines:
            any_found = True
            content = "\n".join(lines).strip()
            if not content:
                content = default_placeholder
        else:
            content = default_placeholder
            all_found_and_non_placeholder = False

        sections[expected_name] = content

    if not any_found:
        # If we found no headings at all, we consider "find_everything" False
        # and "find_nothing" True, regardless of placeholders.
        find_everything = False
        find_nothing = True
    else:
        # If at least one heading matched, "find_nothing" is False.
        find_nothing = False
        # "find_everything" means all expected sections were matched AND
        # none of them is equal to the default placeholder.
        # We recompute this more robustly here.
        find_everything = all(
            _normalize_section_name(expected_name) in collected
            and sections[expected_name] != default_placeholder
            for expected_name in expected_list
        )

    return sections, find_everything, find_nothing



