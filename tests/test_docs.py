"""Docs integrity: README links to every ADR and referenced source modules exist."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_links_resolve_to_existing_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(((?:src|docs|results|deploy)/[^)#\s]+)\)", readme)
    assert len(links) >= 10
    missing = [link for link in links if not (ROOT / link).exists()]
    assert not missing, f"README links to missing files: {missing}"


def test_adrs_have_required_sections():
    adrs = sorted((ROOT / "docs" / "adr").glob("*.md"))
    assert len(adrs) >= 3
    for adr in adrs:
        text = adr.read_text(encoding="utf-8")
        for section in ("## Context", "## Decision", "## Consequences", "**Status:**"):
            assert section in text, f"{adr.name} missing {section}"
