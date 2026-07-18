"""Offline Milestone 9C bibliography and citation consistency checks.

This script reads documentation and tabular artifacts only. It does not import
dataset, model, training, calibration, or inference code.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "references" / "final_references.bib"
MANUSCRIPT_PATH = ROOT / "docs" / "FINAL_MANUSCRIPT_WITH_CITATIONS.md"
REVIEW_PATH = ROOT / "docs" / "VERIFIED_LITERATURE_REVIEW.md"
REFERENCE_TABLE_PATH = ROOT / "results" / "tables" / "exp09c_verified_references.csv"
UNRESOLVED_PATH = ROOT / "results" / "tables" / "exp09c_unresolved_citation_gaps.csv"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def parse_bibtex(text: str) -> dict[str, dict[str, str]]:
    starts = list(re.finditer(r"(?m)^@([A-Za-z]+)\{([^,\s]+),\s*$", text))
    if not starts:
        fail("no BibTeX entries found")

    entries: dict[str, dict[str, str]] = {}
    for index, match in enumerate(starts):
        start = match.end()
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[start:end].strip()
        if not block.endswith("}"):
            fail(f"BibTeX entry {match.group(2)} does not end with a closing brace")
        key = match.group(2)
        if key in entries:
            fail(f"duplicate BibTeX key: {key}")
        fields: dict[str, str] = {"entry_type": match.group(1).lower()}
        for field_match in re.finditer(
            r"(?m)^\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*\{(.*)\},?\s*$", block
        ):
            fields[field_match.group(1).lower()] = field_match.group(2).strip()
        if not fields.get("title") or not fields.get("author") or not fields.get("year"):
            fail(f"BibTeX entry {key} lacks title, author, or year")
        entries[key] = fields

    if text.count("{") != text.count("}"):
        fail("BibTeX file has unbalanced braces")
    return entries


def main() -> None:
    bib_text = BIB_PATH.read_text(encoding="utf-8")
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    entries = parse_bibtex(bib_text)

    manuscript_citations = set(re.findall(r"@([A-Za-z0-9_:-]+)", manuscript))
    package_citations = manuscript_citations | set(
        re.findall(r"@([A-Za-z0-9_:-]+)", review)
    )
    undefined = sorted(package_citations - set(entries))
    if undefined:
        fail(f"undefined citation keys: {undefined}")

    if "[CITATION NEEDED:" in manuscript or "[CITATION UNRESOLVED:" in manuscript:
        fail("citation gap marker remains in cited manuscript")

    duplicate_dois: dict[str, list[str]] = {}
    duplicate_titles: dict[str, list[str]] = {}
    for key, fields in entries.items():
        doi = normalized(fields.get("doi", ""))
        if doi:
            duplicate_dois.setdefault(doi, []).append(key)
        title = normalized(fields["title"])
        duplicate_titles.setdefault(title, []).append(key)
    duplicate_dois = {k: v for k, v in duplicate_dois.items() if len(v) > 1}
    duplicate_titles = {k: v for k, v in duplicate_titles.items() if len(v) > 1}
    if duplicate_dois:
        fail(f"duplicate DOI records: {duplicate_dois}")
    if duplicate_titles:
        fail(f"duplicate title records: {duplicate_titles}")

    with REFERENCE_TABLE_PATH.open(encoding="utf-8", newline="") as handle:
        reference_rows = list(csv.DictReader(handle))
    if len(reference_rows) != len(entries):
        fail(
            f"verified-reference table has {len(reference_rows)} rows; "
            f"BibTeX has {len(entries)}"
        )
    table_keys = [row["citation_key"] for row in reference_rows]
    if len(table_keys) != len(set(table_keys)):
        fail("duplicate citation key in verified-reference table")
    if set(table_keys) != set(entries):
        fail("verified-reference table and BibTeX key sets differ")
    row_by_key = {row["citation_key"]: row for row in reference_rows}
    metadata_mismatches = []
    for key, fields in entries.items():
        row = row_by_key[key]
        if normalized(row["title"]) != normalized(fields["title"]):
            metadata_mismatches.append(f"{key}: title")
        if row["year"].strip().casefold() != fields["year"].strip().casefold():
            metadata_mismatches.append(f"{key}: year")
        if normalized(row.get("doi", "")) != normalized(fields.get("doi", "")):
            metadata_mismatches.append(f"{key}: doi")
    if metadata_mismatches:
        fail(f"BibTeX/table metadata mismatches: {metadata_mismatches}")
    unverified = [
        row["citation_key"]
        for row in reference_rows
        if row["verification_status"].strip().casefold() != "verified"
    ]
    if unverified:
        fail(f"unverified bibliography rows: {unverified}")

    unused = sorted(set(entries) - manuscript_citations)
    unlabelled_unused = []
    for key in unused:
        row = row_by_key[key]
        label_text = " ".join(
            [row.get("manuscript_sections", ""), row.get("notes", "")]
        ).casefold()
        if "background" not in label_text and "verified literature review" not in label_text:
            unlabelled_unused.append(key)
    if unlabelled_unused:
        fail(f"unused BibTeX entries lack a background label: {unlabelled_unused}")

    with UNRESOLVED_PATH.open(encoding="utf-8", newline="") as handle:
        unresolved_rows = list(csv.DictReader(handle))

    print("PASS: Milestone 9C citation audit")
    print(f"bibtex_entries={len(entries)}")
    print(f"manuscript_citation_keys={len(manuscript_citations)}")
    print(f"package_citation_keys={len(package_citations)}")
    print(f"background_or_review_only_entries={len(unused)}")
    print(f"unresolved_citation_gaps={len(unresolved_rows)}")
    print("duplicate_doi_records=0")
    print("duplicate_title_records=0")
    print("dataset_loaded=false")
    print("ood_test_accessed=false")
    print("inference_run=false")


if __name__ == "__main__":
    main()
