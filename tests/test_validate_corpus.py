"""Focused checks for our corpus validator; leave the real policies unchanged."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import validate_corpus as validator


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def documents():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    return {d["document_id"]: d for d in manifest["documents"]}


def policy_text(document):
    return (ROOT / document["source"]).read_text(encoding="utf-8")


def test_real_corpus_passes():
    summaries = validator.validate_corpus(ROOT)
    assert len(summaries) == 5
    assert all(summary.startswith("PASS ") for summary in summaries)


@pytest.mark.parametrize("field,value,message", [
    ("title", "Wrong policy", "incorrect title"),
    ("version", "latest", "invalid version"),
    ("effective_date", "2026-02-30", "day"),
    ("status", "superseded", "incorrect status"),
    ("source", "../outside.md", "incorrect source"),
    ("supersedes", None, "incorrect supersedes"),
])
def test_invalid_metadata_fails(field, value, message):
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    manifest["documents"][1][field] = value
    with pytest.raises(ValueError, match=message):
        validator.check_metadata(manifest)


def test_duplicate_document_fails():
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    manifest["documents"][1] = manifest["documents"][0].copy()
    with pytest.raises(ValueError, match="duplicate document IDs"):
        validator.check_metadata(manifest)


def test_word_counter_excludes_markdown_syntax():
    text = """# Ignore this title

1. Submit **claim**.

EXP-T01: Evidence

| Kind | Limit | Proof |
| --- | --- | --- |
| Meal | $75 | Receipt |
"""
    assert validator.count_words(text) == 10


@pytest.mark.parametrize("target", [1199, 1200, 1500, 1501])
def test_word_count_boundaries(documents, target):
    # A minimal valid document isolates the length rule from other content rules.
    text = """# Remote Work Policy v1.0

## RW-01: Purpose and scope
Content.
## RW-02: Roles and eligibility
Content.
## RW-03: Policy rules
Content.
### RW-03.1: Weekly allowance
Content.
## RW-04: Procedures
1. Submit.
2. Review.
## RW-05: Exceptions and escalation
Content.
## RW-06: Ownership and review
Content.
"""
    padding = target - validator.count_words(text) + 1
    text = text.replace("Content.", " ".join(["word"] * padding), 1)
    assert validator.count_words(text) == target
    if 1200 <= target <= 1500:
        assert validator.check_structure(documents["remote-work-v1.0"], text) == target
    else:
        with pytest.raises(ValueError, match="expected 1200–1500"):
            validator.check_structure(documents["remote-work-v1.0"], text)


@pytest.mark.parametrize("old,new,message", [
    ("# Remote Work Policy v1.0", "# Wrong title", "matching H1"),
    ("## RW-02: Roles and eligibility", "## RW-02: Wrong section", "H2 sections"),
    ("### RW-03.1:", "#### RW-03.1:", "H1/H2/H3"),
    ("### RW-03.1:", "### RW-02.1:", "wrong parent"),
    ("### RW-03.2:", "### RW-03.1:", "Duplicate section IDs"),
    ("2. The manager reviews", "3. The manager reviews", "sequential numbered steps"),
])
def test_broken_structure_fails(documents, old, new, message):
    document = documents["remote-work-v1.0"]
    text = policy_text(document).replace(old, new, 1)
    with pytest.raises(ValueError, match=message):
        validator.check_structure(document, text)


@pytest.mark.parametrize("old,new,message", [
    ("EXP-T01:", "EXP-T99:", "Expected table IDs"),
    ("| --- | --- | --- |", "| bad | --- | --- |", "delimiter"),
    ("| Meals | $75", "| Meals | extra | $75", "column counts"),
    ("| Expense category |", "|  |", "Empty table header"),
    ("| Meals | $75 per employee per calendar day |", "| Meals |", "column counts"),
])
def test_broken_tables_fail(documents, old, new, message):
    document = documents["expense-reimbursement-v1.0"]
    text = policy_text(document).replace(old, new, 1)
    with pytest.raises(ValueError, match=message):
        validator.check_tables(document, text)


@pytest.mark.parametrize("change", ["remove_conflict", "extra_difference"])
def test_unintended_remote_work_changes_fail(documents, change):
    texts = {key: policy_text(documents[key]) for key in ("remote-work-v1.0", "remote-work-v2.0")}
    if change == "remove_conflict":
        texts["remote-work-v2.0"] = texts["remote-work-v2.0"].replace("up to 2 days", "up to 3 days")
        message = "expected one 2-day allowance"
    else:
        texts["remote-work-v2.0"] += "\nAn unintended extra rule.\n"
        message = "additional differences"
    with pytest.raises(ValueError, match=message):
        validator.check_conflict(texts)


@pytest.mark.parametrize("change", ["missing_source", "extra_source", "invalid_json"])
def test_invalid_corpus_copy_fails(tmp_path, change):
    # Mutate a temporary copy, never the real corpus or its planted conflict.
    shutil.copytree(ROOT / "data", tmp_path / "data")
    if change == "missing_source":
        (tmp_path / "data/policies/pto-v1.0.md").unlink()
    elif change == "extra_source":
        (tmp_path / "data/policies/unregistered.md").write_text("Extra policy", encoding="utf-8")
    else:
        (tmp_path / "data/manifest.json").write_text("{invalid JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        validator.validate_corpus(tmp_path)


def test_command_runs_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / "scripts/validate_corpus.py")],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS planted conflict preserved" in result.stdout
