"""Load a small temporary corpus without models, Chroma, or API calls."""

import json

import pytest

from rag.ingestion import load_documents


@pytest.fixture
def corpus(tmp_path):
    policies = tmp_path / "data/policies"
    policies.mkdir(parents=True)
    entries = []
    for version, status, days in (("1.0", "superseded", 3), ("2.0", "current", 2)):
        identifier = f"remote-work-v{version}"
        source = f"data/policies/{identifier}.md"
        (tmp_path / source).write_text(f"# Remote Work v{version}\n\n{days} remote days weekly.\n")
        entries.append({
            "document_id": identifier, "policy_id": "remote-work",
            "version": version, "status": status, "source": source,
            "effective_date": "2025-01-01" if version == "1.0" else "2026-01-01",
            "supersedes": None if version == "1.0" else "remote-work-v1.0",
        })
    manifest = tmp_path / "data/manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "documents": entries}))
    return tmp_path, entries


def test_loading_preserves_exact_text_metadata_and_both_versions(corpus):
    root, entries = corpus
    documents = load_documents(root)
    assert documents == [
        {"id": entry["document_id"], "text": (root / entry["source"]).read_text(),
         "metadata": entry}
        for entry in entries
    ]
    assert {document["metadata"]["version"] for document in documents} == {"1.0", "2.0"}


def test_duplicate_document_ids_are_rejected(corpus):
    root, entries = corpus
    (root / "data/manifest.json").write_text(json.dumps({
        "schema_version": 1, "documents": [entries[0], entries[0]],
    }))
    with pytest.raises(ValueError, match="Duplicate document ID"):
        load_documents(root)


def test_missing_policy_is_reported(corpus):
    root, entries = corpus
    (root / entries[0]["source"]).unlink()
    with pytest.raises(FileNotFoundError):
        load_documents(root)


@pytest.mark.parametrize("source", ["data/outside.md", "data/policies/policy.txt"])
def test_sources_must_be_markdown_inside_the_policy_directory(corpus, source):
    root, entries = corpus
    (root / source).write_text("Content that must not be loaded.")
    entries[0]["source"] = source
    (root / "data/manifest.json").write_text(json.dumps({
        "schema_version": 1, "documents": entries,
    }))
    with pytest.raises(ValueError, match="Markdown file in data/policies"):
        load_documents(root)
