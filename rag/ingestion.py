"""Load policy text and document metadata from the corpus manifest."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_documents(project_root: Path = ROOT) -> list[dict]:
    """Return records with id, text, and metadata, retaining every policy version."""
    root = Path(project_root).resolve()
    manifest = json.loads((root / "data/manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int:
        raise ValueError("Manifest must be an object with integer schema_version 1")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema version")
    entries = manifest.get("documents")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must contain a nonempty documents array")

    documents = []
    seen_ids = set()
    policy_directory = (root / "data/policies").resolve()
    for entry in entries:
        document_id = entry["document_id"]
        if document_id in seen_ids:
            raise ValueError(f"Duplicate document ID: {document_id}")
        source = (root / entry["source"]).resolve()
        if source.suffix != ".md" or not source.is_relative_to(policy_directory):
            raise ValueError(f"{document_id}: source must be a Markdown file in data/policies")
        text = source.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"{document_id}: policy text is empty")
        documents.append({"id": document_id, "text": text, "metadata": entry.copy()})
        seen_ids.add(document_id)
    return documents


if __name__ == "__main__":
    documents = load_documents()
    print(f"Loaded {len(documents)} policy documents")
    for document in documents:
        metadata = document["metadata"]
        print(f"{document['id']} | {metadata['status']} | {len(document['text'])} characters")
