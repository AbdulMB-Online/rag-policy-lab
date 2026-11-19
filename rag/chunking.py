"""Section-aware chunks with overlapping prose and intact Markdown tables.

The 440-token cap includes headings and special tokens, below BGE's 512-token
limit. Prose overlaps by up to 80 tokens; oversized tables raise an error.
"""

import re


TABLE = re.compile(
    r"(?m)(^[A-Z]+-T\d{2}: [^\n]+\n(?:[ \t]*\n)+\|[^\n]*(?:\n\|[^\n]*)*)"
)


def token_count(tokenizer, text, add_special_tokens=True):
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens,
                                truncation=False, verbose=False))


def sections(text):
    """Yield heading ancestry and body text for each nonempty section."""
    headings = list(re.finditer(r"(?m)^(#+) (.+)$", text))
    if not headings or headings[0].start() != 0 or len(headings[0][1]) != 1:
        raise ValueError("Policy must start with an H1 title")
    ancestry = []
    for index, heading in enumerate(headings):
        level = len(heading[1])
        ancestry = ancestry[:level - 1] + [heading[0]]
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end():end].strip()
        if body:
            yield ancestry.copy(), body


def prose_chunks(body, context, tokenizer, max_tokens, overlap):
    """Add whole words up to the cap, then carry over a small tail of context."""
    words = list(re.finditer(r"\S+", body))
    start = 0
    while start < len(words):
        end = start
        while end < len(words):
            candidate = context + "\n\n" + body[words[start].start():words[end].end()]
            if token_count(tokenizer, candidate) > max_tokens:
                break
            end += 1
        if end == start:
            raise ValueError("A word cannot fit with its heading context")
        yield context + "\n\n" + body[words[start].start():words[end - 1].end()]
        if end == len(words):
            break
        following = end
        while following > start + 1:
            tail = body[words[following - 1].start():words[end - 1].end()]
            if token_count(tokenizer, tail, add_special_tokens=False) > overlap:
                break
            following -= 1
        start = following


def chunk_documents(documents, tokenizer, max_tokens=440, overlap=80):
    """Return chunk records with IDs, text, and inherited/section metadata."""
    if not 0 <= overlap < max_tokens <= 512:
        raise ValueError("Require 0 <= overlap < max_tokens <= 512 for BGE")
    chunks = []
    for document in documents:
        for ancestry, body in sections(document["text"]):
            context = "\n".join(ancestry)
            names = [heading.lstrip("# ") for heading in ancestry]
            section_id = names[-1].split(":", 1)[0] if len(names) > 1 else "preamble"
            chunk_index = 0
            for block in TABLE.split(body):
                block = block.strip()
                if not block:
                    continue
                is_table = TABLE.fullmatch(block) is not None
                if is_table:
                    fragments = [context + "\n\n" + block]
                else:
                    if re.search(r"(?m)^\|", block):
                        raise ValueError(f"{section_id}: table needs a caption")
                    fragments = prose_chunks(block, context, tokenizer, max_tokens, overlap)
                for text in fragments:
                    count = token_count(tokenizer, text)
                    if count > max_tokens:
                        kind = "table" if is_table else "text chunk"
                        raise ValueError(f"{section_id}: {kind} exceeds {max_tokens} tokens")
                    metadata = {
                        **document["metadata"], "section_id": section_id,
                        "section_path": " > ".join(names), "chunk_index": chunk_index,
                        "token_count": count, "content_type": "table" if is_table else "text",
                    }
                    if is_table:
                        metadata["table_id"] = block.split(":", 1)[0]
                    chunks.append({
                        "id": f"{document['id']}:{section_id}:{chunk_index}",
                        "text": text, "metadata": metadata,
                    })
                    chunk_index += 1
    return chunks


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from rag.ingestion import load_documents

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    chunks = chunk_documents(load_documents(), tokenizer)
    largest = max(c["metadata"]["token_count"] for c in chunks)
    print(f"Created {len(chunks)} chunks; largest: {largest}/440 tokens")
