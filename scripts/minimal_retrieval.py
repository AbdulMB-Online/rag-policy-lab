"""Embed two known texts, store them in Chroma, and check their retrieval."""

import chromadb
from sentence_transformers import SentenceTransformer


MODEL = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RECORDS = [
    ("remote-work", "Eligible employees may work remotely up to 2 days per calendar "
     "week, subject to manager approval."),
    ("expense-meals", "Meals: $75 per employee per calendar day. Actual eligible "
     "food and nonalcoholic drinks during business travel, including tax and tips."),
]
QUESTIONS = [
    ("How many days can I work from home each week?", "remote-work"),
    ("What is the daily meal reimbursement limit during business travel?", "expense-meals"),
]


def main():
    print(f"Model: {MODEL}", flush=True)
    model = SentenceTransformer(MODEL, device="cpu")
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="minimal-retrieval",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )
    print("Metric: cosine distance (lower means more similar)")

    # Store one record first, then add the second to prove incremental storage.
    for expected_count, (record_id, text) in enumerate(RECORDS, start=1):
        embedding = model.encode([text], normalize_embeddings=True)
        if embedding.shape != (1, 384):
            raise RuntimeError(f"Unexpected embedding shape: {embedding.shape}")
        collection.add(ids=[record_id], documents=[text], embeddings=embedding.tolist())
        count = collection.count()
        if count != expected_count:
            raise RuntimeError(f"Expected {expected_count} stored records, got {count}")
        print(f"Stored {record_id}: embedding shape {embedding.shape}; collection count {count}")

    # Embed questions with the same model, then let Chroma rank both records.
    for question, expected_id in QUESTIONS:
        query_embedding = model.encode([QUERY_INSTRUCTION + question], normalize_embeddings=True)
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=2,
            include=["documents", "distances"],
        )
        print(f"\nQuestion: {question}")
        for rank, (record_id, text, distance) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["distances"][0]
        ), start=1):
            print(f"{rank}. {record_id} | cosine distance: {distance:.4f}\n   {text}")
        if len(results["ids"][0]) != 2 or results["ids"][0][0] != expected_id:
            raise RuntimeError(f"Expected {expected_id} as the first of two results")
        print(f"PASS: expected first result is {expected_id}")

    print("\nPASS: minimal embed-store-retrieve loop; 2/2 questions passed")


if __name__ == "__main__":
    main()
