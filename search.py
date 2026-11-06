import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/research.index")

with Path("vector_store/chunks.json").open(
    "r", encoding="utf-8"
) as file:
    chunks = json.load(file)

question = input("Ask a question about the paper: ")

question_vector = model.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

scores, result_ids = index.search(question_vector, 3)

for rank, chunk_id in enumerate(result_ids[0], start=1):
    chunk = chunks[chunk_id]

    print(
        f"\nResult {rank} — {chunk['source']}, "
        f"page {chunk['page']} (score: {scores[0][rank - 1]:.3f})"
    )
    print(chunk["text"])