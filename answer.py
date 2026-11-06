import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)
index = faiss.read_index("vector_store/research.index")

with Path("vector_store/chunks.json").open(
    "r", encoding="utf-8"
) as file:
    chunks = json.load(file)

question = input("Ask a question about the paper: ")

question_vector = embedder.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

scores, result_ids = index.search(question_vector, 3)

retrieved_chunks = [chunks[chunk_id] for chunk_id in result_ids[0]]

context_parts = []
for chunk in retrieved_chunks:
    context_parts.append(
        f"[Source: {chunk['source']}, page {chunk['page']}]\n"
        f"{chunk['text'][:600]}"
    )

context = "\n\n".join(context_parts)

prompt = f"""
Explain the answer in one clear, complete sentence using only the paper excerpts.
Do not answer with a fragment copied from an excerpt.
If the excerpts do not contain enough information, say that.
Question: {question}

Paper excerpts:
{context}

Answer:
"""

print("\nLoading the local answer model...")
answer_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
)

result = answer_model(
    prompt,
    max_new_tokens=180,
    truncation=True,
)

print("\nAnswer:")
print(result[0]["generated_text"])

print("\nRetrieved source pages:")
for chunk in retrieved_chunks:
    print(f"- {chunk['source']}, page {chunk['page']}")