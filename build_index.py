import json
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


def split_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


documents_folder = Path("data/documents")
pdf_paths = sorted(documents_folder.glob("*.pdf"))

if not pdf_paths:
    raise FileNotFoundError(f"No PDF files found in {documents_folder}")

all_chunks = []

for pdf_path in pdf_paths:
    reader = PdfReader(str(pdf_path))
    pdf_chunk_count = 0

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, chunk_text in enumerate(
            split_into_chunks(page_text), start=1
        ):
            all_chunks.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "page": page_number,
                "chunk": chunk_number,
            })
            pdf_chunk_count += 1

    print(f"{pdf_path.name}: created {pdf_chunk_count} chunks")

if not all_chunks:
    raise ValueError("No readable text was found in the PDF files.")

print(f"Embedding {len(all_chunks)} chunks...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [chunk["text"] for chunk in all_chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

output_dir = Path("vector_store")
output_dir.mkdir(exist_ok=True)

faiss.write_index(index, str(output_dir / "research.index"))

with (output_dir / "chunks.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(all_chunks, file, ensure_ascii=False, indent=2)

print(
    f"Saved {index.ntotal} chunks from "
    f"{len(pdf_paths)} PDF(s) in {output_dir}."
)