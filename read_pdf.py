from pathlib import Path
from pypdf import PdfReader


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


pdf_path = Path("data/documents/rag-paper.pdf")
reader = PdfReader(pdf_path)

all_chunks = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""

    for chunk_number, chunk_text in enumerate(
        split_into_chunks(text), start=1
    ):
        all_chunks.append({
            "text": chunk_text,
            "source": pdf_path.name,
            "page": page_number,
            "chunk": chunk_number,
        })

print(f"Created {len(all_chunks)} chunks.")

for chunk in all_chunks[:3]:
    print(
        f"\nSource: {chunk['source']}, "
        f"page {chunk['page']}, chunk {chunk['chunk']}"
    )
    print(chunk["text"][:250])