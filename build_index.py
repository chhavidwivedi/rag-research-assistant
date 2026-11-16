
import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from read_pdf import extract_pdf


# ----------------------------------
# CONFIGURATION
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_FOLDER = BASE_DIR / "data" / "documents"
VECTOR_FOLDER = BASE_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 32


# ----------------------------------
# SENTENCE-AWARE CHUNKING
# ----------------------------------


def split_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Split text at sentence boundaries while preserving
    character positions in the normalized page text.
    """

    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Find sentence boundaries and preserve their positions.
    sentence_spans = [
        (match.start(), match.end())
        for match in re.finditer(
            r".+?(?:[.!?](?=\s|$)|$)",
            text
        )
        if match.group().strip()
    ]

    if not sentence_spans:
        return []

    chunks = []
    start_idx = 0

    while start_idx < len(sentence_spans):

        chunk_start = sentence_spans[start_idx][0]
        end_idx = start_idx

        # Build a chunk without cutting normal sentences.
        while end_idx < len(sentence_spans):

            sentence_end = sentence_spans[end_idx][1]

            if sentence_end - chunk_start > chunk_size:
                break

            end_idx += 1

        # Handle a sentence longer than the chunk size.
        if end_idx == start_idx:

            sentence_start, sentence_end = sentence_spans[start_idx]

            for position in range(
                sentence_start,
                sentence_end,
                chunk_size - overlap
            ):

                piece_end = min(
                    position + chunk_size,
                    sentence_end
                )

                piece = text[position:piece_end].strip()

                if piece:
                    chunks.append({
                        "text": piece,
                        "char_start": position,
                        "char_end": piece_end
                    })

                if piece_end >= sentence_end:
                    break

            start_idx += 1
            continue

        chunk_end = sentence_spans[end_idx - 1][1]

        chunk_text = text[chunk_start:chunk_end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "char_start": chunk_start,
                "char_end": chunk_end
            })

        if end_idx >= len(sentence_spans):
            break

        # Find the next sentence near the overlap boundary.
        target_position = chunk_end - overlap

        next_start_idx = start_idx + 1

        while (
            next_start_idx < end_idx
            and sentence_spans[next_start_idx][0] < target_position
        ):
            next_start_idx += 1

        # Guarantee forward progress.
        start_idx = max(start_idx + 1, next_start_idx)

    return chunks


    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Invalid chunk size or overlap.")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    # Identify sentence boundaries.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_sentences = []
    current_length = 0
    chunk_start = 0

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # Handle unusually long sentences.
        if len(sentence) > chunk_size:

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text,
                    "char_start": chunk_start,
                    "char_end": chunk_start + len(chunk_text)
                })

                current_sentences = []
                current_length = 0

            for start in range(0, len(sentence), chunk_size - overlap):

                piece = sentence[start:start + chunk_size]

                if piece:
                    chunks.append({
                        "text": piece,
                        "char_start": start,
                        "char_end": start + len(piece)
                    })

                if start + chunk_size >= len(sentence):
                    break

            continue

        additional_length = len(sentence) + (
            1 if current_sentences else 0
        )

        if (
            current_length + additional_length > chunk_size
            and current_sentences
        ):

            chunk_text = " ".join(current_sentences)

            chunks.append({
                "text": chunk_text,
                "char_start": chunk_start,
                "char_end": chunk_start + len(chunk_text)
            })

            # Preserve approximate overlap.
            overlap_sentences = []
            overlap_length = 0

            for previous in reversed(current_sentences):

                if overlap_length + len(previous) > overlap:
                    break

                overlap_sentences.insert(0, previous)
                overlap_length += len(previous) + 1

            current_sentences = overlap_sentences
            current_length = sum(
                len(s) + 1 for s in current_sentences
            )

        if not current_sentences:
            chunk_start = text.find(sentence)

        current_sentences.append(sentence)
        current_length += additional_length

    if current_sentences:

        chunk_text = " ".join(current_sentences)

        chunks.append({
            "text": chunk_text,
            "char_start": chunk_start,
            "char_end": chunk_start + len(chunk_text)
        })

    return chunks


# ----------------------------------
# PROCESS ALL DOCUMENTS
# ----------------------------------

def process_documents():

    pdf_paths = sorted(
        DOCUMENTS_FOLDER.glob("*.pdf")
    )

    if not pdf_paths:
        raise FileNotFoundError(
            "No PDFs found in data/documents."
        )

    all_chunks = []
    chunk_id = 0

    for pdf_path in pdf_paths:

        print(f"\nProcessing {pdf_path.name}")

        pages = extract_pdf(pdf_path)

        pdf_chunk_count = 0

        for page in pages:

            page_chunks = split_into_chunks(page["text"])

            for number, item in enumerate(
                page_chunks, start=1
            ):

                chunk_id += 1
                pdf_chunk_count += 1

                all_chunks.append({
                    "id": chunk_id,
                    "text": item["text"],
                    "source": page["source"],
                    "page": page["page"],
                    "chunk": number,
                    "char_start": item["char_start"],
                    "char_end": item["char_end"]
                })

        print(
            f"Created {pdf_chunk_count} chunks "
            f"from {pdf_path.name}"
        )

    if not all_chunks:
        raise ValueError("No readable chunks were generated.")

    return all_chunks


# ----------------------------------
# GENERATE EMBEDDINGS
# ----------------------------------

def generate_embeddings(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    return embeddings


# ----------------------------------
# SAVE FAISS INDEX
# ----------------------------------

def save_index(chunks, embeddings):

    VECTOR_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    index_temp = VECTOR_FOLDER / "research.index.tmp"
    chunks_temp = VECTOR_FOLDER / "chunks.json.tmp"

    faiss.write_index(index, str(index_temp))

    with chunks_temp.open(
        "w", encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    index_temp.replace(
        VECTOR_FOLDER / "research.index"
    )

    chunks_temp.replace(
        VECTOR_FOLDER / "chunks.json"
    )

    print("\nIndex successfully created!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total vectors: {index.ntotal}")


# ----------------------------------
# MAIN
# ----------------------------------

def main():

    chunks = process_documents()

    embeddings = generate_embeddings(chunks)

    save_index(chunks, embeddings)


if __name__ == "__main__":
    main()
