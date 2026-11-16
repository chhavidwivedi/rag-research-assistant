
import re
import json
from pathlib import Path

import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer


# ----------------------------------
# CONFIGURATION
# ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
VECTOR_DIR = BASE_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

TOP_K_CANDIDATES = 40
TOP_K_RESULTS = 5
MAX_RESULTS_PER_PAGE = 2


# ----------------------------------
# LOAD RESOURCES
# ----------------------------------

def load_search_resources():

    index_path = VECTOR_DIR / "research.index"
    chunks_path = VECTOR_DIR / "chunks.json"

    if not index_path.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run build_index.py."
        )

    if not chunks_path.exists():
        raise FileNotFoundError(
            "chunks.json not found. Run build_index.py."
        )

    embedder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)

    index = faiss.read_index(str(index_path))

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal == 0 or not chunks:
        raise ValueError("Research index is empty.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk count do not match.")

    return embedder, reranker, index, chunks


# ----------------------------------
# BIBLIOGRAPHY FILTER
# ----------------------------------

def is_bibliography_chunk(text):

    lower = text.lower()

    citation_starts = len(
        re.findall(r"\[\d+\]", text)
    )

    reference_terms = [
        "arxiv preprint",
        "doi:",
        "issn",
        "proceedings of",
    ]

    term_count = sum(
        term in lower for term in reference_terms
    )

    return citation_starts >= 4 and term_count >= 1


# ----------------------------------
# TEXT NORMALIZATION
# ----------------------------------

def normalize_text(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


# ----------------------------------
# SEMANTIC RETRIEVAL
# ----------------------------------

def retrieve_candidates(
    question,
    embedder,
    index,
    chunks,
):

    question_vector = embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    candidate_count = min(
        TOP_K_CANDIDATES,
        index.ntotal,
    )

    scores, result_ids = index.search(
        question_vector,
        candidate_count,
    )

    candidates = []

    for score, chunk_id in zip(
        scores[0],
        result_ids[0],
    ):

        if chunk_id < 0 or chunk_id >= len(chunks):
            continue

        chunk = chunks[chunk_id]

        if is_bibliography_chunk(chunk["text"]):
            continue

        candidates.append({
            "chunk": chunk,
            "similarity_score": float(score),
        })

    return candidates


# ----------------------------------
# CROSS-ENCODER RERANKING
# ----------------------------------

def rerank_results(
    question,
    candidates,
    reranker,
):

    if not candidates:
        return []

    pairs = [
        (
            question,
            item["chunk"]["text"],
        )
        for item in candidates
    ]

    cross_scores = reranker.predict(pairs)

    for item, score in zip(candidates, cross_scores):
        item["rerank_score"] = float(score)

    ranked_results = sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return ranked_results


# ----------------------------------
# DEDUPLICATION AND SOURCE DIVERSITY
# ----------------------------------

def select_diverse_results(ranked_results):

    selected = []
    seen_texts = set()
    page_counts = {}

    for result in ranked_results:

        chunk = result["chunk"]

        normalized = normalize_text(chunk["text"])

        if normalized in seen_texts:
            continue

        source_page = (
            chunk["source"],
            chunk["page"]
        )

        if page_counts.get(source_page, 0) >= MAX_RESULTS_PER_PAGE:
            continue

        seen_texts.add(normalized)

        page_counts[source_page] = (
            page_counts.get(source_page, 0) + 1
        )

        selected.append(result)

        if len(selected) >= TOP_K_RESULTS:
            break

    return selected


# ----------------------------------
# COMPLETE SEARCH FUNCTION
# ----------------------------------

def search_papers(
    question,
    embedder,
    reranker,
    index,
    chunks,
):

    if not question or not question.strip():
        return []

    candidates = retrieve_candidates(
        question,
        embedder,
        index,
        chunks,
    )

    ranked_results = rerank_results(
        question,
        candidates,
        reranker,
    )

    results = select_diverse_results(ranked_results)

    return results


# ----------------------------------
# COMMAND LINE TEST
# ----------------------------------

if __name__ == "__main__":

    print("Loading research search system...")

    embedder, reranker, index, chunks = load_search_resources()

    print(f"Loaded {len(chunks)} chunks.")

    while True:

        question = input(
            "\nAsk a question (or type exit): "
        ).strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            print("Please enter a question.")
            continue

        results = search_papers(
            question,
            embedder,
            reranker,
            index,
            chunks,
        )

        if not results:
            print("No results found.")
            continue

        for rank, result in enumerate(results, start=1):

            chunk = result["chunk"]

            print(
                f"\nResult {rank}: "
                f"{chunk['source']} | Page {chunk['page']}"
            )

            print(
                f"Similarity score: "
                f"{result['similarity_score']:.3f}"
            )

            print(
                f"CrossEncoder score: "
                f"{result['rerank_score']:.3f}"
            )

            print("Passage:", chunk["text"])
