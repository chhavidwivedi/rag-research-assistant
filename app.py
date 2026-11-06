import json
import subprocess
import sys
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline


st.set_page_config(page_title="Research Assistant")
st.title("RAG Research Assistant")
st.write("Ask a question about the research paper.")

st.session_state.setdefault("messages", [])
if st.button("Clear conversation"):
    st.session_state.messages.clear()
    st.rerun()

uploaded_files = st.file_uploader(
    "Choose research PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files and st.button("Save PDFs and rebuild index"):
    documents_folder = Path("data/documents")
    documents_folder.mkdir(parents=True, exist_ok=True)

    saved_names = []

    with st.spinner("Saving PDFs and rebuilding the search index..."):
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            destination = documents_folder / safe_name
            destination.write_bytes(uploaded_file.getvalue())
            saved_names.append(safe_name)

        build_result = subprocess.run(
            [sys.executable, "build_index.py"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
        )

    if build_result.returncode != 0:
        st.error("The index could not be rebuilt.")
        st.code(build_result.stderr or build_result.stdout)
    else:
        st.cache_resource.clear()
        st.success(
            f"Saved and indexed {len(saved_names)} PDF(s): "
            f"{', '.join(saved_names)}"
        )


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    index = faiss.read_index("vector_store/research.index")

    with Path("vector_store/chunks.json").open(
        "r", encoding="utf-8"
    ) as file:
        chunks = json.load(file)

    answer_model = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
    )

    return embedder, reranker, index, chunks, answer_model

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
question = st.chat_input("Ask a question about your papers")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the papers and preparing an answer..."):
            embedder, reranker, index, chunks, answer_model = load_resources()

            question_vector = embedder.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            candidate_count = min(10, index.ntotal)
            _, candidate_ids = index.search(
                question_vector,
                candidate_count,
            )

            candidates = [
                chunks[int(chunk_id)]
                for chunk_id in candidate_ids[0]
                if chunk_id >= 0
            ]

            rerank_scores = reranker.predict(
                [(question, chunk["text"]) for chunk in candidates]
            )

            ranked_results = sorted(
                zip(candidates, rerank_scores),
                key=lambda item: float(item[1]),
                reverse=True,
            )[:3]

            results = [
                (chunk, float(score))
                for chunk, score in ranked_results
            ]

            if results and results[0][1] >= 0:
                context = "\n\n".join(
                    f"[Source: {chunk['source']}, page {chunk['page']}]\n"
                    f"{chunk['text'][:1000]}"
                    for chunk, _score in results[:2]
                )

                llm_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Use only the supplied paper excerpts. For list "
                            "questions, give only the exact names explicitly "
                            "listed in the excerpts; do not add descriptions. "
                            "If the excerpts do not contain the answer, say so."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Paper excerpts:\n{context}"
                        ),
                    },
                ]

                generated = answer_model(
                    llm_messages,
                    max_new_tokens=160,
                    do_sample=False,
                )

                answer = generated[0]["generated_text"][-1]["content"]

                source_chunk = results[0][0]
                answer += (
                    f"\n\nSource: {source_chunk['source']}, "
                    f"page {source_chunk['page']}"
                )
            else:
                answer = (
                    "I couldn't find relevant evidence for this question "
                    "in the papers."
                )

        st.write(answer)

        st.markdown("**Retrieved passages**")
        for chunk, score in results:
            with st.expander(
                f"{chunk['source']} — page {chunk['page']} "
                f"(rerank score: {score:.2f})"
            ):
                st.write(chunk["text"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })