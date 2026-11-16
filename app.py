
import streamlit as st

from search import load_search_resources, search_papers
from answer_extractive import (
    generate_extractive_answer,
    INSUFFICIENT_EVIDENCE
)


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Research Assistant")

st.markdown(
    """
    Ask questions about research papers and receive
    evidence-based answers with paper and page citations.

    This application uses semantic retrieval, CrossEncoder
    reranking, and extractive answer generation.
    """
)

st.divider()


# ----------------------------------
# LOAD RESOURCES
# ----------------------------------

@st.cache_resource
def get_resources():
    return load_search_resources()


# ----------------------------------
# SESSION STATE
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.header("About the project")

    st.write(
        """
        This research assistant uses:

        - Sentence Transformers for embeddings
        - FAISS for vector similarity search
        - CrossEncoder for reranking
        - Extractive answer generation
        """
    )

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------
# DISPLAY PREVIOUS MESSAGES
# ----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ----------------------------------
# USER INPUT
# ----------------------------------

question = st.chat_input(
    "Ask a question about your research papers..."
)


# ----------------------------------
# PROCESS QUESTION
# ----------------------------------

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching research papers and preparing your answer..."
            ):

                # Load models and FAISS index.
                embedder, reranker, index, chunks = get_resources()

                # Retrieve and rerank relevant passages.
                results = search_papers(
                    question,
                    embedder,
                    reranker,
                    index,
                    chunks
                )

                # Generate extractive answer.
                answer = generate_extractive_answer(
                    question,
                    results
                )

            # ----------------------------------
            # DISPLAY ANSWER
            # ----------------------------------

            st.markdown("### Answer")

            st.markdown(answer)

            # ----------------------------------
            # DISPLAY RETRIEVED EVIDENCE
            # ----------------------------------

            if answer != INSUFFICIENT_EVIDENCE and results:

                st.markdown("### Supporting evidence")

                for i, result in enumerate(results, start=1):

                    chunk = result["chunk"]

                    score = result.get(
                        "rerank_score",
                        "Not available"
                    )

                    with st.expander(
                        f"{i}. {chunk['source']} — "
                        f"Page {chunk['page']} "
                        f"(Reranker score: {score})"
                    ):

                        st.write(chunk["text"])

            else:

                st.info(
                    "No sufficiently relevant evidence was found "
                    "in the uploaded research papers."
                )

            # ----------------------------------
            # SAVE ASSISTANT RESPONSE
            # ----------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as error:

            st.error(
                "Something went wrong while processing your question."
            )

            with st.expander("Technical error details"):
                st.exception(error)
