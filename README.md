# RAG Research Assistant

A local app that searches research papers and answers questions using passages retrieved from their PDFs.

## Features

- Upload one or more PDF papers through the app.
- Extract text while keeping each paper’s filename and page number.
- Split text into overlapping chunks.
- Create embeddings and store them in a FAISS index.
- Retrieve and rerank passages related to a question.
- Generate answers from retrieved passages and show the source pages.
- Decline to answer when the strongest retrieved passage has a low relevance score.
- Keeps question-and-answer history during the session and lets you clear the conversation.

## Requirements

- Python 3.11 or newer
- Windows PowerShell for the setup commands below

## Setup

Open a terminal in the project folder and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

The app opens in your browser. Choose PDF files and click **Save PDFs and rebuild index** to add them to the searchable collection.

You can also copy PDFs into `data/documents/` and rebuild the index from the terminal:

```powershell
python build_index.py
```

If you rebuild the index from the terminal while the app is running, restart Streamlit so it reloads the index.

## How it works

1. `pypdf` extracts text from each PDF and keeps its page number.
2. The text is split into chunks of about 1,000 characters, with 200 characters of overlap.
3. Sentence Transformers converts each chunk into an embedding.
4. FAISS retrieves up to 10 candidate passages for a question.
5. A CrossEncoder reranks the candidates. The app displays the top three passages and gives the top two to the answer model.
6. Qwen2.5-0.5B-Instruct generates an answer using the retrieved passages.

The embedding model, reranker, and answer model download from Hugging Face the first time they are used and run locally.

## Project files

- `app.py` — Streamlit interface, PDF upload, search, reranking, and answer generation.
- `build_index.py` — reads PDFs, creates chunks and embeddings, and saves the FAISS index.
- `data/documents/` — PDF papers.
- `vector_store/` — generated FAISS index and chunk metadata.
- `requirements.txt` — Python packages needed by the project.

## Limitations

- Scanned PDFs without selectable text need OCR before indexing.
- PDF text extraction can contain line breaks or encoding issues.
- The answer model can make mistakes; check its answer against the displayed source passages.
- The relevance-score cutoff is a basic safeguard, not a guarantee that every answer is supported.