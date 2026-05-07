# Codebase Intelligence Chatbot (Streamlit + Grok + FAISS)

An AI-powered **Codebase Intelligence Chatbot** built fully with:

- Python
- Streamlit
- Grok API (OpenAI-compatible client)
- FAISS vector database
- SentenceTransformers embeddings

This app uses a **RAG (Retrieval-Augmented Generation)** pipeline to answer questions about an uploaded codebase ZIP.

## Features

- Upload a project as ZIP and index it automatically
- Recursive scanning of code files
- Supported file types:
  - `.py`, `.js`, `.java`, `.cpp`, `.html`, `.css`, `.json`, `.md`
- Intelligent chunking of large files
- Embedding generation with `sentence-transformers`
- Vector search with FAISS for semantic retrieval
- Grok-powered contextual answers
- Streamlit chat UI with:
  - project summary
  - indexing status
  - chat history
  - retrieved file names and snippets
  - clear database button

## Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── utils/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── grok_client.py
│
├── data/
├── vector_store/
└── README.md
```

## How RAG Works Here

1. User uploads ZIP
2. Files are extracted and scanned
3. Code is chunked into smaller parts
4. Chunks are embedded and stored in FAISS
5. User asks a question
6. Top relevant chunks are retrieved
7. Retrieved context + question are sent to Grok API
8. Final answer is shown in chat

## Setup Instructions

### 1) Clone or copy project

Place all files into your local folder.

### 2) Create virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Configure environment variables

Create a `.env` file in project root:

```bash
XAI_API_KEY=your_xai_api_key_here
GROK_MODEL=grok-3-mini
```

You can also set these in your shell if preferred.

### 5) Run the Streamlit app

```bash
python -m streamlit run app.py
```

## How to Use

1. Open the app in browser (Streamlit gives local URL)
2. In sidebar, upload a project ZIP
3. Click **Index Project**
4. Wait for indexing success message
5. Ask questions in chat like:
   - What does this project do?
   - Explain authentication flow
   - Where is DB connection implemented?
   - Which file handles APIs?
   - Summarize project architecture

## Notes

- This app skips likely binary files and very large files (safety/performance).
- For best results, include complete source files in the ZIP.
- If answer quality drops, re-index with a cleaner codebase ZIP.
