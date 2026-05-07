import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from utils.chunker import create_code_chunks
from utils.embeddings import EmbeddingStore
from utils.grok_client import GrokCodeAssistant
from utils.loader import extract_zip, read_code_files, scan_code_files
from utils.retriever import CodeRetriever


st.set_page_config(page_title="Codebase Intelligence Chatbot", layout="wide")


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "project_root" not in st.session_state:
        st.session_state.project_root = None
    if "index_ready" not in st.session_state:
        st.session_state.index_ready = False
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = 0
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = 0
    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved = []


def clear_project_data(embedding_store: EmbeddingStore):
    embedding_store.clear_store()
    st.session_state.messages = []
    st.session_state.project_root = None
    st.session_state.index_ready = False
    st.session_state.indexed_files = 0
    st.session_state.indexed_chunks = 0
    st.session_state.last_retrieved = []


def index_uploaded_project(uploaded_zip, embedding_store: EmbeddingStore):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "uploaded_project.zip"
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extract_dir = Path("data") / "extracted_project"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        project_root = extract_zip(str(zip_path), str(extract_dir))
        file_paths = scan_code_files(project_root)
        documents = read_code_files(file_paths)
        chunks = create_code_chunks(documents)
        total_chunks = embedding_store.build_index(chunks)

        st.session_state.project_root = project_root
        st.session_state.index_ready = True
        st.session_state.indexed_files = len(documents)
        st.session_state.indexed_chunks = total_chunks


def main():
    initialize_state()
    embedding_store = EmbeddingStore(vector_store_dir="vector_store")

    st.title("Codebase Intelligence Chatbot")
    st.caption("RAG pipeline with Streamlit + FAISS + SentenceTransformers + Grok API")

    with st.sidebar:
        st.header("Project Indexing")
        uploaded_zip = st.file_uploader("Upload codebase ZIP", type=["zip"])

        if uploaded_zip is not None:
            if st.button("Index Project", use_container_width=True):
                with st.spinner("Extracting, chunking, embedding, and indexing project..."):
                    try:
                        index_uploaded_project(uploaded_zip, embedding_store)
                        st.success("Project indexed successfully.")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

        st.markdown("---")
        st.subheader("Index Status")
        st.write(f"Ready: {'Yes' if st.session_state.index_ready else 'No'}")
        st.write(f"Files Indexed: {st.session_state.indexed_files}")
        st.write(f"Chunks Indexed: {st.session_state.indexed_chunks}")

        if st.button("Clear Vector Database", use_container_width=True):
            try:
                clear_project_data(embedding_store)
                st.success("Vector database and session history cleared.")
            except Exception as e:
                st.error(f"Failed to clear data: {e}")

    st.subheader("Project Summary")
    if st.session_state.index_ready:
        st.info(
            f"Indexed project root: `{st.session_state.project_root}`  \n"
            f"Total files: **{st.session_state.indexed_files}** | "
            f"Total chunks: **{st.session_state.indexed_chunks}**"
        )
    else:
        st.warning("Upload a ZIP project and click 'Index Project' to start.")

    st.subheader("Chat")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask about architecture, modules, flow, files, or specific functions...")
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        if not st.session_state.index_ready:
            answer = "Please upload and index a project first."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
            return

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant code and generating answer..."):
                try:
                    retriever = CodeRetriever(embedding_store)
                    retrieved = retriever.retrieve(user_question, top_k=6)
                    st.session_state.last_retrieved = retrieved

                    assistant = GrokCodeAssistant(model=os.getenv("GROK_MODEL", "grok-3-mini"))
                    answer = assistant.answer_question(user_question, retrieved)
                except Exception as e:
                    answer = f"Failed to generate answer: {e}"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.last_retrieved:
        st.subheader("Retrieved Context")
        for item in st.session_state.last_retrieved:
            with st.expander(
                f"{item['file_name']} | chunk {item['chunk_index']} | score {item['score']:.4f}"
            ):
                st.code(item["text"][:1500], language="text")
                st.caption(f"Path: {item['file_path']}")


if __name__ == "__main__":
    main()
