import shutil
import tempfile
from pathlib import Path

import streamlit as st

from utils.chunker import create_code_chunks
from utils.embeddings import EmbeddingStore
from utils.grok_client import GrokCodeAssistant
from utils.loader import extract_zip, read_code_files, scan_code_files
from utils.retriever import CodeRetriever


# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Codebase Intelligence Chatbot",
    layout="wide"
)


# ---------------- SESSION STATE ---------------- #

def initialize_state():
    defaults = {
        "messages": [],
        "project_root": None,
        "index_ready": False,
        "indexed_files": 0,
        "indexed_chunks": 0,
        "last_retrieved": []
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------- CLEAR DATA ---------------- #

def clear_project_data(embedding_store):
    embedding_store.clear_store()

    st.session_state.messages = []
    st.session_state.project_root = None
    st.session_state.index_ready = False
    st.session_state.indexed_files = 0
    st.session_state.indexed_chunks = 0
    st.session_state.last_retrieved = []


# ---------------- INDEX PROJECT ---------------- #

def index_uploaded_project(uploaded_zip, embedding_store):

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = Path(temp_dir) / "project.zip"

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extract_dir = Path("data") / "extracted_project"

        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract ZIP
        project_root = extract_zip(
            str(zip_path),
            str(extract_dir)
        )

        # Scan files
        file_paths = scan_code_files(project_root)

        # Read files
        documents = read_code_files(file_paths)

        # Chunk files
        chunks = create_code_chunks(documents)

        # Build FAISS index
        total_chunks = embedding_store.build_index(chunks)

        # Update state
        st.session_state.project_root = project_root
        st.session_state.index_ready = True
        st.session_state.indexed_files = len(documents)
        st.session_state.indexed_chunks = total_chunks


# ---------------- MAIN APP ---------------- #

def main():

    initialize_state()

    embedding_store = EmbeddingStore(
        vector_store_dir="vector_store"
    )

    st.title("Codebase Intelligence Chatbot")

    st.caption(
        "RAG Pipeline using Streamlit + FAISS + "
        "Sentence Transformers + Groq API"
    )

    # ---------------- SIDEBAR ---------------- #

    with st.sidebar:

        st.header("Project Indexing")

        uploaded_zip = st.file_uploader(
            "Upload Codebase ZIP",
            type=["zip"]
        )

        # INDEX BUTTON
        if uploaded_zip is not None:

            if st.button(
                "Index Project",
                use_container_width=True
            ):

                with st.spinner(
                    "Indexing project..."
                ):

                    try:
                        index_uploaded_project(
                            uploaded_zip,
                            embedding_store
                        )

                        st.success(
                            "Project indexed successfully."
                        )

                    except Exception as e:
                        st.error(
                            f"Indexing failed: {str(e)}"
                        )

        st.divider()

        st.subheader("Index Status")

        st.write(
            f"Ready: {'Yes' if st.session_state.index_ready else 'No'}"
        )

        st.write(
            f"Files Indexed: {st.session_state.indexed_files}"
        )

        st.write(
            f"Chunks Indexed: {st.session_state.indexed_chunks}"
        )

        # CLEAR BUTTON
        if st.button(
            "Clear Vector Database",
            use_container_width=True
        ):

            try:
                clear_project_data(
                    embedding_store
                )

                st.success(
                    "Vector database cleared."
                )

            except Exception as e:
                st.error(
                    f"Failed to clear database: {str(e)}"
                )

    # ---------------- PROJECT SUMMARY ---------------- #

    st.subheader("Project Summary")

    if st.session_state.index_ready:

        st.info(
            f"""
            Indexed Project Root:
            `{st.session_state.project_root}`

            Files Indexed:
            {st.session_state.indexed_files}

            Chunks Indexed:
            {st.session_state.indexed_chunks}
            """
        )

    else:
        st.warning(
            "Upload a ZIP file and click "
            "'Index Project' first."
        )

    # ---------------- CHAT SECTION ---------------- #

    st.subheader("Chat")

    # Display chat history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input(
        "Ask questions about the codebase..."
    )

    if user_question:

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        with st.chat_message("user"):
            st.markdown(user_question)

        # Check if project indexed
        if not st.session_state.index_ready:

            answer = (
                "Please upload and index "
                "a project first."
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.markdown(answer)

            return

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "Retrieving code and generating answer..."
            ):

                try:

                    # Retrieve relevant chunks
                    retriever = CodeRetriever(
                        embedding_store
                    )

                    retrieved = retriever.retrieve(
                        user_question,
                        top_k=6
                    )

                    st.session_state.last_retrieved = retrieved

                    # Groq LLM
                    assistant = GrokCodeAssistant(
                        model="llama3-8b-8192"
                    )

                    # Generate answer
                    answer = assistant.answer_question(
                        user_question,
                        retrieved
                    )

                except Exception as e:

                    answer = (
                        f"Failed to generate answer:\n\n{str(e)}"
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

    # ---------------- RETRIEVED CONTEXT ---------------- #

    if st.session_state.last_retrieved:

        st.subheader("Retrieved Context")

        for item in st.session_state.last_retrieved:

            with st.expander(
                f"{item['file_name']} | "
                f"chunk {item['chunk_index']} | "
                f"score {item['score']:.4f}"
            ):

                st.code(
                    item["text"][:1500],
                    language="python"
                )

                st.caption(
                    f"Path: {item['file_path']}"
                )


# ---------------- RUN APP ---------------- #

if __name__ == "__main__":
    main()
