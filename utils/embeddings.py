import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingStore:
    """Handles embedding generation and FAISS index persistence."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_dir: str = "vector_store",
    ):
        self.model_name = model_name
        self.vector_store_dir = Path(vector_store_dir)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.vector_store_dir / "faiss.index"
        self.meta_path = self.vector_store_dir / "metadata.json"

        self.model = SentenceTransformer(model_name)

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        )
        return np.array(embeddings, dtype="float32")

    def build_index(self, chunks: List[Dict[str, str]]) -> int:
        """Generate embeddings, build FAISS index, and save metadata."""
        if not chunks:
            raise ValueError("No chunks were provided for indexing.")

        texts = [c["text"] for c in chunks]
        embeddings = self._embed_texts(texts)
        embedding_dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(embedding_dim)  # Cosine similarity on normalized vectors.
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))

        metadata = {
            "model_name": self.model_name,
            "total_chunks": len(chunks),
            "chunks": chunks,
        }
        self.meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        return len(chunks)

    def load_index_and_metadata(self):
        """Load index and metadata from disk."""
        if not self.index_path.exists() or not self.meta_path.exists():
            raise FileNotFoundError("Vector index not found. Please index a project first.")

        index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return index, metadata

    def embed_query(self, query: str) -> np.ndarray:
        query_vector = self._embed_texts([query])
        return query_vector

    def clear_store(self) -> None:
        """Delete saved index and metadata."""
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
