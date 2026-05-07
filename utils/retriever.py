from typing import Dict, List

import numpy as np

from utils.embeddings import EmbeddingStore


class CodeRetriever:
    """Retrieves most relevant code chunks for a natural language query."""

    def __init__(self, embedding_store: EmbeddingStore):
        self.embedding_store = embedding_store

    def retrieve(self, query: str, top_k: int = 6) -> List[Dict]:
        index, metadata = self.embedding_store.load_index_and_metadata()
        chunks = metadata["chunks"]

        query_vector = self.embedding_store.embed_query(query)
        distances, indices = index.search(np.array(query_vector, dtype="float32"), top_k)

        results: List[Dict] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "file_name": chunk["file_name"],
                    "file_path": chunk["file_path"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }
            )
        return results
