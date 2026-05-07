from typing import Dict, List


def chunk_text(content: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    """Split long text into overlapping chunks."""
    if len(content) <= chunk_size:
        return [content]

    chunks: List[str] = []
    start = 0
    text_len = len(content)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = content[start:end]
        chunks.append(chunk)
        if end == text_len:
            break
        start = max(end - overlap, 0)

    return chunks


def create_code_chunks(
    documents: List[Dict[str, str]], chunk_size: int = 1200, overlap: int = 150
) -> List[Dict[str, str]]:
    """Create metadata-rich chunks from loaded code documents."""
    all_chunks: List[Dict[str, str]] = []

    for doc in documents:
        file_chunks = chunk_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for index, chunk in enumerate(file_chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc['file_path']}::chunk_{index}",
                    "file_path": doc["file_path"],
                    "file_name": doc["file_name"],
                    "extension": doc["extension"],
                    "chunk_index": index,
                    "text": chunk,
                }
            )

    return all_chunks
