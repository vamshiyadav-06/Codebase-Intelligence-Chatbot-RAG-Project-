import os
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple


SUPPORTED_EXTENSIONS = {".py", ".js", ".java", ".cpp", ".html", ".css", ".json", ".md"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".idea", ".vscode"}
MAX_FILE_SIZE_BYTES = 1_000_000  # Skip files larger than 1 MB for safety/performance.


def extract_zip(zip_path: str, output_dir: str) -> str:
    """Extract a ZIP file into output_dir and return extracted root folder."""
    extract_root = Path(output_dir)
    extract_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # If zip contains a single top-level folder, use it as project root.
    top_items = [p for p in extract_root.iterdir()]
    if len(top_items) == 1 and top_items[0].is_dir():
        return str(top_items[0])
    return str(extract_root)


def is_text_file_safe(file_path: Path) -> bool:
    """Best-effort text file check to avoid binary files."""
    try:
        with open(file_path, "rb") as f:
            sample = f.read(4096)
        if b"\x00" in sample:
            return False
        return True
    except Exception:
        return False


def scan_code_files(project_root: str) -> List[str]:
    """Recursively collect supported source files from project_root."""
    code_files: List[str] = []
    root = Path(project_root)

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(skip_part in path.parts for skip_part in SKIP_DIRS):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue

        if is_text_file_safe(path):
            code_files.append(str(path))

    return code_files


def read_code_files(file_paths: List[str]) -> List[Dict[str, str]]:
    """Read file contents safely and return list of dictionaries."""
    documents: List[Dict[str, str]] = []

    for file_path in file_paths:
        path = Path(file_path)
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            if not content.strip():
                continue
            documents.append(
                {
                    "file_path": str(path),
                    "file_name": path.name,
                    "extension": path.suffix.lower(),
                    "content": content,
                }
            )
        except Exception:
            # Continue if one file fails so indexing can proceed.
            continue

    return documents
