from pathlib import Path
from typing import List, Tuple

from utils.logger import get_logger
from core.data_loader import load_text_from_file

logger = get_logger(__name__)

def chunk_text(
    text: str,
    chunk_size: int,       # characters per chunk (roughly ~150-250 tokens)
    chunk_overlap: int,    # characters of overlap
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        # ensure we don’t cut off mid-word too badly by extending to next whitespace if possible
        if end < n:
            next_space = text.find(" ", end)
            if 0 < next_space - end < 20:  # small nudge to nearest space
                chunk = text[start:next_space]
                end = next_space
        chunks.append(chunk.strip())
        start = max(end - chunk_overlap, end)  # avoid infinite loop; no negative steps
    # dedupe empties
    return [c for c in chunks if c]


def chunk_docs(
    input_dir: Path,
    chunk_size,       # characters per chunk (roughly ~150-250 tokens)
    chunk_overlap,
) -> List[Tuple[Path, List[str]]]:
    docs = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}:
            text = load_text_from_file(path)
            if not text:
                continue
            docs.append((path, chunk_text(text, chunk_size, chunk_overlap)))
    return docs