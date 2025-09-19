from pathlib import Path
from pypdf import PdfReader
from utils.logger import get_logger

logger = get_logger(__name__)

def load_text_from_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        text_parts = []
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                # pypdf can return None for empty pages—guard it
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        except Exception as e:
            logger.warning(f"Failed to read PDF {path}: {e}")
        # document_parts = ["Page 1 text...", "Page 2 text...", "Page 3 text..."] AND "/n".join(document_parts) = "Page 1 text...\nPage 2 text...\nPage 3 text..."
        return "\n".join(text_parts)
    else:
        return ""  # unsupported