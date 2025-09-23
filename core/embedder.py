from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

from core.chunker import chunk_docs
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------
# Embedding + Metadata
# ---------------------------


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs: Optional[Dict[str, Any]] = None
) -> SentenceTransformer:
    
    model_kwargs = model_kwargs or {}
    logger.info(f"Loading embedding model: {model_name} | model config: {model_kwargs}")

    # Load model (can be swapped with custom loaders later if needed)
    model = SentenceTransformer(model_name, **model_kwargs)
    
    return model

    
def collect_chunks_and_metadata(
    input_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Tuple[List[str], List[Dict]]:
    """
    Scan documents in input_dir, chunk them, and return both chunks and metadata.
    """
    logger.info(f"Scanning input directory: {input_dir}")
    chunks, metadatas = [], []

    for file_path, file_chunks in chunk_docs(input_dir, chunk_size, chunk_overlap):
        for i, chunk in enumerate(file_chunks):
            chunks.append(chunk)
            metadatas.append({
                "source": str(file_path),
                "chunk_id": i,
                "preview": chunk[:200].replace("\n", " ") + ("..." if len(chunk) > 200 else ""),
                "full": chunk
            })

    if not chunks:
        raise ValueError(f"No supported documents found in {input_dir}")

    logger.info(f"Collected {len(chunks)} chunks from documents")
    return chunks, metadatas


def embed_chunks(
    chunks: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Embed text chunks using a configurable embedding model.
    Supports SentenceTransformer and models with extra parameters.
    """
    model = get_embedding_model(model_name, model_kwargs)

    vectors = []
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding"):
        batch = chunks[i:i + batch_size]

        # Handle model-specific kwargs (e.g. Matryoshka needs `d`)
        vec = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        vectors.append(vec)

    X = np.vstack(vectors).astype("float32")
    logger.info(f"Finished embedding. Shape: {X.shape}")
    return X


def get_embeddings_and_metadata(
    input_dir: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    batch_size: int = 64,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Convenience wrapper: collects chunks+metadata, then embeds them.
    """
    chunks, metadatas = collect_chunks_and_metadata(input_dir, chunk_size, chunk_overlap)
    embeddings = embed_chunks(chunks, model_name=model_name, batch_size=batch_size, model_kwargs=model_kwargs)
    return embeddings, metadatas
