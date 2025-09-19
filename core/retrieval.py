import numpy as np
from typing import List, Dict, Optional
from pymilvus import Collection
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger
from utils.vectordb import get_milvus_connection


logger = get_logger(__name__)


def search_milvus(
    queries: List[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    similarity_metric_type: str = "COSINE",
    index_search_params: Optional[Dict] = None,
    top_k: int = 5,
    alias: str = "default",
    host: str = "localhost",
    port: str = "19530",
    collection_name: str = "document_embeddings"
) -> List[List[Dict]]:
    """
    Perform similarity search in Milvus and return top-k results with metadata.

    Args:
        queries: List of raw query strings
        embedding_model: Name of the embedding model (e.g., from sentence-transformers)
        similarity_metric_type: "COSINE", "L2", or "IP"
        index_search_params: Index-specific search parameters (e.g., {"ef": 200} for HNSW, {"nprobe": 10} for IVF)
        top_k: Number of top results to return
        alias: Alias of targeted Milvus database to search
        host: Host of targeted Milvus database to search
        port: Port of targeted Milvus database to search
        collection_name: Name of the Milvus collection

    Returns:
        List of results per query, each as a list of dicts containing 'score' and metadata fields
    """
    # Connect to Milvus (if not already connected)
    get_milvus_connection(alias=alias, host=host, port=port)

    # Generate embeddings for queries
    logger.info(f"Encoding {len(queries)} queries using {embedding_model} ...")
    model = SentenceTransformer(embedding_model)
    query_embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

    # Load the collection
    collection = Collection(collection_name)
    collection.load()
    logger.info(f"Loaded collection {collection_name} | Host: {host} | Port: {port}...")

    if index_search_params is None:
        index_search_params = {}

    results = collection.search(
        data=query_embeddings.tolist(),
        anns_field="embedding", 
        param=index_search_params,
        limit=top_k,
        metric_type=similarity_metric_type,
        output_fields=["source", "chunk_id", "preview", "full"]
    )

    all_results = []
    for i, hits in enumerate(results):
        query_results = []
        for hit in hits:
            query_results.append({
                "id": hit.id,
                "score": hit.score,
                "source": hit.entity.get("source"),
                "chunk_id": hit.entity.get("chunk_id"),
                "preview": hit.entity.get("preview"),
                "full": hit.entity.get("full"),
                "query": queries[i]
            })
        all_results.append(query_results)

    return all_results