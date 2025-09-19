import numpy as np
from typing import List, Dict, Optional
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

from utils.logger import get_logger
from utils.vectordb import get_milvus_connection

logger = get_logger(__name__)


def close_milvus_connection(alias: str = "default") -> None:
    """Disconnect from Milvus explicitly."""
    if any(c[0] == alias and c[1] for c in connections.list_connections()):
        connections.disconnect(alias)
        logger.info("Disconnected from Milvus connection")


def build_milvus_index(
    embeddings: np.ndarray,
    metadatas: List[Dict],
    similarity_metric_type: str,
    index_type: str,
    hyperparameters: Optional[Dict] = None,
    collection_name: str = "document_embeddings",
    alias: str = "default",
    host: str = "localhost",
    port: str = "19530",
) -> None:
    """
    Build a Milvus collection, insert embeddings with metadata, and create an index.

    Parameters
    ----------
    embeddings : np.ndarray
        Numpy array of shape (num_vectors, dim).
    metadatas : List[Dict]
        List of dictionaries containing metadata for each vector.
    similarity_metric_type : str
        Similarity metric (e.g., "IP", "COSINE", "L2").
    index_type : str
        Index type (e.g., "IVF_PQ", "HNSW").
    hyperparameters : Optional[Dict], default None
        Index-specific hyperparameters. If None, will default to empty dict.
    collection_name : str
        Name of the Milvus collection.
    alias : str
        Milvus connection alias.
    host : str
        Milvus host.
    port : str
        Milvus port.
    """
    # Ensure hyperparameters is a dict
    if hyperparameters is None:
        hyperparameters = {}

    # 1) Connect to Milvus
    logger.info(f"Connecting to Milvus at {host}:{port} ...")
    get_milvus_connection(alias=alias, host=host, port=port)

    # 2) Drop collection if exists
    if utility.has_collection(collection_name):
        logger.info(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    # 3) Create collection schema
    dim = embeddings.shape[1]
    logger.info(f"Creating new collection: {collection_name} with dimension {dim}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="preview", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="full", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="Document embeddings")
    collection = Collection(name=collection_name, schema=schema)

    # 4) Prepare and insert data
    logger.info(f"Inserting {len(metadatas)} records into Milvus...")
    insert_data = [
        {
            "embedding": embeddings[i].tolist(),
            "source": metadatas[i].get("source", ""),
            "chunk_id": metadatas[i].get("chunk_id", i),
            "preview": metadatas[i].get("preview", ""),
            "full": metadatas[i].get("full", ""),
        }
        for i in range(len(metadatas))
    ]
    collection.insert(insert_data)
    logger.info("Insertion complete.")

    # 5) Create index
    logger.info("Creating index on `embedding` field...")
    index_params = {
        "metric_type": similarity_metric_type,
        "index_type": index_type,
        "params": hyperparameters,
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # 6) Load collection
    collection.load()
    logger.info(
        f"[OK] Inserted {embeddings.shape[0]} vectors into Milvus collection `{collection_name}` "
        f"(dim={dim}, index type={index_type}, similarity metric={similarity_metric_type})."
    )