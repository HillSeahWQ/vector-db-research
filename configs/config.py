from pathlib import Path


# Directories
# ---------------------------------------------------------------------------------------------------------------------------------------
DATA_DIR = Path().cwd() / "data" / "kyndryl-docs-test"
LOG_DIR = Path().cwd() / "logs"
# ---------------------------------------------------------------------------------------------------------------------------------------


# Chunking
# ---------------------------------------------------------------------------------------------------------------------------------------
CHUNK_SIZE = 200      # characters per chunk (~150-250 tokens)
CHUNK_OVERLAP = 50    # characters of overlap
BATCH_SIZE = 64       # batch building of index
# ---------------------------------------------------------------------------------------------------------------------------------------


# Embedding model
# ---------------------------------------------------------------------------------------------------------------------------------------
BASE_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
BASE_EMBEDDING_MODEL_KWARGS = None
MATRYOSHKA_EMBEDDING_MODEL = "tomaarsen/mpnet-base-nli-matryoshka"
MATRYOSHKA_EMBEDDING_MODEL_KWARGS = { # None if using default params
    "truncate_dim": 64 # for Matroyoshka model
}


# Milvus config
# ---------------------------------------------------------------------------------------------------------------------------------------
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
SIMILARITY_METRIC_TYPE = "IP"
# ---------------------------------------------------------------------------------------------------------------------------------------


# **TO EDIT TO TEST**
# ---------------------------------------------------------------------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = MATRYOSHKA_EMBEDDING_MODEL
EMBEDDING_MODEL_KWARGS = MATRYOSHKA_EMBEDDING_MODEL_KWARGS
INDEX_TYPE = "IVF_PQ" # "FLAT" - brute force, exact search | "IVF_FLAT" - inverted file with flat quantization | "IVF_SQ8" - IVF with scalar quantization (8-bit) | "IVF_PQ" - IVF with product quantization | "HNSW" - Hierarchical Navigable Small World graph | "ANNOY" - Approximate Nearest Neighbors Oh Yeah | "DISKANN" - disk-based ANN, memory-efficient (Milvus 2.x+).
INDEXING_HYPERPARAMETERS = {
    "nlist": 1,  # nlist=1 simulates flat exhaustive search
    "m": 16,     # number of subvectors
    "nbits": 8   # bits per subvector
}
INDEX_SEARCH_PARAMS = None
DATASET_NAME = "kyndryl_pdfs"
TOP_K=5
QUERIES = [
    "How much does Kyndryl cover for surgeries"
]
# ---------------------------------------------------------------------------------------------------------------------------------------