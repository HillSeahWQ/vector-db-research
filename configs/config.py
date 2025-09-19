from pathlib import Path

# Directories
DATA_DIR = Path().cwd() / 'data' / 'kyndryl-docs-test'
LOG_DIR = Path().cwd() / 'logs'

# Embedding model
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

# Chunking
CHUNK_SIZE = 200      # characters per chunk (~150-250 tokens)
CHUNK_OVERLAP = 50    # characters of overlap
BATCH_SIZE = 64       # batch building of index

# Milvus config
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
SIMILARITY_METRIC_TYPE = "IP"
INDEX_TYPE = "IVF_PQ"
HYPERPARAMETERS = {
    "nlist": 1,  # nlist=1 simulates flat exhaustive search
    "m": 16,     # number of subvectors
    "nbits": 8   # bits per subvector
}
DATASET_NAME = "kyndryl_pdfs"
INDEX_SEARCH_PARAMS = None
TOP_K=5
