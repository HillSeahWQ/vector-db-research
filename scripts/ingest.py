import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from utils.vectordb import build_collection_name
from core.embedder import get_embeddings_and_metadata
from core.milvus_index import build_milvus_index
import configs.config as cfg

logger = get_logger(__name__)


def main():
    # Build collection name
    collection_name = build_collection_name(
        dataset=cfg.DATASET_NAME,
        index_type=cfg.INDEX_TYPE,
        similarity_metric_type=cfg.SIMILARITY_METRIC_TYPE,
        hyperparameters=cfg.HYPERPARAMETERS
    )
    logger.info(f"Collection name: {collection_name}")

    # Load, chunk, and embed documents
    logger.info(f"Processing documents in: {cfg.DATA_DIR}")
    embeddings, metadatas = get_embeddings_and_metadata(
        input_dir=cfg.DATA_DIR,
        model_name=cfg.MODEL_NAME,
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP,
        batch_size=cfg.BATCH_SIZE
    )
    logger.info(f"Generated embeddings for {len(embeddings)} chunks")

    # Build Milvus index
    logger.info(f"Building Milvus index on collection: {collection_name}")
    build_milvus_index(
        embeddings=embeddings,
        metadatas=metadatas,
        similarity_metric_type=cfg.SIMILARITY_METRIC_TYPE,
        index_type=cfg.INDEX_TYPE,
        hyperparameters=cfg.HYPERPARAMETERS,
        collection_name=collection_name,
        host=cfg.MILVUS_HOST,
        port=cfg.MILVUS_PORT,
    )
    logger.info("Milvus index build complete!")


if __name__ == "__main__":
    main()
