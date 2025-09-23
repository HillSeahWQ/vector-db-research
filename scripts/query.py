import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger
from utils.vectordb import build_collection_name
from core.retrieval import search_milvus
import configs.config as cfg

logger = get_logger(__name__)


def main():
    # Build collection name (must match ingestion)
    collection_name = build_collection_name(
        dataset=cfg.DATASET_NAME,
        index_type=cfg.INDEX_TYPE,
        similarity_metric_type=cfg.SIMILARITY_METRIC_TYPE,
        hyperparameters=cfg.INDEXING_HYPERPARAMETERS,
    )
    logger.info(f"Using collection: {collection_name}")

    # Example queries
    queries=cfg.QUERIES
    logger.info(f"Running {len(queries)} queries against collection `{collection_name}`")

    # Search
    all_results = search_milvus(
        queries=queries,
        embedding_model=cfg.EMBEDDING_MODEL_NAME,
        embedding_model_kwargs=cfg.EMBEDDING_MODEL_KWARGS,
        similarity_metric_type=cfg.SIMILARITY_METRIC_TYPE,
        index_search_params=cfg.INDEX_SEARCH_PARAMS,
        top_k=cfg.TOP_K,
        alias="default",
        host=cfg.MILVUS_HOST,
        port=cfg.MILVUS_PORT,
        collection_name=collection_name,
    )

    # Display results
    for query, results in zip(queries, all_results):
        logger.info(f"\nQuery: {query}")
        for rank, hit in enumerate(results, start=1):
            logger.info(f"  {rank}. {hit}")


if __name__ == "__main__":
    main()