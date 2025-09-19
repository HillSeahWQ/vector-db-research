from pymilvus import connections
from utils.logger import get_logger


logger = get_logger(__name__)


def build_collection_name(dataset, index_type, similarity_metric_type, hyperparameters):
    
    collection_name  = dataset + "_" + index_type + "_" + similarity_metric_type
    for k, v in hyperparameters.items():
        collection_name += f"_{k}_{v}"
        
    return collection_name


def get_milvus_connection(
    alias: str = "default",
    host: str = "localhost",
    port: str = "19530",
    reset: bool = False,
) -> str:
    """
    Get or create a Milvus connection.

    Parameters
    ----------
    alias : str
        Connection alias (default: "default").
    host : str
        Milvus host.
    port : str
        Milvus port.
    reset : bool
        If True, disconnect existing connection and reconnect.

    Returns
    -------
    str
        The alias name of the connection.
    """
    if reset:
        connections.disconnect(alias)
        logger.info("Disconnected from Milvus connection (RESET)")

    if not any(c[0] == alias and c[1] for c in connections.list_connections()):
        connections.connect(alias=alias, host=host, port=port)
        logger.info(f"Successfully connected to Milvus at {host}:{port}")

    return alias