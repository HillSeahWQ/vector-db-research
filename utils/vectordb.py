def build_collection_name(dataset, index_type, similarity_metric_type, hyperparameters):
    
    collection_name  = dataset + "_" + index_type + "_" + similarity_metric_type
    for k, v in hyperparameters.items():
        collection_name += f"_{k}_{v}"
        
    return collection_name