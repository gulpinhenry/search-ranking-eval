from utils import *
from indexers import *
from retrieval_pipeline import * 

if __name__ == "__main__":
    pm = PassageManager("../embedding/all-MiniLM-L6-v2_encoded_passages.pkl",
                   "../../dense-vector-retrieval-system/MSMARCO-embeddings/collection.tsv",
                   "../../dense-vector-retrieval-system/MSMARCO-embeddings/msmarco_passages_subset.tsv"
                   )
    QueryManager(["../../dense-vector-retrieval-system/MSMARCO-embeddings/queries.dev.tsv"],
                 ["../embedding/all-MiniLM-L6-v2_encoded_dev_queries.pkl"])
    
    cosine_similarity_indexer = CosineSimilarityIndexer()
    result = cosine_similarity_indexer.search("2", pm.get_all_ids(), 10)
    print(result)
    
    