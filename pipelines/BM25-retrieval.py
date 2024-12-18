import code
from utils import *
from indexers import *
from retrieval_pipeline import * 

if __name__ == "__main__":
    # Initialize PassageManager without embeddings
    pm = PassageManager(
        embeddings_file=None,
        collection_file="./datasets/msmarco_hw3/collection.tsv",
        subset_file="./datasets/msmarco_hw3/msmarco_passages_subset.tsv"
    )
    # Initialize QueryManager with query files
    qm = QueryManager(
        query_files=["./datasets/msmarco_hw3/queries.dev.tsv"],
        embedding_files=None
    )
    
    # Initialize BM25Indexer
    bm25_indexer = BM25Indexer()
    result = bm25_indexer.search("2", pm.get_all_ids(), 10)
    print(result)
    
    # Use RetrievalPipeline with BM25Indexer only
    pipe = RetrievalPipeline([BM25Indexer()])
    results, scores = pipe.retrieve("2", pm.get_all_ids(), [10])
    print(results)
    
    code.interact(local=locals())