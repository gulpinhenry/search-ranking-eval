import code
from utils import *
from indexers import *
from retrieval_pipeline import * 

if __name__ == "__main__":
    pm = PassageManager("../embedding/all-MiniLM-L6-v2_encoded_passages.pkl",
                   "../../dense-vector-retrieval-system/MSMARCO-embeddings/collection.tsv",
                   "../../dense-vector-retrieval-system/MSMARCO-embeddings/msmarco_passages_subset.tsv"
                   )
    qm = QueryManager(["../../dense-vector-retrieval-system/MSMARCO-embeddings/queries.dev.tsv"],
                 ["../embedding/all-MiniLM-L6-v2_encoded_dev_queries.pkl"])
    
    cross_encoder_indexer = CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")
    result = cross_encoder_indexer.search("2", pm.get_all_ids()[-1000:], 10)
    print(result)
    
    pipe = RetrievalPipeline([CosineSimilarityIndexer(), 
                              CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")])
    results, scores = pipe.retrieve("2", pm.get_all_ids(), [1000, 10])
    print(results)
    print(scores)
    
    code.interact(local=locals())
    
    