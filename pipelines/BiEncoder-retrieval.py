import code
from utils import *
from indexers import *
from retrieval_pipeline import * 

if __name__ == "__main__":    
    pm = PassageManager("../embedding/all-MiniLM-L6-v2_encoded_passages.pkl",
                   "../datasets/msmarco_hw3/collection.tsv",
                   "../datasets/msmarco_hw3/msmarco_passages_subset.tsv"
                   )
    qm = QueryManager(["../datasets/msmarco_hw3/queries.dev.tsv"],
                 ["../embedding/all-MiniLM-L6-v2_encoded_passages.pkl"])
    
    cosine_similarity_indexer = CosineSimilarityIndexer()
    result = cosine_similarity_indexer.search("2", pm.get_all_ids(), 10)
    print(result)
    
    pipe = RetrievalPipeline([CosineSimilarityIndexer(), 
                              CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")])
    results, scores = pipe.retrieve("2", pm.get_all_ids(), [1000, 10])
    print(results)
    print(scores)
    
    code.interact(local=locals())
    
    