import code
from utils import *
from indexers import *
from retrieval_pipeline import * 
import pytrec_eval

if __name__ == "__main__":
    pm = PassageManager(
        embeddings_file="../embedding/all-MiniLM-L6-v2_encoded_passages.pkl",
        collection_file="../../dense-vector-retrieval-system/MSMARCO-embeddings/collection.tsv",
        subset_file="../../dense-vector-retrieval-system/MSMARCO-embeddings/msmarco_passages_subset.tsv"
    )
    qm = QueryManager(
        query_files=["../../dense-vector-retrieval-system/MSMARCO-embeddings/queries.dev.tsv"],
        embedding_files=["../embedding/all-MiniLM-L6-v2_encoded_dev_queries.pkl"]
    )
    
    hnsw_indexer = HNSWIndexer()
    result = hnsw_indexer.search("2", pm.get_all_ids(), 10)
    print("Top results from HNSW Indexer:")
    print(result)
    
    # Optional: Use retrieval pipeline with HNSWIndexer and CrossEncoderIndexer
    pipe = RetrievalPipeline([
        HNSWIndexer(), 
        CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")
    ])
    results, scores = pipe.retrieve("2", pm.get_all_ids(), [1000, 10])
    print("Results after retrieval pipeline:")
    print(results)
    print(scores)
    
    qrels = load_qrels('../datasets/qrels.dev.tsv')

    run = {'2': {doc_id: float(score) for doc_id, score in zip(results, scores)}}


    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg'})

    metrics = evaluator.evaluate(run)
    for metric in sorted(metrics.keys()):
        print(f'{metric}: {metrics[metric]}')
    # for metric in sorted(metrics['2'].keys()):
    #     print(f'{metric}: {metrics["2"][metric]}')
    code.interact(local=locals())