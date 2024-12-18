import code
from utils import *
from indexers import *
from retrieval_pipeline import * 
import pytrec_eval

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
    qrels = load_qrels('./datasets/msmarco_hw3/qrels.dev.tsv')

    
    run = {'2': {doc_id: float(score) for doc_id, score in zip(results, scores)}}
    run = {}
    for query_id in qm.get_all_query_ids():
        results, scores = pipe.retrieve(query_id, pm.get_all_ids(), [10])
        run[query_id] = {doc_id: float(score) for doc_id, score in zip(results, scores)}


    evaluator = pytrec_eval.RelevanceEvaluator(qrels,  pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

    code.interact(local=locals())