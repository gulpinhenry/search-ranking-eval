import code
import os
from tqdm import tqdm
from indexers import BM25Indexer, CosineSimilarityIndexer, CrossEncoderIndexer, HNSWIndexer
from retrieval_pipeline import RetrievalPipeline
from utils import PassageManager, QueryManager
import time

class Evaluation:
    def __init__(self, output_file_folder, pipe, top_n, output_top_n, qrel_file):
        """
        Initializes the Evaluation class and opens the output file for writing.

        Parameters:
        - output_file_path (str): Path to save the run file.
        """
        self.pm = PassageManager()
        self.pipe = pipe
        self.top_n = top_n
        self.output_top_n = output_top_n
        self.output_file_folder = output_file_folder
        self.files = [] # files corresponding to output_top_n
        for output_n in output_top_n:
            file_name = os.path.join(output_folder, f"{os.path.basename(qrel_file)}-{str(pipe)}-{str(top_n)}-{output_n}.txt".replace(" ", "").replace("/","-"))
            self.files.append(open(file_name, 'w', encoding='utf-8'))

    def write_results_to_run_file(self, ranked_doc_ids, scores, query_id, run_id='run_1'):
        """
        Writes the retrieval results to the run file.

        Parameters:
        - ranked_doc_ids (list): List of ranked document IDs.
        - scores (list): Corresponding list of scores for the ranked documents.
        - query_id (str): The ID of the query.
        - run_id (str): Identifier for the run (default is 'run_1').
        """
        for rank, (doc_id, score) in enumerate(zip(ranked_doc_ids, scores), start=1):
            self.file.write(f"{query_id}\t0\t{doc_id}\t{rank}\t{score:.6f}\t{run_id}\n")

        print(f"Results for query ID {query_id} written to {self.output_file_folder}.")

    def evaluate_pipeline(self, query_id, run_id='run_1'):
        """
        Evaluates the retrieval pipeline and writes results to the run file for a single query.

        Parameters:
        - pipeline: The retrieval pipeline to use.
        - query_id (str): The ID of the query to evaluate.
        - top_n (list of int): Number of top results to retrieve.
        - run_id (str): Identifier for the run (default is 'run_1').
        """
        results, scores = self.pipe.retrieve(query_id, self.pm.get_all_ids(), self.top_n)
        for output_n_index in range(len(self.output_top_n)):
            output_n = self.output_top_n[output_n_index]
            for rank, (doc_id, score) in enumerate(zip(results[:output_n], scores[:output_n]), start=1):
                self.files[output_n_index].write(f"{query_id}\t0\t{doc_id}\t{rank}\t{score:.6f}\t{run_id}\n")
            print(f"Results for query ID {query_id} written to {self.output_file_folder} top_n: {output_n}.")
            

    def evaluate_multiple_queries(self, query_ids, run_id='run_1'):
        """
        Evaluates the retrieval pipeline for multiple queries and writes results to the run file.

        Parameters:
        - query_ids (list): List of query IDs to evaluate.
        - run_id (str): Identifier for the run (default is 'run_1').
        """
        for query_id in tqdm(query_ids, desc="Evaluating Queries"):
            self.evaluate_pipeline(query_id, run_id)

    def close(self):
        """Closes the output file."""
        for file in self.files:
            file.close()
        print(f"Files {self.output_file_folder} closed.")
            
    def get_unique_query_ids(self, qrels_file_path):
        unique_query_ids = set()  # Use a set to store unique query IDs

        with open(qrels_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()  # Split the line into parts
                if len(parts) >= 1:  # Ensure there's at least one part (the query ID)
                    query_id = parts[0]  # The first part is the query ID
                    unique_query_ids.add(query_id)  # Add to the set

        return unique_query_ids

def run_evaluation(pipeline, top_n, qrels_files, output_folder, output_top_n):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Initialize RetrievalPipeline
    pipe_start = time.time()
    pipe = pipeline  # Use the provided pipeline
    pipe_time = time.time() - pipe_start
    print(f"RetrievalPipeline Initialization Time: {pipe_time:.2f} seconds")

    for qrels_file in qrels_files:
        # Initialize Evaluation for the current QRELs file
        
        eval_start = time.time()
        evaluation = Evaluation("../output/evaluation_results", pipe, top_n, output_top_n, qrels_file)

        eval_time = time.time() - eval_start
        print(f"Evaluation Initialization Time for {qrels_file}: {eval_time:.2f} seconds")

        # List of query IDs for evaluation
        query_ids_start = time.time()
        query_ids = evaluation.get_unique_query_ids(qrels_file)
        query_ids_time = time.time() - query_ids_start
        print(f"Query IDs Retrieval Time for {qrels_file}: {query_ids_time:.2f} seconds")

        # Evaluate for multiple query IDs
        eval_queries_start = time.time()
        evaluation.evaluate_multiple_queries(query_ids, top_n)
        eval_queries_time = time.time() - eval_queries_start
        print(f"Evaluation of Multiple Queries Time for {qrels_file}: {eval_queries_time:.2f} seconds")

        # Close the evaluation instance properly
        evaluation.close() 
        
            
# Example usage
if __name__ == "__main__":
    start_time = time.time()

    # Initialize PassageManager
    pm_start = time.time()
    pm = PassageManager(
        embeddings_file="../embedding/all-MiniLM-L6-v2_encoded_passages.pkl",
        collection_file="../datasets/collection.tsv",
        subset_file="../datasets/msmarco_passages_subset.tsv"
    )
    pm_time = time.time() - pm_start
    print(f"PassageManager Initialization Time: {pm_time:.2f} seconds")

    # Initialize QueryManager
    qm_start = time.time()
    qm = QueryManager(
        query_files=["../datasets/queries.eval.tsv", "../datasets/queries.dev.tsv"],
        embedding_files=["../embedding/all-MiniLM-L6-v2_encoded_eval_queries.pkl", 
                         "../embedding/all-MiniLM-L6-v2_encoded_dev_queries.pkl"]
    )
    qm_time = time.time() - qm_start
    print(f"QueryManager Initialization Time: {qm_time:.2f} seconds")

    # # Initialize Evaluation class
    # eval_start = time.time()
    # evaluation = Evaluation("../output/run_file.tsv")
    # eval_time = time.time() - eval_start
    # print(f"Evaluation Initialization Time: {eval_time:.2f} seconds")

    # Initialize RetrievalPipeline
    pipe_start = time.time()
    hnsw = HNSWIndexer()
    bm25 = BM25Indexer()
    # pipe = RetrievalPipeline([
    #     bm25, 
    #     CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")
    # ])
    pipelines = [
        RetrievalPipeline([
        bm25, 
        CosineSimilarityIndexer()]),
        
        RetrievalPipeline([
        bm25, 
        CrossEncoderIndexer("cross-encoder/ms-marco-TinyBERT-L-2-v2")]),
        
        RetrievalPipeline([
        bm25, 
        hnsw]),
        
        RetrievalPipeline([
        hnsw, 
        CrossEncoderIndexer("cross-encoder/ms-marco-MiniLM-L-12-v2")]),
        
        RetrievalPipeline([
        hnsw, 
        CrossEncoderIndexer("cross-encoder/ms-marco-TinyBERT-L-2-v2")]),
        
        RetrievalPipeline([
        hnsw, 
        bm25]),
        
    ]
    pipe_time = time.time() - pipe_start
    print(f"RetrievalPipeline Initialization Time: {pipe_time:.2f} seconds")

    top_n = [1000, 100]
    qrels_files = ["../datasets/qrels.eval.one.tsv", "../datasets/qrels.eval.two.tsv", "../datasets/qrels.dev.fixed.tsv"]
    output_folder = "../output/evaluation_results"
    for pipe in pipelines:
        run_evaluation(pipe, top_n, qrels_files, output_folder, [100, 10])

    # Start interactive console
    code.interact(local=locals())