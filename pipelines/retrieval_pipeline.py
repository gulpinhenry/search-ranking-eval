import numpy as np

class RetrievalPipeline:
    def __init__(self, indexers):
        """
        Initialize the retrieval pipeline with a list of indexers.

        Parameters:
            indexers (list): A list of indexers to use in the retrieval process.
        """
        self.indexers = indexers

    def retrieve(self, query_id, documents_id, top_k):
        """
        Retrieve the top documents based on the query using the specified indexers.

        Parameters:
            query_id (str): The query id string.
            documents (list): A list of document IDs.
            top_k (list): A list of integers specifying the number of top results to retrieve for each indexer.

        Returns:
            List of top documents and their corresponding scores (if applicable).
        """
        # Store results from the current step
        results = documents_id
        scores = None
        # Sequentially apply each indexer
        for indexer, k in zip(self.indexers, top_k):
            search_result = indexer.search(query_id, results, k)  # Pass document IDs instead of embeddings
            results = [x[0] for x in search_result]
            scores = [x[1] for x in search_result]


        return results, scores  # No scores if the last indexer is not a scorer