import torch
import numpy as np
from sentence_transformers import CrossEncoder
from .utils import BaseIndexer, QueryManager, PassageManager

class CrossEncoderIndexer(BaseIndexer):
    def __init__(self, model_name):
        """
        Initialize the CrossEncoderIndexer with a pre-trained model.

        Parameters:
            model_name (str): The name of the pre-trained CrossEncoder model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name).to(self.device)

    def search(self, query_id, documents_id, top_k):
        """
        Search the index for the top_k results based on the query ID using the CrossEncoder model.

        Parameters:
            query_id: The ID of the query.
            documents_id: The list of document IDs.
            top_k: The number of top results to return.

        Returns:
            List of tuples containing (document_id, score) for the top_k results.
        """
        # Retrieve the actual query text using QueryManager
        query = QueryManager().get_query(query_id)
        if query is None:
            raise ValueError(f"Query with ID {query_id} not found.")

        # Prepare the input for the CrossEncoder on the GPU
        inputs = [(query, doc_id) for doc_id in documents_id]
        inputs = [(query, doc_id) for doc_id in documents_id]

        # Get scores from the model
        scores = self.model.predict(inputs)

        # Create a list of tuples (document_id, score)
        scored_documents = list(zip(documents_id, scores))

        # Sort the documents by score in descending order and get the top_k
        top_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)[:top_k]

        return top_documents


class CosineSimilarityIndexer(BaseIndexer):
    def __init__(self):
        """
        Initialize the CosineSimilarityIndexer with a PassageManager instance.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.passage_manager = PassageManager()

    def search(self, query_id, documents_id, top_k):
        """
        Search the index for the top_k results based on the query ID using cosine similarity.

        Parameters:
            query_id: The ID of the query.
            documents_id: The list of document IDs.
            top_k: The number of top results to return.

        Returns:
            List of tuples containing (document_id, score) for the top_k results.
        """
        # Retrieve the actual query embedding using QueryManager
        query_embedding = self.passage_manager.get_embedding(query_id)
        if query_embedding is None:
            raise ValueError(f"No embedding found for query ID {query_id}.")

        # Move query embedding to GPU
        query_embedding = torch.tensor(query_embedding, device=self.device).unsqueeze(0)  # Shape: [1, embedding_dim]

        # Get the embeddings for the documents and move to GPU
        document_embeddings = torch.stack([
            torch.tensor(self.passage_manager.get_embedding(doc_id), device=self.device)
            for doc_id in documents_id
        ])  # Shape: [num_documents, embedding_dim]

        # Calculate cosine similarity
        # Normalize the embeddings
        query_norm = query_embedding / query_embedding.norm(dim=1, keepdim=True)  # Shape: [1, embedding_dim]
        document_norms = document_embeddings / document_embeddings.norm(dim=1, keepdim=True)  # Shape: [num_documents, embedding_dim]

        # Calculate cosine similarity
        similarities = torch.mm(query_norm, document_norms.t()).squeeze(0)  # Shape: [num_documents]

        # Create a list of tuples (document_id, score)
        scored_documents = list(zip(documents_id, similarities.cpu().detach().numpy()))  # Move to CPU for compatibility with numpy

        # Sort the documents by score in descending order and get the top_k
        top_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)[:top_k]

        return top_documents

# Example usage
if __name__ == "__main__":
    # Initialize PassageManager
    passage_manager = PassageManager('../embedding/embeddings.pkl', 'collection.tsv', 'subset.tsv')
    query_manager = QueryManager('queries.tsv')

    # Initialize CrossEncoderIndexer
    cross_encoder_indexer = CrossEncoderIndexer('cross-encoder/msmarco-MiniLM-L-6-v2')

    # Initialize CosineSimilarityIndexer
    cosine_similarity_indexer = CosineSimilarityIndexer()

    # Sample query and document IDs for testing
    query_id = "query_1"
    documents_id = passage_manager.get_all_ids()
    top_k = 5

    # Perform the search using CrossEncoder
    cross_results = cross_encoder_indexer.search(query_id, documents_id, top_k)
    print("Top Results from CrossEncoder:")
    for doc_id, score in cross_results:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")

    # Perform the search using Cosine Similarity
    cosine_results = cosine_similarity_indexer.search(query_id, documents_id, top_k)
    print("Top Results from Cosine Similarity:")
    for doc_id, score in cosine_results:
        print(f"Document ID: {doc_id}, Score: {score:.4f}")