import time
import torch
from torch import nn
import numpy as np
from sentence_transformers import CrossEncoder
from utils import BaseIndexer, QueryManager, PassageManager
import math
import faiss


class CrossEncoderIndexer(BaseIndexer):
    def __init__(self, model_name):
        """
        Initialize the CrossEncoderIndexer with a pre-trained model.

        Parameters:
            model_name (str): The name of the pre-trained CrossEncoder model.
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CrossEncoder(model_name, device=self.device)
        self.passage_manager = PassageManager()

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

        # Prepare the input for the CrossEncoder
        inputs = [self.passage_manager.get_passage(i) for i in documents_id]

        # Get ranks from the model
        ranks = self.model.rank(query, inputs)

        # Create a list of tuples (document_id, score)
        scored_documents = [(documents_id[rank['corpus_id']], rank['score']) for rank in ranks]

        # Sort the documents by score in descending order and get the top_k
        top_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)[:top_k]

        return top_documents
    
    def __str__(self):
        return f"C({str(self.model_name)})"


# Bi-encoder
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
        query_embedding = QueryManager().get_embedding(query_id)
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
    
    def __str__(self):
        return f"Cos()"
    
class BM25Indexer(BaseIndexer):
    def __init__(self):
        """
        Initialize the BM25Indexer with a custom BM25 implementation.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.passage_manager = PassageManager()
        self.corpus_ids = self.passage_manager.get_all_ids()
        self.corpus_texts = [self.passage_manager.get_passage(doc_id) for doc_id in self.corpus_ids]
        # Tokenize the corpus
        self.corpus_tokens = [doc.split() for doc in self.corpus_texts]
        # Build a mapping from document ID to tokens
        self.doc_id_to_tokens = {doc_id: tokens for doc_id, tokens in zip(self.corpus_ids, self.corpus_tokens)}
        # Compute document frequencies (DF) for each term
        self.df = {}
        for tokens in self.corpus_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1
        # Compute inverse document frequencies (IDF) for each term
        self.num_documents = len(self.corpus_tokens)
        self.idf = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log((self.num_documents - freq + 0.5) / (freq + 0.5) + 1)
        # Compute document lengths and average document length
        self.doc_lengths = {doc_id: len(tokens) for doc_id, tokens in self.doc_id_to_tokens.items()}
        if self.num_documents == 0:
            raise ValueError("No documents found. Please check the subset file or use the full collection.")
        self.avgdl = sum(self.doc_lengths.values()) / self.num_documents
        # Set BM25 parameters
        self.k1 = 1.5
        self.b = 0.75
        self.term_frequencies = {}
        for docId in self.corpus_ids:
            term_frequencies_in_doc = {}
            for token in self.doc_id_to_tokens[docId]:
                term_frequencies_in_doc[token] = term_frequencies_in_doc.get(token, 0) + 1
            self.term_frequencies[docId] = term_frequencies_in_doc

    def score(self, query_tokens, doc_id):
        """
        Compute the BM25 score for a single document and query.
        """
        score = 0.0
        doc_tokens = self.doc_id_to_tokens.get(doc_id, [])
        doc_len = self.doc_lengths.get(doc_id, 0)
        # term_frequencies = {}
        # for token in doc_tokens:
        #     term_frequencies[token] = term_frequencies.get(token, 0) + 1
        for term in query_tokens:
            if term in self.idf:
                tf = self.term_frequencies.get(doc_id, {}).get(term, 0)
                numerator = self.idf[term] * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += numerator / denominator
        return score

    def search(self, query_id, documents_id, top_k):
        """
        Search the index for the top_k results based on the query ID using BM25.

        Parameters:
            query_id: The ID of the query.
            documents_id: The list of document IDs to search over.
            top_k: The number of top results to return.

        Returns:
            List of tuples containing (document_id, score) for the top_k results.
        """
        # Retrieve the actual query text using QueryManager
        query = QueryManager().get_query(query_id)
        if query is None:
            raise ValueError(f"Query with ID {query_id} not found.")
        # Tokenize the query
        query_tokens = query.split()
        # Compute BM25 scores for the documents
        scored_documents = []
        for doc_id in documents_id:
            if doc_id in self.doc_id_to_tokens:
                score = self.score(query_tokens, doc_id)
                scored_documents.append((doc_id, score))
        # Sort the documents by score in descending order and get the top_k
        top_documents = sorted(scored_documents, key=lambda x: x[1], reverse=True)[:top_k]
        return top_documents

    def __str__(self):
        return f"BM25({self.k1}-{self.b})"

class HNSWIndexer(BaseIndexer):
    def __init__(self, m=8, ef_construction=200, ef_search=50):
        """
        Initialize the HNSWIndexer and build the index using passage embeddings.
        """
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.passage_manager = PassageManager()
        self.index = None
        self.ids = []
        self.build_index()

    def build_index(self):
        """
        Build the HNSW index using embeddings from PassageManager.
        """
        embeddings = []
        ids = []
        for doc_id in self.passage_manager.get_all_ids():
            embedding = self.passage_manager.get_embedding(doc_id)
            if embedding is not None:
                embeddings.append(embedding)
                ids.append(doc_id)
        if not embeddings:
            raise ValueError("No embeddings found in PassageManager.")
        embeddings = np.stack(embeddings).astype('float32')
        self.ids = np.array(ids)
        d = embeddings.shape[1]  # Dimension of embeddings

        # Initialize HNSW index
        self.index = faiss.IndexHNSWFlat(d, self.m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search

        # Normalize embeddings for inner product similarity
        faiss.normalize_L2(embeddings)

        # Add embeddings to the index
        self.index.add(embeddings)

    def search(self, query_id, documents_id, top_k):
        """
        Search for the top_k similar documents to the query.

        Parameters:
            query_id (str): The ID of the query.
            documents_id (list): The list of document IDs to search over.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[str, float]]: List of tuples containing (document_id, score).
        """
        if self.index is None:
            raise ValueError("Index not built.")

        # Step 1: Retrieve the query embedding
        query_embedding = QueryManager().get_embedding(query_id)
        if query_embedding is None:
            raise ValueError(f"No embedding found for query ID {query_id}.")

        # Step 2: Prepare the query embedding
        query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Step 3: Search the index
        distances, indices = self.index.search(query_embedding, top_k)

        # Step 4: Map internal indices to document IDs and scores
        top_documents = []
        if indices.size > 0:
            for doc_idx, score in zip(indices[0], distances[0]):
                doc_id = self.ids[doc_idx]
                top_documents.append((doc_id, float(score)))
                

        return top_documents[:top_k]   
    
    def __str__(self):
        return f"HNSW({self.m}-{self.ef_construction}-{self.ef_search})"
    

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

