import numpy as np
import pandas as pd
import pickle

class SingletonMeta(type):
    """A metaclass for Singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class PassageManager(metaclass=SingletonMeta):
    def __init__(self, embeddings_file, collection_file, subset_file):
        """
        Initialize PassageManager and load embeddings from a pickle file.

        Parameters:
            embeddings_file (str): The path to the pickle file containing embeddings.
            collection_file (str): The path to the collection.tsv file.
            subset_file (str): The path to the subset.tsv file.
        """
        self.passages = {}  # Dictionary to store passages by ID
        self.embeddings = {}  # Dictionary to store embeddings by ID
        self.ids = []  # List to store IDs for ordered access
        
        self.load_embeddings(embeddings_file)
        self.load_passages(collection_file, subset_file)

    def load_embeddings(self, filename):
        """Load embeddings from a pickle file."""
        print(f"Loading embeddings from: {filename}")
        with open(filename, 'rb') as file:
            passage_ids, encoded_passages = pickle.load(file)

        if not isinstance(passage_ids, list) or not isinstance(encoded_passages, list):
            raise ValueError("Pickle file must contain a tuple of (passage_ids, encoded_passages) as lists.")

        if len(passage_ids) != len(encoded_passages):
            raise ValueError("The length of passage_ids must match the length of encoded_passages.")

        for passage_id, embedding in zip(passage_ids, encoded_passages):
            self.embeddings[passage_id] = embedding
            self.ids.append(passage_id)

    def load_passages(self, collection_file, subset_file=None):
        """Load passages from a collection.tsv file and filter based on subset.tsv."""
        print(f"Loading collection file: {collection_file}")
        collection_data = pd.read_csv(collection_file, sep='\t', header=None, names=["id", "passage"], dtype={"id": str, "passage": str})

        if (subset_file):
            print(f"Loading subset file: {subset_file}")
            subset_ids = pd.read_csv(subset_file, header=None, names=["id"], dtype={"id": str})

            print("Filtering passages...")
            filtered_data = collection_data[collection_data['id'].isin(subset_ids['id'])]
        else:
            filtered_data = collection_data

        for _, row in filtered_data.iterrows():
            self.passages[row['id']] = row['passage']

    def get_passage(self, passage_id):
        """Get a passage by its ID."""
        return self.passages.get(passage_id, None)

    def get_embedding(self, passage_id):
        """Get the embedding of a passage by its ID."""
        return self.embeddings.get(passage_id, None)

    def get_all_ids(self):
        """Get a list of all passage IDs."""
        return self.ids


class QueryManager(metaclass=SingletonMeta):
    def __init__(self, query_files, embedding_files):
        """
        Initialize QueryManager and load queries from multiple TSV files.

        Parameters:
            query_files (list of str): A list of paths to the query.tsv files.
            embedding_files (list of str, optional): A list of paths to the query embeddings files (pickle).
        """
        self.queries = {}  # Dictionary to store queries by ID
        self.embeddings = {}  # Dictionary to store query embeddings by ID
        
        # Load queries from all specified query files
        for query_file in query_files:
            self.load_queries(query_file)
        
        # Load embeddings from all specified embedding files if provided
        for embedding_file in embedding_files:
            self.load_embeddings(embedding_file)

    def load_queries(self, query_file):
        """Load queries from a TSV file."""
        print(f"Loading query file: {query_file}")
        query_data = pd.read_csv(query_file, sep='\t', header=None, names=["id", "query"], dtype={"id": str, "query": str})

        for _, row in query_data.iterrows():
            self.queries[row['id']] = row['query']

    def load_embeddings(self, embedding_file):
        """Load query embeddings from a pickle file."""
        print(f"Loading query embeddings from: {embedding_file}")
        with open(embedding_file, 'rb') as file:
            query_ids, encoded_embeddings = pickle.load(file)

        if not isinstance(query_ids, list) or not isinstance(encoded_embeddings, list):
            raise ValueError("Embedding file must contain a tuple of (query_ids, encoded_embeddings) as lists.")

        if len(query_ids) != len(encoded_embeddings):
            raise ValueError("The length of query_ids must match the length of encoded_embeddings.")

        for query_id, embedding in zip(query_ids, encoded_embeddings):
            self.embeddings[query_id] = embedding

    def get_query(self, query_id):
        """Get a query by its ID."""
        return self.queries.get(query_id, None)

    def get_embedding(self, query_id):
        """Get the embedding of a query by its ID."""
        return self.embeddings.get(query_id, None)

    def get_all_query_ids(self):
        """Get a list of all query IDs."""
        return list(self.queries.keys())

class BaseIndexer():
    def search(self, query_id, documents_id, top_k):
        """
        Search the index for the top_k results based on the query ID.

        Parameters:
            query_id: The ID of the query.
            documents_id: The ID of the documents.
            top_k: The number of top results to return.

        Returns:
            documents_id, scores for the top_k results.
        """
        pass