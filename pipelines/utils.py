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
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        
        if not isinstance(data, dict):
            raise ValueError("Pickle file must contain a dictionary with passage IDs as keys.")
        
        for passage_id, embedding in data.items():
            self.embeddings[passage_id] = embedding
            self.ids.append(passage_id)

    def load_passages(self, collection_file, subset_file):
        """Load passages from a collection.tsv file and filter based on subset.tsv."""
        print(f"Loading collection file: {collection_file}")
        collection_data = pd.read_csv(collection_file, sep='\t', header=None, names=["id", "passage"], dtype={"id": str, "passage": str})

        print(f"Loading subset file: {subset_file}")
        subset_ids = pd.read_csv(subset_file, header=None, names=["id"], dtype={"id": str})

        print("Filtering passages...")
        filtered_data = collection_data[collection_data['id'].isin(subset_ids['id'])]

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
    def __init__(self, query_file):
        """
        Initialize QueryManager and load queries from a TSV file.

        Parameters:
            query_file (str): The path to the query.tsv file.
        """
        self.queries = {}  # Dictionary to store queries by ID
        self.load_queries(query_file)

    def load_queries(self, query_file):
        """Load queries from a TSV file."""
        print(f"Loading query file: {query_file}")
        query_data = pd.read_csv(query_file, sep='\t', header=None, names=["id", "query"], dtype={"id": str, "query": str})

        for _, row in query_data.iterrows():
            self.queries[row['id']] = row['query']

    def get_query(self, query_id):
        """Get a query by its ID."""
        return self.queries.get(query_id, None)

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