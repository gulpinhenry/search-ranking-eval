# Usage
``` shell
python encode_passages.py \
    --collection_file <path_to_collection.tsv> \
    --subset_file <path_to_subset.tsv> \
    --output_file <path_to_output.pkl> \
    [--model_name <model_name>] \
    [--batch_size <batch_size>]
```

## Required Arguments
- --collection_file: Path to the collection.tsv file containing all passages. Each line should have a passage ID and the passage text, separated by a tab.
- --output_file: Path to save the output pickle file. The file will contain a tuple: (passage_ids, embeddings).

## Optional Arguments
- --subset_file: Path to the subset.tsv file containing passage IDs that need to be encoded. Each line should contain a single ID. (use for the collection for passages only if needed)
- --model_name: Name of the SentenceTransformer model to use for encoding. Defaults to all-MiniLM-L6-v2. You can choose any model from the SentenceTransformers model repository.
- --batch_size: Batch size for encoding passages. Defaults to 64. Adjust this based on your system's memory capacity.
