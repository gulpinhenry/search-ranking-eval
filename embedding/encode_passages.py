import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import torch

def encode_passages(collection_file, subset_file, output_file, model_name, batch_size):
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the SentenceTransformer model and move it to the appropriate device
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    model = model.to(device)

    # Step 1: Load the collection.tsv file
    print(f"Loading collection file: {collection_file}")
    collection_data = pd.read_csv(collection_file, sep='\t', header=None, names=["id", "passage"], dtype={"id": str, "passage": str})

    if (subset_file):
    # Step 2: Load the subset.tsv file
        print(f"Loading subset file: {subset_file}")
        subset_ids = pd.read_csv(subset_file, header=None, names=["id"], dtype={"id": str})

        # Keep only the passages specified in subset.tsv
        print("Filtering passages...")
        filtered_data = collection_data[collection_data['id'].isin(subset_ids['id'])]
    else:
        filtered_data = collection_data

    # Step 3: Encode the filtered passages
    print("Encoding passages...")
    passage_texts = filtered_data["passage"].tolist()
    passage_ids = filtered_data["id"].tolist()

    encoded_passages = []

    # Encode in batches for efficiency
    for i in tqdm(range(0, len(passage_texts), batch_size), desc="Encoding batches"):
        batch = passage_texts[i:i + batch_size]
        # Move batches to GPU if available and encode
        encoded = model.encode(batch, convert_to_numpy=True, device=device)
        encoded_passages.extend(encoded)

    # Step 4: Save the encoded passages to a pickle file
    print(f"Saving encoded passages to pickle file: {output_file}")
    with open(output_file, "wb") as f:
        pickle.dump((passage_ids, encoded_passages), f)

    print("Encoding complete!")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Encode passages using SentenceTransformers and save to a pickle file.")
    parser.add_argument("--collection_file", type=str, required=True, help="Path to the collection.tsv file.")
    parser.add_argument("--subset_file", type=str, default=None, help="Path to the subset.tsv file containing IDs to encode.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the encoded passages (pickle file).")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the SentenceTransformers model to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding.")

    args = parser.parse_args()

    # Run the encoding process
    encode_passages(
        collection_file=args.collection_file,
        subset_file=args.subset_file,
        output_file=args.output_file,
        model_name=args.model_name,
        batch_size=args.batch_size
    )