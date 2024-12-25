#!/bin/bash

# Set the directory containing the evaluation result files
eval_dir="../../search-ranking-eval/output/evaluation_results_1"

# Set the directory containing the qrels files
qrels_dir="../MSMARCO-embeddings"

# Loop through all files in the evaluation results directory
for file in "$eval_dir"/*; do
    # Check if the file exists
    if [[ -f "$file" ]]; then
        # Extract the basename of the file
        base_name=$(basename "$file")
        
        # Split the basename to get the first part for the new qrels file
        qrels_base="${base_name%%-*}"  # Get everything before the first '-'
        
        # Construct the new qrels filename
        new_qrels_file="$qrels_dir/$qrels_base"

        # Check if the new qrels file exists
        if [[ -f "$new_qrels_file" ]]; then
            # Run the trec_eval command
            ./trec_eval -m recall -m map -m ndcg -m recip_rank "$new_qrels_file" "$file"

            # Optionally, you can echo the command being run for debugging
            echo "Running: ./trec_eval -m recall -m map -m ndcg -m recip_rank $new_qrels_file $file"
        else
            echo "Warning: Qrels file $new_qrels_file does not exist."
        fi
    fi
done
