#!/bin/bash

INPUT_DIR=${1:-"./datasets"}
OUTPUT_DIR=${2:-"./prior-samples"}
MODEL_NAME=${3:-"meta-llama/Meta-Llama-3-8B-Instruct"}
REASONING=${4:-"false"}

# Find all the JSON files in the input directory
input_files=$(find "$INPUT_DIR" -type f -name "*.json")

if [ "$REASONING" = "true" ]; then
    echo "Generating samples with reasoning using model: $MODEL_NAME"
else
    echo "Generating samples without reasoning using model: $MODEL_NAME"
fi

samples_dir=$OUTPUT_DIR/$MODEL_NAME
mkdir -p "$samples_dir"

# Iterate over these files
for file in $input_files; do
    base_name=$(basename "$file" .json)

    if [ "$REASONING" = "true" ]; then
        output_file="$samples_dir/${base_name}_reasoning.json"
    else
        output_file="$samples_dir/${base_name}.json"
    fi

    echo "Processing $file..."

    # Use the LLM to generate samples
    if [ "$REASONING" = "true" ]; then
        uv run llmprior.py \
            --model-name "$MODEL_NAME" \
            --input-path "$file" \
            --output-path "$output_file" \
            --reasoning
    else
        uv run llmprior.py \
            --model-name "$MODEL_NAME" \
            --input-path "$file" \
            --output-path "$output_file"
    fi

    echo "Samples saved to $output_file"
done
