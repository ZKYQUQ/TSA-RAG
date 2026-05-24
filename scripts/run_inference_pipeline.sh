#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end TSA-RAG inference pipeline.
# Edit these paths before running.

INPUT_PARQUET=${INPUT_PARQUET:-data/asqa_dev_pre_retrieval.parquet}
WORK_DIR=${WORK_DIR:-outputs/asqa}
TITLE_INDEX=${TITLE_INDEX:-data/wiki/v2_title_index_w_links.json}
DOCUMENT_TREES=${DOCUMENT_TREES:-data/wiki/document_tree_w_links}
ROUTER_MODEL=${ROUTER_MODEL:-models/router}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-models/retriever}
READER_MODEL=${READER_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}
CONTRIEVER_MODEL_PATH=${CONTRIEVER_MODEL_PATH:-facebook/contriever-msmarco}

mkdir -p "$WORK_DIR"

python -u src/preprocess/graph_propagation.py \
  --input_file "$INPUT_PARQUET" \
  --output_file "$WORK_DIR/01_graph_propagated.parquet" \
  --contriever_model_path "$CONTRIEVER_MODEL_PATH" \
  --global_weight 0.6 \
  --local_weight 0.4 \
  --max_hops 5 \
  --top_k 5

python -u src/inference/path_extract_inference.py \
  --substage extract_paths \
  --input_file "$WORK_DIR/01_graph_propagated.parquet" \
  --output_file "$WORK_DIR/02_paths.parquet" \
  --title_index_file "$TITLE_INDEX" \
  --document_trees_path "$DOCUMENT_TREES"

python -u src/inference/path_extract_inference.py \
  --substage inference \
  --input_file "$WORK_DIR/02_paths.parquet" \
  --output_file "$WORK_DIR/03_path_raw.parquet" \
  --model_path "$ROUTER_MODEL" \
  --max_new_tokens 512

python -u src/inference/path_extract_inference.py \
  --substage parse_and_filter \
  --input_file "$WORK_DIR/03_path_raw.parquet" \
  --output_file "$WORK_DIR/04_path_filtered.parquet"

python -u src/inference/tree_extract_inference.py \
  --substage extract_document_subtrees \
  --input_file "$WORK_DIR/04_path_filtered.parquet" \
  --output_file "$WORK_DIR/05_subtrees.parquet"

python -u src/inference/tree_extract_inference.py \
  --substage inference \
  --input_file "$WORK_DIR/05_subtrees.parquet" \
  --output_file "$WORK_DIR/06_tree_raw.parquet" \
  --model_path "$RETRIEVER_MODEL" \
  --max_new_tokens 512

python -u src/inference/tree_extract_inference.py \
  --substage parse_and_filter \
  --input_file "$WORK_DIR/06_tree_raw.parquet" \
  --output_file "$WORK_DIR/07_tree_filtered.parquet"

python -u src/inference/reader_inference.py \
  --input_file "$WORK_DIR/07_tree_filtered.parquet" \
  --output_file "$WORK_DIR/final_output.json" \
  --language_model "$READER_MODEL" \
  --retrieve \
  --extract \
  --max_new_tokens 256
