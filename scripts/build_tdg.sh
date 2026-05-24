#!/usr/bin/env bash
set -euo pipefail

# Build the offline Tiered Document Graph from Wikipedia dump shards.
# The full Wikipedia 2018 dump is large; run each step on sufficient CPU/GPU/storage.

RAW_XML_DIR=${RAW_XML_DIR:-data/wiki/raw_xmls}
TEMP_DIR=${TEMP_DIR:-data/wiki/temp_pages}
TREE_DIR=${TREE_DIR:-data/wiki/v1_cleaned_trees_jsonl}
TREE_LINK_DIR=${TREE_LINK_DIR:-data/wiki/document_tree_w_links}
TITLE_INDEX=${TITLE_INDEX:-data/wiki/v1_cleaned_title_index.json}
TITLE_INDEX_LINKS=${TITLE_INDEX_LINKS:-data/wiki/v2_title_index_w_links.json}
CONTRIEVER_MODEL_PATH=${CONTRIEVER_MODEL_PATH:-facebook/contriever-msmarco}

# Parses Wikipedia XML pages into hierarchical document trees.
python -u src/preprocess/parse_to_jsonlStruct.py \
  --input_folder "$RAW_XML_DIR" \
  --temp_folder "$TEMP_DIR" \
  --output_folder "$TREE_DIR" \
  --max_workers 8

# Builds a title-to-file index for fast tree lookup.
python -u src/preprocess/file_utils.py \
  --jsonl_dir "$TREE_DIR" \
  --output_index_file "$TITLE_INDEX"

# Resolves Wikipedia links and attaches link metadata to trees.
python -u src/preprocess/extract_links.py \
  --input_folder "$TREE_DIR" \
  --output_folder "$TREE_LINK_DIR" \
  --index_file "$TITLE_INDEX" \
  --max_workers 8

python -u src/preprocess/file_utils.py \
  --jsonl_dir "$TREE_LINK_DIR" \
  --output_index_file "$TITLE_INDEX_LINKS"

# Builds the Neo4j document graph with Contriever embeddings.
python -u src/preprocess/build_graph.py \
  --title_index_path "$TITLE_INDEX_LINKS" \
  --document_tree_path "$TREE_LINK_DIR" \
  --contriever_model_path "$CONTRIEVER_MODEL_PATH"
