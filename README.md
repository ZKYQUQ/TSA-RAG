# TSA-RAG

This repository provides codes for TSA-RAG, a structure-driven retrieval-augmented generation framework. TSA-RAG builds a Tiered Document Graph (TDG) over Wikipedia, combining a cross-document graph with a hierarchical tree for each document. At inference time, TSA-RAG follows an Expand-Route-Retrieve paradigm to construct compact structure-aware evidence for answer generation.

## Environment

```bash
conda env create -f environment.yml
```

For graph construction and expansion, start Neo4j and set:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
export CONTRIEVER_MODEL_PATH=facebook/contriever-msmarco
```

## Data Preparation

The input parquet should contain one row per question and include:

- a question field: `ambiguous_question`, `question_text`, `title`, or `question`
- `pre_retrieved_passages`: a list of initial seed passages, where each item contains at least `title` and `text`
- optional dataset-specific fields for evaluation formatting

## Build TDG

Edit paths in `scripts/build_tdg.sh`, then run:

```bash
bash scripts/build_tdg.sh
```

## Inference

Edit the model and data paths in `scripts/run_inference_pipeline.sh`, then run:

```bash
bash scripts/run_inference_pipeline.sh
```

Step-by-step commands:

### Path-Consistent Graph Expansion
Expands seed documents into query-focused subgraphs.
```bash
mkdir -p outputs/asqa

python src/preprocess/graph_propagation.py \
  --input_file data/asqa_dev_pre_retrieval.parquet \
  --output_file outputs/asqa/01_graph_propagated.parquet \
  --global_weight 0.6 \
  --local_weight 0.4 \
  --max_hops 5 \
  --top_k 5
```

### Coarse-Grained Document Routing
Converts subgraphs into paths and selects relevant documents/headings.
```bash
python src/inference/path_extract_inference.py \
  --substage extract_paths \
  --input_file outputs/asqa/01_graph_propagated.parquet \
  --output_file outputs/asqa/02_paths.parquet \
  --title_index_file data/wiki/v2_title_index_w_links.json \
  --document_trees_path data/wiki/document_tree_w_links

python src/inference/path_extract_inference.py \
  --substage inference \
  --input_file outputs/asqa/02_paths.parquet \
  --output_file outputs/asqa/03_path_raw.parquet \
  --model_path models/router \
  --max_new_tokens 512

python src/inference/path_extract_inference.py \
  --substage parse_and_filter \
  --input_file outputs/asqa/03_path_raw.parquet \
  --output_file outputs/asqa/04_path_filtered.parquet
```

### Fine-Grained Knowledge Retrieval
Builds pruned subtrees and extracts relevant paragraph nodes.
```bash
python src/inference/tree_extract_inference.py \
  --substage extract_document_subtrees \
  --input_file outputs/asqa/04_path_filtered.parquet \
  --output_file outputs/asqa/05_subtrees.parquet

python src/inference/tree_extract_inference.py \
  --substage inference \
  --input_file outputs/asqa/05_subtrees.parquet \
  --output_file outputs/asqa/06_tree_raw.parquet \
  --model_path models/retriever \
  --max_new_tokens 512

python src/inference/tree_extract_inference.py \
  --substage parse_and_filter \
  --input_file outputs/asqa/06_tree_raw.parquet \
  --output_file outputs/asqa/07_tree_filtered.parquet
```

### Structure-Aware Generation
Generates final answers from structured evidence context.
```bash
python src/inference/reader_inference.py \
  --input_file outputs/asqa/07_tree_filtered.parquet \
  --output_file outputs/asqa/final_output.json \
  --language_model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --retrieve \
  --extract \
  --max_new_tokens 256
```

## Evaluation

Short-form QA:

```bash
python src/eval/short_form_eval.py \
  --result_file outputs/hotpotqa/final_output.json
```

Long-form evaluation utilities are under `src/ALCE`. Please follow the argument format in `src/ALCE/README.md`.
