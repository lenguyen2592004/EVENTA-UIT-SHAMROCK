# BEiT3 Folder

This folder contains the implementation and related components for the BEiT3 model.

## Installation

Install dependencies via:

```bash
pip install -r requirements.txt
```

Refer to the documentation for training and inference details.

## Creating FAISS Index

Before running inference, you must create the FAISS index and metadata file from your image dataset. Use the provided script to generate index shards and metadata. If needed, merge the shards into a single index file.

Example command to create index shards and metadata:

```bash
python datasets.py \
 --image_folder "/path/to/your/database_images" \
 --limit_images 0 \
 --chunk_size 10000 \
 --output_index_dir "./faiss_index_shards" \
 --metadata_output_path "./beit3_final_metadata.npy"
```

If you need to merge the shards, use your preferred FAISS index merging tool to create a final index (e.g., beit3_final_index.index).

## Running Inference

Run the retrieval script using:

```bash
python run_retrieval.py \
 --final_index_path "/path/to/your/beit3_final_index.index" \
 --final_metadata_path "/path/to/your/beit3_final_metadata.npy" \
 --query_csv_path "/path/to/your/queries.csv" \
 --submission_output_path "./my_submission_results.csv" \
 --top_k 20
```
