# BEiT3 Folder

This folder contains the implementation and related components for the BEiT3 model.

## Installation

Install dependencies via:

```bash
pip install -r requirements.txt
```

Refer to the documentation for training and inference details.

python run_beit3_retrieval.py \
 --final_index_path "/path/to/your/beit3_final_index.index" \
 --final_metadata_path "/path/to/your/beit3_final_metadata.npy" \
 --query_csv_path "/path/to/your/queries.csv" \
 --submission_output_path "./my_submission_results.csv" \
 --top_k 20
