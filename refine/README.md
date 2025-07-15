# Refine Module

This module performs post-processing and refinement of retrieval results.

## How to Use

Run the refine script by replacing the placeholder paths with your respective directories and files. Ensure all paths use forward slashes.

Example command:

```bash
python run_refine.py \
 --input_dir "./my_results" \
 --config_path "/path/to/config.json" \
 --output_dir "./refined_results"
```

python refine_queries_script.py \
 --input_csv "./my_data/query_private.csv" \
 --output_csv "./results/submission_refined.csv" \
 --model_path "OpenGVLab/InternVL3-8B" \
 --tp 1 # Ví dụ chỉ dùng 1 GPU
