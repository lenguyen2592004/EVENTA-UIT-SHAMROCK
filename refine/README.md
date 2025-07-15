python refine_queries_script.py \
 --input_csv "./my_data/query_private.csv" \
 --output_csv "./results/submission_refined.csv" \
 --model_path "OpenGVLab/InternVL3-8B" \
 --tp 1 # Ví dụ chỉ dùng 1 GPU
