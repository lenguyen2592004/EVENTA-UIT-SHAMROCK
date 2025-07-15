# NanoCLIP Inference Guide

Use the command below to run inference. Replace the placeholder paths with your actual paths.

```bash
python run_inference.py \
 --image_dir "/path/to/database/database_images_compressed90_scaled05" \
 --test_csv_path "/path/to/query/query_public.csv" \
 --submission_csv_path "./submission_public.csv" \
 --k_candidates 20 \
 --batch_size 64 \
 --num_workers 4 \
 --model_paths "/path/to/model_fold0.ckpt" "/path/to/model_fold1.ckpt" "/path/to/model_fold2.ckpt"
```

For private queries, adjust the --test_csv_path parameter accordingly.
