python run_inference.py \
 --image_dir "/path/to/your/database_images" \
 --test_csv_path "/path/to/your/custom_query.csv" \
 --submission_csv_path "./my_ensemble_submission.csv" \
 --k_candidates 20 \
 --batch_size 64 \
 --num_workers 4 \
 --model_paths "/path/to/model_fold0.ckpt" "/path/to/model_fold1.ckpt" "/path/to/model_fold2.ckpt"
