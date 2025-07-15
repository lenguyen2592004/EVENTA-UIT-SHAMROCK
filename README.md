# NanoCLIP Ensemble Project Guide

## 1. Dataset Preparation

Download and prepare the following files from [https://drive.google.com/drive/folders/1xWgEDWPcIl-JtNPDMa7hlXPHs2Sw\_\_vG](https://drive.google.com/drive/folders/1xWgEDWPcIl-JtNPDMa7hlXPHs2Sw__vG):

- **Database Images** – Download the zipped folder:  
  Database/database_images_compressed90_scaled05.zip
  Extract its contents to:  
  `database/database_images_compressed90_scaled05/`

- **Query Sets**

  - Public queries: Download the CSV from **Track 2 - Public Set** and save as:  
    `query/query_public.csv`
  - Private queries: Download the CSV from **Track 2 - Private Set** and save as:  
    `query/query_private.csv`

- **Database Metadata** – Download the JSON file from  
  Database/database.json
  and place it into:  
  `database/database.json`

- **Training Data**
  - Training images: Download the zipped folder:  
    Train Set/train_images_compressed90_scaled05.zip
    and extract it to:  
    `train_images_compressed90_scaled05/`
  - Ground Truth: Download the CSV file:  
    Train Set/gt_train.csv
    and place it appropriately.

Ensure the folder structure matches the paths above.

## 2. Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## 3. How to Run the Project

You can execute the project in three different ways. Choose one depending on your needs:

### 3.1 Using CLIP or BEiT-3 Retrieval

This mode uses the retrieval scripts provided in the `clip` folder. Ensure the dataset has been prepared as above. For BEiT-3, please read instructions in `beit-3` folder.

**Example Command:**

```bash
python run_retrieval.py \
 --image_dir database/database_images_compressed90_scaled05 \
 --test_csv_path query/query_public.csv \
 --submission_csv_path submission_public.csv
```

For private queries, change the CSV with:

```bash
--test_csv_path query/query_private.csv
```

### 3.2 Using Bi-Encoder

The Bi-Encoder solution provides a full end-to-end pipeline covering dataset creation, fine-tuning, and inference. Note that query texts can be used as original or refined.

#### Dataset Creation

To create the training dataset (with train/val split if desired), run:

```bash
python bi-encoder/datasets/Track3_t2i_not_val.py \
 --image_root path/to/train_images_compressed90_scaled05 \
 --json_path database/database.json \
 --t2i_csv_path query/query_private.csv \
 --output_dir ./dataset_output \
 --val_ratio 0.1 \
 --split_train_val \
 --random_seed 42
```

#### Fine-Tuning (File Tuning)

To fine-tune the model on your training data, run:

```bash
python bi-encoder/train_nanoclip.py \
 --train_csv_path path/to/gt_train.csv \
 --image_root_dir path/to/train_images_compressed90_scaled05 \
 --log_dir ./logs \
 --checkpoint_dir ./checkpoints \
 --num_folds 5 \
 --batch_size 32 \
 --max_epochs 20 \
 --lr 1e-4 \
 --img_model "dinov2_vitb14" \
 --unfreeze_n_blocks 6 \
 --install_faiss_gpu \
 --enable_rich_progress_bar \
 --patience 7
```

#### Inference

To perform inference with Bi-Encoder (using original queries):

```bash
python bi-encoder/run_inference.py \
 --image_dir database/database_images_compressed90_scaled05 \
 --test_csv_path query/query_public.csv \
 --submission_csv_path submission_public.csv \
 --k_candidates 20 \
 --batch_size 64 \
 --num_workers 4 \
 --model_paths "path/to/model_fold0.ckpt" "path/to/model_fold1.ckpt" "path/to/model_fold2.ckpt"
```

For private queries, replace the CSV:

```bash
--test_csv_path query/query_private.csv
```

### 3.3 Using Query Refinement

If you wish to refine your queries before inference, use the `refine_queries_script.py`. This script adjusts each query using a language model.

**Refine Queries Example:**

```bash
python refine/refine_queries_script.py \
 --model_path "OpenGVLab/InternVL3-8B" \
 --input_csv query/query_private.csv \
 --output_csv refined/query_private_refined.csv
```

Then, run inference with the refined queries:

```bash
python bi-encoder/run_inference.py \
 --image_dir database/database_images_compressed90_scaled05 \
 --test_csv_path refined/query_private_refined.csv \
 --submission_csv_path submission_private.csv \
 --k_candidates 20 \
 --batch_size 64 \
 --num_workers 4 \
 --model_paths "path/to/model_fold0.ckpt" "path/to/model_fold1.ckpt" "path/to/model_fold2.ckpt"
```

If you prefer to use the original queries, simply point to the CSV in the `query/` folder.

## 4. Additional Notes

- Always verify paths and folder structures before running.
- Adjust hyperparameters via command line arguments as needed.
- For detailed configurations, please refer to the README files in the `bi-encoder`, `clip`, and `beit3` folders.
