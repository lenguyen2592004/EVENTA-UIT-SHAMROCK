## 1. Dataset Preparation

Download the dataset from the following link:
[Download Dataset](https://drive.google.com/drive/folders/1xWgEDWPcIl-JtNPDMa7hlXPHs2Sw__vG)

After downloading, extract and organize the files as below:

- **database/database_images_compressed90_scaled05/**: Folder containing the database images.
- **query/**: Contains the CSV files:
  - query_public.csv (public query data)
  - query_private.csv (private query data)
- **train_images_compressed90_scaled05/**: Folder containing training images.
- **database/database.json**: JSON file with database information.

## 2. Environment Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## 3. Running Inference

Example command for public queries:

```bash
python bi-encoder/run_inference.py --image_dir database/database_images_compressed90_scaled05 --test_csv_path query/query_public.csv --submission_csv_path submission_public.csv
```

Example command for private queries:

```bash
python bi-encoder/run_inference.py --image_dir database/database_images_compressed90_scaled05 --test_csv_path query/query_private.csv --submission_csv_path submission_private.csv
```

## 4. Additional Notes

- Other parameters can be adjusted via command line.
- Please double-check the paths and directory structure before running.
