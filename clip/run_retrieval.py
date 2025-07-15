import json
import os
import zipfile
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss
from sklearn.metrics import average_precision_score
import argparse # Thêm module argparse

# Configuration (These will be default values, can be overridden by command line arguments)
# DATASET_PATH_DEFAULT = "/kaggle/input/eventa-track-2"
# IMAGE_DIR_DEFAULT = '/kaggle/input/database-eventa/database_images_compressed90_scaled05'
# QUERY_CSV_DEFAULT = "/kaggle/input/refine-private/refine_private.csv"
# OUTPUT_DIR_DEFAULT = "./output/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(dataset_path, image_dir, query_csv):
    """Load and preprocess EVENTA dataset."""
    database_json = os.path.join(dataset_path, "database.json")
    with open(database_json, "r") as f:
        database = json.load(f)
    
    image_text_pairs = []
    image_ids = []
    for article_id, article in database.items():
        title = article.get("title", "")
        content = article.get("content", "")[:512]
        text = f"{title}. {content}"
        for img_id in article.get("images", []):
            img_path = os.path.join(image_dir, f"{img_id}.jpg")
            if os.path.exists(img_path):
                image_text_pairs.append({"image_path": img_path, "text": text, "article_id": article_id})
                image_ids.append(img_id)
    
    queries = pd.read_csv(query_csv)
    query_data = queries[["query_index", "query_text"]].rename(columns={"query_index": "query_id"})
    
    return image_text_pairs, image_ids, query_data, database

def create_small_dataset(image_text_pairs, query_data, database, num_images=100, num_queries=10):
    """Create a small dataset for testing."""
    set_seed(RANDOM_SEED)
    
    # Sample images
    sampled_pairs = random.sample(image_text_pairs, min(num_images, len(image_text_pairs)))
    sampled_image_ids = [pair["image_path"].split("/")[-1].replace(".jpg", "") for pair in sampled_pairs]
    
    # Sample queries and simulate ground-truth
    sampled_queries = query_data.sample(n=min(num_queries, len(query_data)), random_state=RANDOM_SEED)
    ground_truth = {}
    for _, query in sampled_queries.iterrows():
        # Simulate ground-truth: assume query matches images from one random article
        article_id = random.choice(list(database.keys()))
        relevant_images = [img for img in database[article_id]["images"] if img in sampled_image_ids]
        ground_truth[query["query_id"]] = relevant_images if relevant_images else [random.choice(sampled_image_ids)] # Ensure at least one image
    
    return sampled_pairs, sampled_image_ids, sampled_queries, ground_truth

def initialize_model():
    """Initialize CLIP model and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def fine_tune_model(model, processor, image_text_pairs):
    """Placeholder for fine-tuning CLIP."""
    print("Fine-tuning placeholder: Implement contrastive loss training here.")
    # Example:
    # from torch.utils.data import DataLoader, Dataset
    # # Define a simple dataset and dataloader for fine-tuning
    # class CustomDataset(Dataset):
    #     def __init__(self, image_text_pairs, processor):
    #         self.image_text_pairs = image_text_pairs
    #         self.processor = processor
    #     def __len__(self):
    #         return len(self.image_text_pairs)
    #     def __getitem__(self, idx):
    #         item = self.image_text_pairs[idx]
    #         image = Image.open(item["image_path"]).convert("RGB")
    #         text = item["text"]
    #         inputs = self.processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True)
    #         return {k: v.squeeze(0) for k, v in inputs.items()}
    # 
    # dataset = CustomDataset(image_text_pairs, processor)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # 
    # model.train()
    # for epoch in range(1): # example for 1 epoch
    #     for batch in dataloader:
    #         input_ids = batch["input_ids"].to(DEVICE)
    #         attention_mask = batch["attention_mask"].to(DEVICE)
    #         pixel_values = batch["pixel_values"].to(DEVICE)
    #         
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #         print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    # model.eval()
    return model

def compute_image_embeddings(model, processor, image_text_pairs, batch_size=32):
    """Compute image embeddings for the database."""
    image_embeddings = []
    print(f"Computing embeddings for {len(image_text_pairs)} images...")
    for i in range(0, len(image_text_pairs), batch_size):
        batch = image_text_pairs[i:i + batch_size]
        images = [Image.open(item["image_path"]).convert("RGB") for item in batch]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            embeddings = model.get_image_features(**inputs)
        image_embeddings.append(embeddings.cpu().numpy())
        print(f"Processed {min(i + batch_size, len(image_text_pairs))}/{len(image_text_pairs)} images.", end='\r')
    print("\nImage embedding computation complete.")
    return np.vstack(image_embeddings)

def compute_query_embeddings(model, processor, query_texts, batch_size=32):
    """Compute text embeddings for queries."""
    query_embeddings = []
    print(f"Computing embeddings for {len(query_texts)} queries...")
    for i in range(0, len(query_texts), batch_size):
        batch_texts = query_texts[i:i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        query_embeddings.append(embeddings.cpu().numpy())
        print(f"Processed {min(i + batch_size, len(query_texts))}/{len(query_texts)} queries.", end='\r')
    print("\nQuery embedding computation complete.")
    return np.vstack(query_embeddings)

def build_faiss_index(image_embeddings):
    """Build Faiss index for efficient similarity search."""
    dimension = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension) # Use Inner Product for cosine similarity with L2-normalized vectors
    faiss.normalize_L2(image_embeddings) # Normalize image embeddings
    index.add(image_embeddings)
    return index

def retrieve_images(query_embeddings, image_ids, index, k=100):
    """Retrieve top-k images for each query."""
    faiss.normalize_L2(query_embeddings) # Normalize query embeddings
    distances, indices = index.search(query_embeddings, k)
    results = []
    for idx_list in indices:
        # Map Faiss indices back to actual image_ids
        top_images = []
        for i in idx_list:
            if i < len(image_ids):
                top_images.append(image_ids[i])
            else:
                top_images.append("#") # Fallback if index out of bounds (shouldn't happen with correct indexing)
        top_images += ["#"] * (k - len(top_images)) # Pad with '#' if less than k results
        results.append(top_images[:k]) # Ensure we only take up to k
    return results

def compute_metrics(retrieved_images, query_ids, ground_truth, k=100):
    """Compute evaluation metrics: mAP, MRR, Recall@1, Recall@5, Recall@10."""
    ap_scores = []
    rr_scores = []
    recall_at_1 = []
    recall_at_5 = []
    recall_at_10 = []
    
    for i, query_id in enumerate(query_ids):
        retrieved = retrieved_images[i]
        relevant = ground_truth.get(query_id, [])
        if not relevant:
            # print(f"Warning: No ground truth for query_id {query_id}. Skipping.")
            continue
        
        # Binary relevance for AP
        # Ensure that only unique retrieved images are considered for AP calculation if needed,
        # but for simplicity, we assume 'retrieved' already contains unique, ranked items.
        y_true = [1 if img in relevant and img != "#" else 0 for img in retrieved]
        # For average_precision_score, y_score should reflect confidence/rank.
        # A simple way is to use a decreasing score based on rank (1 for 1st, 0.99 for 2nd etc.)
        # Or, just use the provided dummy scores which are rank-based.
        y_score = [1 - j / k for j in range(k)] # Dummy scores based on rank, higher rank = lower score
        
        if sum(y_true) > 0: # Only compute AP if there's at least one relevant item
            try:
                ap_scores.append(average_precision_score(y_true, y_score))
            except ValueError:
                # This can happen if y_true contains only one class (all 0s or all 1s after filtering)
                # print(f"Could not compute AP for query {query_id} (all y_true are same).")
                pass
        
        # MRR
        found_relevant_in_rr = False
        for rank, img in enumerate(retrieved, 1):
            if img in relevant and img != "#":
                rr_scores.append(1 / rank)
                found_relevant_in_rr = True
                break
        if not found_relevant_in_rr:
            rr_scores.append(0) # No relevant item found
        
        # Recall@k
        retrieved_set = set([img for img in retrieved if img != "#"])
        relevant_set = set(relevant)

        recall_at_1.append(any(img in relevant_set for img in retrieved[:1]))
        recall_at_5.append(any(img in relevant_set for img in retrieved[:5]))
        recall_at_10.append(any(img in relevant_set for img in retrieved[:10])) # Corrected to 10
    
    metrics = {
        "mAP": np.mean(ap_scores) if ap_scores else 0.0,
        "MRR": np.mean(rr_scores) if rr_scores else 0.0,
        "Recall@1": np.mean(recall_at_1) if recall_at_1 else 0.0,
        "Recall@5": np.mean(recall_at_5) if recall_at_5 else 0.0,
        "Recall@10": np.mean(recall_at_10) if recall_at_10 else 0.0
    }
    return metrics

def test_on_small_dataset(model, processor, image_text_pairs, image_ids, query_data, database):
    """Test the retrieval pipeline on a small dataset."""
    print("Creating small dataset...")
    small_pairs, small_image_ids, small_queries, ground_truth = create_small_dataset(
        image_text_pairs, query_data, database, num_images=100, num_queries=10
    )
    
    print("Computing image embeddings for small dataset...")
    image_embeddings = compute_image_embeddings(model, processor, small_pairs)
    
    print("Computing query embeddings for small dataset...")
    query_embeddings = compute_query_embeddings(model, processor, small_queries["query_text"].tolist())
    
    print("Building Faiss index for small dataset...")
    index = build_faiss_index(image_embeddings)
    
    print("Retrieving images for small dataset...")
    retrieved_images = retrieve_images(query_embeddings, small_image_ids, index, k=10) # Using k=10 for small dataset test
    
    print("Computing evaluation metrics...")
    metrics = compute_metrics(retrieved_images, small_queries["query_id"].tolist(), ground_truth, k=10)
    
    print("\nEvaluation Metrics on Small Dataset:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

def generate_submission(query_data, retrieved_images, output_dir):
    """Generate submission CSV."""
    submission_csv_path = os.path.join(output_dir, "submission.csv")
    submission_zip_path = os.path.join(output_dir, "submission.zip")

    submission_data = {
        "query_id": query_data["query_id"],
    }
    # Ensure all columns are present, even if some rows have fewer than 100 images
    for i in range(100):
        submission_data[f"image_id_{i+1}"] = [row[i] if i < len(row) else "#" for row in retrieved_images]

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(submission_csv_path, index=False)
    
    with zipfile.ZipFile(submission_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(submission_csv_path, arcname="submission.csv")
    print(f"Submission saved to {submission_zip_path}")

def main():
    """Main function to execute the image retrieval pipeline."""
    parser = argparse.ArgumentParser(description="Image Retrieval Pipeline for EVENTA Track 2.")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH_DEFAULT,
                        help=f"Path to the EVENTA dataset directory (contains database.json). Default: {DATASET_PATH_DEFAULT}")
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR_DEFAULT,
                        help=f"Path to the directory containing database images. Default: {IMAGE_DIR_DEFAULT}")
    parser.add_argument("--query_csv", type=str, default=QUERY_CSV_DEFAULT,
                        help=f"Path to the CSV file containing queries. Default: {QUERY_CSV_DEFAULT}")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR_DEFAULT,
                        help=f"Directory to save outputs (e.g., submission.csv, submission.zip). Default: {OUTPUT_DIR_DEFAULT}")
    parser.add_argument("--test_small", action="store_true",
                        help="Set this flag to run a test on a small subset of the dataset.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding computation.")
    parser.add_argument("--k_retrieval", type=int, default=100,
                        help="Number of top images to retrieve for each query.")

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    set_seed(RANDOM_SEED)
    
    # Step 1: Load dataset
    print("Loading dataset...")
    image_text_pairs, image_ids, query_data, database = load_dataset(
        args.dataset_path, args.image_dir, args.query_csv
    )
    
    # Step 2: Initialize model
    print("Initializing CLIP model...")
    model, processor = initialize_model()
    
    # Step 3: Fine-tune model (placeholder)
    print("Fine-tuning model...")
    model = fine_tune_model(model, processor, image_text_pairs) # You can add args for fine-tuning params if needed
    
    # Step 4: Test on small dataset (if enabled)
    if args.test_small:
        print("\nRunning test on small dataset...")
        test_on_small_dataset(model, processor, image_text_pairs, image_ids, query_data, database)
    
    # Step 5: Compute embeddings for full dataset
    print(f"\nComputing image embeddings for full dataset (batch_size={args.batch_size})...")
    image_embeddings = compute_image_embeddings(model, processor, image_text_pairs, args.batch_size)
    
    print(f"Computing query embeddings for full dataset (batch_size={args.batch_size})...")
    query_embeddings = compute_query_embeddings(model, processor, query_data["query_text"].tolist(), args.batch_size)
    
    # Step 6: Build Faiss index and retrieve images
    print("Building Faiss index for full dataset...")
    index = build_faiss_index(image_embeddings)
    
    print(f"Retrieving top {args.k_retrieval} images for full dataset...")
    retrieved_images = retrieve_images(query_embeddings, image_ids, index, k=args.k_retrieval)
    
    # Step 7: Generate submission
    print("Generating submission...")
    generate_submission(query_data, retrieved_images, args.output_dir)

if __name__ == "__main__":
    main()