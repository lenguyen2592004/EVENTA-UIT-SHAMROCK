import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile # Import ImageFile để xử lý ảnh hỏng
import pandas as pd
from pathlib import Path
import numpy as np
import os
from tqdm.autonotebook import tqdm
import lightning as L
from collections import Counter, defaultdict
import argparse
import sys # Thêm import sys để in lỗi ra stderr

# Đảm bảo PIL xử lý các ảnh bị cắt/hỏng
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- Định nghĩa các lớp Model (giữ nguyên) ---
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode # Import rõ ràng

# Định nghĩa MEAN/STD chuẩn cho ImageNet
IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]


class ImageEncoder(nn.Module):
    SUPPORTED_MODELS = [
        'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'
    ]
    def __init__(self, output_dim=64, img_model='dinov2_vits14', unfreeze_n_blocks=4):
        super().__init__()
        if img_model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid image model name. Choose between {self.SUPPORTED_MODELS}')
        self.encoder = torch.hub.load('facebookresearch/dinov2', img_model, verbose=False)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Đảm bảo unfreeze_n_blocks không vượt quá số lượng khối thực tế
        num_blocks = len(self.encoder.blocks)
        if unfreeze_n_blocks > num_blocks:
            print(f"Cảnh báo: unfreeze_n_blocks ({unfreeze_n_blocks}) lớn hơn tổng số khối của encoder ({num_blocks}). Sẽ bỏ đóng băng tất cả các khối.", file=sys.stderr)
            unfreeze_n_blocks = num_blocks

        for block in self.encoder.blocks[-unfreeze_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        
        if self.encoder.norm is not None:
            for param in self.encoder.norm.parameters():
                param.requires_grad = True
        
        self.fc = nn.Linear(self.encoder.embed_dim, output_dim)
    def forward(self, x):
        dino_output = self.encoder.forward_features(x)
        x = dino_output['x_norm_clstoken']
        x = self.fc(x)
        return x

class TextEncoder(nn.Module):
    SUPPORTED_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/bert-base-nli-mean-tokens",
    ]
    def __init__(self, output_dim=64, lang_model="sentence-transformers/all-MiniLM-L6-v2", unfreeze_n_blocks=4):
        super().__init__()
        if lang_model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid text model name. Choose between {self.SUPPORTED_MODELS}')

        self.lang_model = lang_model
        self.encoder = AutoModel.from_pretrained(lang_model)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        num_layers = len(self.encoder.encoder.layer)
        if unfreeze_n_blocks > num_layers:
            print(f"Cảnh báo: unfreeze_n_blocks ({unfreeze_n_blocks}) lớn hơn tổng số lớp encoder ({num_layers}). Sẽ bỏ đóng băng tất cả các lớp encoder.", file=sys.stderr)
            unfreeze_n_blocks = num_layers

        for layer in self.encoder.encoder.layer[-unfreeze_n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        if self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True
        
        self.fc = nn.Linear(self.encoder.config.hidden_size, output_dim)
    def forward(self, input_ids, attention_mask=None):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.fc(x)
        return x

class NanoCLIP(L.LightningModule):
    def __init__(
        self,
        txt_model="sentence-transformers/all-MiniLM-L6-v2",
        img_model='dinov2_vits14',
        embed_size=64,
        unfreeze_n_blocks=4,
        # Thêm các tham số khác nếu cần để load_from_checkpoint không bị lỗi
        lr=0.0001,
        warmup_epochs=0,
        weight_decay=0.0001,
        milestones=[5, 10, 15],
        lr_mult=0.1,
        **kwargs
    ):
        super().__init__()
        # Lưu các hyperparameters để load_from_checkpoint có thể khởi tạo lại mô hình
        self.save_hyperparameters() 
        
        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.img_encoder = ImageEncoder(self.embed_size, self.img_model, self.unfreeze_n_blocks)
        self.txt_encoder = TextEncoder(self.embed_size, self.txt_model, self.unfreeze_n_blocks)
    
    # Cần định nghĩa configure_optimizers để load_from_checkpoint hoạt động, dù không dùng ở đây
    def configure_optimizers(self):
        return None 
        
    def forward(self, image, captions_input_ids, captions_attention_mask):
        image_embedding = self.img_encoder(image)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        text_embedding = self.txt_encoder(captions_input_ids, captions_attention_mask)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        return image_embedding, text_embedding

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            print(f"Cảnh báo: File ảnh không tồn tại: {image_path}. Bỏ qua mẫu này.", file=sys.stderr)
            return None # Trả về None nếu ảnh không tồn tại
            
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image)
            return image
        except (IOError, OSError) as e:
            print(f"Cảnh báo: Lỗi khi đọc ảnh {image_path}: {e}. Bỏ qua mẫu này.", file=sys.stderr)
            return None # Trả về None nếu có lỗi khi đọc ảnh


# --- Hàm chính cho Inference ---
def main():
    parser = argparse.ArgumentParser(description="Thực hiện Image Retrieval sử dụng Ensemble NanoCLIP models.")
    
    # Paths
    parser.add_argument("--image_dir", type=str, 
                        default='/kaggle/input/database-eventa/database_images_compressed90_scaled05' if os.path.exists('/kaggle/input') else './database_images',
                        help="Đường dẫn đến thư mục chứa tất cả các ảnh cơ sở dữ liệu.")
    parser.add_argument("--test_csv_path", type=str, 
                        default='/kaggle/input/eventa-track-2/query_public.csv' if os.path.exists('/kaggle/input') else './query_public.csv',
                        help="Đường dẫn đến file CSV chứa các query test.")
    parser.add_argument("--submission_csv_path", type=str, 
                        default='submission.csv',
                        help="Đường dẫn file CSV đầu ra cho submission.")
    parser.add_argument("--model_paths", nargs='+', type=str, 
                        default=[ # Đường dẫn mặc định cho Kaggle
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold0/version_0/checkpoints/fold0_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold1/version_0/checkpoints/fold1_epoch=07_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold2/version_0/checkpoints/fold2_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold3/version_0/checkpoints/fold3_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold4/version_0/checkpoints/fold4_epoch=07_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold5/version_0/checkpoints/fold5_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold6/version_0/checkpoints/fold6_epoch=07_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold7/version_0/checkpoints/fold7_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold8/version_0/checkpoints/fold8_epoch=09_recall@5.ckpt",
                            "/kaggle/input/t2i-eventabi-encoder/logs/nano_clip_fold9/version_0/checkpoints/fold9_epoch=07_recall@5.ckpt"
                        ],
                        help="Danh sách đường dẫn đến các checkpoint mô hình NanoCLIP (cách nhau bằng dấu cách).")
    
    # Inference Parameters
    parser.add_argument("--k_candidates", type=int, default=15,
                        help="Số lượng ứng viên ảnh hàng đầu từ mỗi mô hình để sử dụng cho chiến lược bỏ phiếu ensemble.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Kích thước batch để tính toán embedding ảnh.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Số lượng workers cho DataLoader ảnh.")
    
    args = parser.parse_args()

    # Thiết lập thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # 1. Tải models
    print(f"Đang tải {len(args.model_paths)} models...")
    models = []
    for path in args.model_paths:
        try:
            model = NanoCLIP.load_from_checkpoint(path, map_location=device)
            model.eval()
            models.append(model)
        except Exception as e:
            print(f"Lỗi khi tải mô hình từ '{path}': {e}. Bỏ qua mô hình này.", file=sys.stderr)
    
    if not models:
        print("Lỗi: Không có mô hình nào được tải thành công. Không thể tiếp tục.", file=sys.stderr)
        sys.exit(1)

    # Lấy tokenizer từ mô hình đầu tiên (giả định tất cả các mô hình dùng cùng txt_model)
    tokenizer = AutoTokenizer.from_pretrained(models[0].hparams.txt_model)

    # 2. Chuẩn bị transform cho ảnh
    valid_transform = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

    # 3. Chuẩn bị dữ liệu ảnh
    print(f"Đang quét ảnh trong thư mục: {args.image_dir}")
    try:
        image_filenames = sorted(os.listdir(args.image_dir))
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục ảnh '{args.image_dir}'. Vui lòng kiểm tra lại đường dẫn.", file=sys.stderr)
        sys.exit(1)

    image_paths = [os.path.join(args.image_dir, fname) for fname in image_filenames]
    num_images = len(image_paths)
    if num_images == 0:
        print(f"Cảnh báo: Không tìm thấy ảnh nào trong thư mục '{args.image_dir}'.", file=sys.stderr)
        # Tiếp tục để tạo submission rỗng hoặc thoát
        sys.exit(1)

    print(f"Tìm thấy {num_images} ảnh trong cơ sở dữ liệu.")

    # 4. Tính toán trước embedding cho ảnh
    all_image_embeddings = [] # Sẽ lưu (số lượng models, số lượng ảnh, embed_size)
    image_dataset = ImageDataset(image_paths, valid_transform)
    # Collate function để lọc các mẫu None (ảnh lỗi)
    def collate_fn_image_only(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None # Trả về None nếu batch rỗng
        return torch.stack(batch)

    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=args.num_workers, pin_memory=True,
                              collate_fn=collate_fn_image_only)

    print("Đang tính toán trước embedding ảnh cho tất cả các mô hình...")
    with torch.no_grad():
        for i, model in enumerate(tqdm(models, desc="Đang xử lý mô hình")):
            model_img_embeds = []
            for image_batch in image_loader:
                if image_batch is None: # Xử lý batch rỗng
                    continue
                image_batch = image_batch.to(device)
                img_embed_batch = model.img_encoder(image_batch)
                img_embed_batch = F.normalize(img_embed_batch, p=2, dim=-1)
                model_img_embeds.append(img_embed_batch.cpu())
            
            if model_img_embeds: # Chỉ thêm nếu có embedding được tạo
                all_image_embeddings.append(torch.cat(model_img_embeds, dim=0).to(device))
            else:
                print(f"Cảnh báo: Không có embedding ảnh nào được tạo cho mô hình {i}. Bỏ qua mô hình này trong ensemble.", file=sys.stderr)
                models[i] = None # Đánh dấu mô hình này là None để bỏ qua sau
    
    # Lọc bỏ các mô hình không có embedding (do lỗi hoặc không có ảnh)
    models = [m for m in models if m is not None]
    if not models:
        print("Lỗi: Không có mô hình nào và/hoặc embedding ảnh hợp lệ để thực hiện ensemble. Thoát.", file=sys.stderr)
        sys.exit(1)


    # 5. Duyệt qua các query, tìm kiếm và tạo submission
    print(f"\nĐang đọc file query: {args.test_csv_path}")
    try:
        test_df = pd.read_csv(args.test_csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file CSV test '{args.test_csv_path}'.", file=sys.stderr)
        sys.exit(1)
    
    submission_rows = []

    print(f"Sử dụng chiến lược bỏ phiếu Ensemble Top-{args.k_candidates}.")
    print("Đang xử lý các query và tìm top 10 ảnh...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Đang xử lý Query"):
            query_id = row['query_index']
            query_text = row['query_text']
            
            tokenized = tokenizer(
                query_text, padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            # --- LOGIC ENSEMBLE ---
            candidate_votes = Counter()
            candidate_scores = defaultdict(float) # Tổng điểm tương đồng

            for i, model in enumerate(models):
                # Nếu mô hình này đã bị đánh dấu là None, bỏ qua
                if model is None:
                    continue

                text_embed = model.txt_encoder(input_ids, attention_mask)
                text_embed = F.normalize(text_embed, p=2, dim=-1)
                
                img_embeds_for_model = all_image_embeddings[i] # Lấy embedding ảnh tương ứng với mô hình
                
                scores = (text_embed @ img_embeds_for_model.T).squeeze()
                
                top_k_scores, top_k_indices = torch.topk(scores, min(args.k_candidates, len(scores)))
                
                top_k_scores = top_k_scores.cpu().numpy()
                top_k_indices = top_k_indices.cpu().numpy()

                # Ghi nhận phiếu bầu và điểm số cho các ứng viên
                for j, idx_in_db in enumerate(top_k_indices):
                    # image_filenames đã được sắp xếp giống với thứ tự embedding
                    image_id = image_filenames[idx_in_db].split('.')[0]
                    candidate_votes[image_id] += 1
                    candidate_scores[image_id] += top_k_scores[j] # Cộng dồn điểm số

            # Xếp hạng các ứng viên dựa trên phiếu bầu (ưu tiên cao hơn)
            # Sau đó là tổng điểm tương đồng (ưu tiên cao hơn)
            all_candidates = list(candidate_votes.keys())
            sorted_candidates = sorted(
                all_candidates,
                key=lambda img_id: (candidate_votes[img_id], candidate_scores[img_id]),
                reverse=True
            )
            
            # Lấy top 10 cuối cùng
            top_10_image_ids = sorted_candidates[:10]
            
            # Đảm bảo có đủ 10 ID ảnh, điền '#N/A' nếu thiếu (mặc dù ít khi xảy ra)
            top_10_image_ids += ['#N/A'] * (10 - len(top_10_image_ids))

            # Tạo dòng cho submission
            submission_row = [query_id] + top_10_image_ids
            submission_rows.append(submission_row)

    # 6. Lưu file submission
    submission_df = pd.DataFrame(
        submission_rows, 
        columns=['query_id'] + [f'image_id_{i}' for i in range(1, 11)]
    )

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(args.submission_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(args.submission_csv_path, index=False)

    print(f"File submission đã được tạo thành công tại: {args.submission_csv_path}")

if __name__ == "__main__":
    main()