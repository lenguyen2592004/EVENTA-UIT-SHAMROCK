import os
import subprocess
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile # Thêm ImageFile để xử lý ảnh hỏng
import faiss
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode # Import rõ ràng
from transformers import AutoTokenizer, AutoModel
from open_clip import create_model_from_pretrained, get_tokenizer # Giữ lại nếu bạn thực sự dùng open_clip
from timm import create_model # timm.create_model
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
import argparse
import sys

# --- Cấu hình chung ---
# Đảm bảo PIL xử lý các ảnh bị cắt/hỏng
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Xác định nếu đang chạy trong môi trường Kaggle để thiết lập đường dẫn mặc định
IS_KAGGLE = os.path.exists('/kaggle/input')

# Định nghĩa các hằng số MEAN/STD cho chuẩn hóa ảnh (như trong mã gốc)
# IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # Dùng cho BEiT3, không phải CLIP/DINO
# IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5) # Dùng cho BEiT3, không phải CLIP/DINO

# MEAN/STD chuẩn cho ImageNet thường được dùng với các mô hình pre-trained
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# --- Hàm thiết lập môi trường ---
def setup_environment(
    base_dir,
    install_open_clip=True,
    install_faiss_gpu=False,
    skip_pip_install=False
):
    """
    Cài đặt thư viện.
    """
    print("--- Bắt đầu thiết lập môi trường ---")
    
    original_cwd = os.getcwd() # Lưu thư mục làm việc ban đầu
    os.chdir(base_dir) # Chuyển đến thư mục gốc để đảm bảo các lệnh pip hoạt động đúng

    try:
        if not skip_pip_install:
            print("Đang cài đặt các gói cần thiết...")
            # Sử dụng --break-system-packages cho cảnh báo môi trường ảo trên một số hệ thống
            pip_install_cmd = ['pip', 'install', '-q', '--break-system-packages']

            if install_open_clip:
                # open_clip_torch==2.23.0 transformers==4.35.2 matplotlib
                subprocess.run(pip_install_cmd + ['open_clip_torch==2.23.0', 'transformers==4.35.2', 'matplotlib'], check=True, capture_output=True, text=True)
            else:
                # Nếu không cài open_clip, vẫn cần transformers và matplotlib
                subprocess.run(pip_install_cmd + ['transformers==4.35.2', 'matplotlib'], check=True, capture_output=True, text=True)

            if install_faiss_gpu:
                print("Đang cài đặt faiss-gpu...")
                subprocess.run(pip_install_cmd + ['faiss-gpu'], check=True, capture_output=True, text=True)
            else:
                print("Đang cài đặt faiss-cpu...")
                subprocess.run(pip_install_cmd + ['faiss-cpu'], check=True, capture_output=True, text=True)
            
            # Cài đặt lightning riêng vì có thể có phiên bản cụ thể
            subprocess.run(pip_install_cmd + ['lightning'], check=True, capture_output=True, text=True)

        else:
            print("Bỏ qua cài đặt pip.")
            
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình thiết lập môi trường (Lệnh '{' '.join(e.cmd)}' thất bại): {e.stderr}", file=sys.stderr)
        os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu trước khi thoát
        sys.exit(1)
    except Exception as e:
        print(f"Một lỗi không mong muốn xảy ra trong quá trình thiết lập: {e}", file=sys.stderr)
        os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu trước khi thoát
        sys.exit(1)
    
    print("--- Thiết lập môi trường hoàn tất ---")
    os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu

# --- Dataset và Collate ---
class ImageTextDataset(Dataset):
    """
    Dataset hỗ trợ train/val/test dựa vào flag split.
    CSV chứa 2 cột: image_path và text (có thể thiếu ở test).
    """
    def __init__(self, csv_file, img_root_dir=None, split='train', img_transform=None, indices=None):
        self.data = pd.read_csv(csv_file)
        self.img_root_dir = Path(img_root_dir) if img_root_dir else None
        self.img_transform = img_transform
        self.split = split.lower()
        assert self.split in ['train', 'val', 'test'], "split phải là 'train', 'val', hoặc 'test'"
        
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        # Kiểm tra và chuyển đổi cột 'image_path' nếu cần thiết (ví dụ: đường dẫn tương đối)
        if self.img_root_dir and not self.data['image_path'].iloc[0].startswith(str(self.img_root_dir)):
            self.data['image_path'] = self.data['image_path'].apply(lambda x: str(self.img_root_dir / x))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        
        # Đảm bảo path là string
        if not isinstance(img_path, str):
            img_path = str(img_path)

        # Kiểm tra file ảnh có tồn tại không
        if not os.path.exists(img_path):
            print(f"Cảnh báo: File ảnh không tồn tại: {img_path}. Bỏ qua mẫu này.", file=sys.stderr)
            # Trả về một mẫu rỗng hoặc mẫu dự phòng để tránh lỗi
            # Tùy chọn: có thể raise lỗi hoặc skip mẫu này bằng cách trả về None và lọc ở DataLoader
            return self.__getitem__((idx + 1) % len(self)) # Thử mẫu tiếp theo nếu bị lỗi

        image = Image.open(img_path).convert('RGB')
        if self.img_transform:
            image = self.img_transform(image)

        caption = self.data.iloc[idx]['text'] if 'text' in self.data.columns else None
        if pd.isna(caption) or self.split == 'test':
            caption = "" # Sử dụng chuỗi rỗng thay vì None cho tokenizer

        return image, caption


class CollateImageText:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        # Lọc các mẫu None nếu hàm __getitem__ có thể trả về None
        batch = [item for item in batch if item is not None]
        if not batch: # Nếu batch rỗng sau khi lọc
            return None 

        images, captions = zip(*batch)
        images = torch.stack(images)

        # Trả về chỉ ảnh nếu không có caption hoặc ở chế độ test
        if captions[0] is None or (len(captions) > 0 and captions[0] == ""): # Kiểm tra chuỗi rỗng
            return (images,)

        encoding = self.tokenizer(
            list(captions),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )

        return images, encoding['input_ids'], encoding['attention_mask']

# --- Loss Function ---
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, penalty_weight=1.0):
        super().__init__()
        self.temperature = temperature
        self.penalty_weight = penalty_weight

    def forward(self, image_embedding, text_embedding, topk=5):
        batch_size = image_embedding.shape[0]
        labels = torch.arange(batch_size, device=image_embedding.device)
    
        logits = torch.matmul(image_embedding, text_embedding.T) / self.temperature
        
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        total_loss = (loss_i2t + loss_t2i) / 2
    
        ranks_i2t = (logits > logits[range(batch_size), labels].unsqueeze(1)).sum(dim=1)
        penalty_i2t = (ranks_i2t >= topk).float().mean()
    
        ranks_t2i = (logits.T > logits.T[range(batch_size), labels].unsqueeze(1)).sum(dim=1)
        penalty_t2i = (ranks_t2i >= topk).float().mean()
    
        rank_penalty = (penalty_i2t + penalty_t2i) / 2
    
        total_loss = total_loss + rank_penalty * self.penalty_weight
    
        acc_t2i = (torch.argmax(logits, dim=0) == labels).float().mean()
    
        return total_loss, acc_t2i

# --- Encoders ---
class ImageEncoder(nn.Module):
    # DINOv2 models: "facebookresearch/dinov2:main"
    SUPPORTED_MODELS = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        # 'dinov2_vitg14'
    ]
    def __init__(self, output_dim=64, img_model='dinov2_vits14', unfreeze_n_blocks=4):
        super().__init__()
        if img_model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid image model name. Choose between {self.SUPPORTED_MODELS}')
        
        # tải dinov2 từ torch.hub
        self.encoder = torch.hub.load('facebookresearch/dinov2', img_model)
        
        # Đóng băng tất cả các tham số
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Bỏ đóng băng các khối cuối cùng
        # Kiểm tra để đảm bảo đủ số lượng khối
        if unfreeze_n_blocks > len(self.encoder.blocks):
            print(f"Cảnh báo: unfreeze_n_blocks ({unfreeze_n_blocks}) lớn hơn tổng số khối của encoder ({len(self.encoder.blocks)}). Sẽ bỏ đóng băng tất cả các khối.", file=sys.stderr)
            unfreeze_n_blocks = len(self.encoder.blocks) # Bỏ đóng băng tất cả

        for block in self.encoder.blocks[-unfreeze_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # Bỏ đóng băng lớp norm
        if self.encoder.norm is not None:
            for param in self.encoder.norm.parameters():
                param.requires_grad = True

        self.fc = nn.Linear(self.encoder.embed_dim, output_dim)
        
    def forward(self, x):
        dino_output = self.encoder.forward_features(x)
        x = dino_output['x_norm_clstoken'] # sử dụng cls token
        x = self.fc(x)
        return x
    

class TextEncoder(nn.Module):
    # sentence-transformers models
    SUPPORTED_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/bert-base-nli-mean-tokens",
        # Thêm các mô hình khác nếu cần
    ]
    def __init__(self, output_dim=64, lang_model="sentence-transformers/all-MiniLM-L6-v2", unfreeze_n_blocks=4):
        super().__init__()
        if lang_model not in self.SUPPORTED_MODELS:
            raise ValueError(f'Invalid text model name. Choose between {self.SUPPORTED_MODELS}')

        self.lang_model = lang_model
        self.encoder = AutoModel.from_pretrained(lang_model)
        
        # Đóng băng tất cả các tham số
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Bỏ đóng băng các lớp encoder cuối cùng
        if unfreeze_n_blocks > len(self.encoder.encoder.layer):
            print(f"Cảnh báo: unfreeze_n_blocks ({unfreeze_n_blocks}) lớn hơn tổng số lớp encoder ({len(self.encoder.encoder.layer)}). Sẽ bỏ đóng băng tất cả các lớp encoder.", file=sys.stderr)
            unfreeze_n_blocks = len(self.encoder.encoder.layer) # Bỏ đóng băng tất cả

        for layer in self.encoder.encoder.layer[-unfreeze_n_blocks:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Bỏ đóng băng lớp pooler
        if self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True
        
        self.fc = nn.Linear(self.encoder.config.hidden_size, output_dim)
    
    def forward(self, input_ids, attention_mask=None):
        # Lấy [CLS] token embedding
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.fc(x)
        return x

# --- Lightning Module ---
class NanoCLIP(L.LightningModule):
    def __init__(
        self,
        txt_model="sentence-transformers/all-MiniLM-L6-v2",
        img_model='dinov2_vits14',
        embed_size=64,
        unfreeze_n_blocks=4,
        lr=0.0001,
        warmup_epochs=0,
        weight_decay=0.0001,
        milestones=[5, 10, 15],
        lr_mult=0.1,
    ):
        super().__init__()
        
        self.txt_model = txt_model
        self.img_model = img_model
        self.embed_size = embed_size
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.lr_mult = lr_mult
        
        self.save_hyperparameters()
        
        self.img_encoder = ImageEncoder(self.embed_size, self.img_model, unfreeze_n_blocks)
        self.txt_encoder = TextEncoder(self.embed_size, self.txt_model, unfreeze_n_blocks)
        self.loss_fn = ContrastiveLoss(temperature=0.05)

    def configure_optimizers(self):
        optimizer_params = [
            {"params": self.img_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": self.txt_encoder.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.milestones, gamma=self.lr_mult
        )    
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch < self.warmup_epochs:
            total_warmup_steps = self.warmup_epochs * len(self.trainer.train_dataloader)
            lr_scale = min(1.0, (self.trainer.global_step + 1) / total_warmup_steps)
            for pg in optimizer.param_groups:
                initial_lr = pg.get("initial_lr", self.lr)
                pg["lr"] = lr_scale * initial_lr

        optimizer.step(closure=optimizer_closure)
        self.log('_LR', optimizer.param_groups[-1]['lr'], prog_bar=False, logger=True)
    
    def forward(self, image, captions_input_ids, captions_attention_mask):
        image_embedding = self.img_encoder(image)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)
        
        text_embedding = self.txt_encoder(captions_input_ids, captions_attention_mask)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        
        return image_embedding, text_embedding
    
    def training_step(self, batch, batch_idx):
        if batch is None: # Xử lý batch rỗng do ảnh hỏng
            return None
        images, captions_input_ids, captions_attention_mask = batch
        
        loss, batch_accuracy = self(images, captions_input_ids, captions_attention_mask)
        
        self.log("loss", loss, prog_bar=True, logger=True)
        self.log("batch_acc", batch_accuracy, prog_bar=True, logger=True)
        return loss
    
    def on_validation_epoch_start(self):
        self.validation_descriptors = {"img": [], "txt": []}
        
    def validation_step(self, batch, batch_idx):
        if batch is None: # Xử lý batch rỗng
            return
        images, captions_input_ids, captions_attention_mask = batch
        
        img_descriptors, txt_descriptors = self(images, captions_input_ids, captions_attention_mask)
        img_descriptors = img_descriptors.detach().cpu().numpy()
        txt_descriptors = txt_descriptors.detach().cpu().numpy()
        
        self.validation_descriptors["img"].append(img_descriptors)
        self.validation_descriptors["txt"].append(txt_descriptors)
    
    def on_validation_epoch_end(self):
        if not self.validation_descriptors["img"] or not self.validation_descriptors["txt"]:
            print("Cảnh báo: Không có descriptor để tính toán recall/mrr trên tập validation. Có thể do tất cả các batch đều rỗng hoặc lỗi.", file=sys.stderr)
            self.log("recall@1", 0.0, prog_bar=True, logger=True)
            self.log("recall@5", 0.0, prog_bar=True, logger=True)
            self.log("recall@10", 0.0, prog_bar=True, logger=True)
            self.log("mrr", 0.0, prog_bar=True, logger=True)
            self.validation_descriptors.clear()
            return

        img_descriptors = np.concatenate(self.validation_descriptors["img"], axis=0)
        txt_descriptors = np.concatenate(self.validation_descriptors["txt"], axis=0)
        
        B = img_descriptors.shape[0]    
        labels = np.arange(B)

        recall_1, recall_5, recall_10 = self._calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10])
        self.log("recall@1", recall_1, prog_bar=True, logger=True)
        self.log("recall@5", recall_5, prog_bar=True, logger=True)
        self.log("recall@10", recall_10, prog_bar=True, logger=True)

        mrr = self._calculate_mrr(img_descriptors, txt_descriptors, labels)
        self.log("mrr", mrr, prog_bar=True, logger=True)

        self.validation_descriptors.clear()
    
    @staticmethod
    def _calculate_recall(img_descriptors, txt_descriptors, labels, k_values=[1, 5, 10]):
        embed_size = img_descriptors.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size) # Sử dụng L2 cho các embedding đã chuẩn hóa
        
        faiss_index.add(img_descriptors)
        _, predictions = faiss_index.search(txt_descriptors, max(k_values))
        
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                if np.any(np.in1d(pred[:n], labels[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k /= len(labels)
                
        return correct_at_k

    @staticmethod
    def _calculate_mrr(img_descriptors, txt_descriptors, labels):
        faiss_index = faiss.IndexFlatL2(img_descriptors.shape[1])
        faiss_index.add(img_descriptors)
        _, predictions = faiss_index.search(txt_descriptors, k=10) # Max k=10 for MRR
        mrr = 0
        for i, pred in enumerate(predictions):
            rank = np.where(pred == labels[i])[0]
            if len(rank) > 0:
                mrr += 1 / (rank[0] + 1)
        return mrr / len(labels)

# --- Image Transforms ---
def get_transforms():
    train_transform = T.Compose([
        T.Resize((256, 256), interpolation=InterpolationMode.BICUBIC),
        T.RandomResizedCrop((224, 224), scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        T.RandomRotation(degrees=10),
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        T.RandomErasing(p=0.3, scale=(0.02, 0.08), ratio=(0.3, 3.3), value='random'),
    ])

    valid_transform = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return train_transform, valid_transform

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình NanoCLIP với cross-validation.")
    
    # Environment Setup Arguments
    parser.add_argument("--skip_setup", action="store_true",
                        help="Bỏ qua việc cài đặt các gói python. Giả định môi trường đã được thiết lập.")
    parser.add_argument("--install_faiss_gpu", action="store_true",
                        help="Cài đặt faiss-gpu thay vì faiss-cpu.")

    # Data Paths
    parser.add_argument("--train_csv_path", type=str, 
                        default='/kaggle/input/track3-t2i-not-val/train.csv' if IS_KAGGLE else './train.csv',
                        help="Đường dẫn đến file CSV train.")
    parser.add_argument("--image_root_dir", type=str, 
                        default='/kaggle/input/train-eventa/train_compressed_scaled_images/train_images_compressed90_scaled05' if IS_KAGGLE else './train_images_compressed90_scaled05',
                        help="Đường dẫn gốc đến thư mục chứa ảnh train.")
    parser.add_argument("--log_dir", type=str, 
                        default="./logs",
                        help="Thư mục để lưu TensorBoard logs.")
    parser.add_argument("--checkpoint_dir", type=str,
                        default="./checkpoints",
                        help="Thư mục để lưu model checkpoints.")

    # Training Parameters
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Seed ngẫu nhiên cho tính tái lập.")
    parser.add_argument("--num_folds", type=int, default=10,
                        help="Số lượng fold cho K-Fold cross-validation.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Kích thước batch cho DataLoader.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Số lượng workers cho DataLoader.")
    parser.add_argument("--max_epochs", type=int, default=10,
                        help="Số lượng epoch tối đa để huấn luyện.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2,
                        help="Kiểm tra validation sau mỗi N epoch.")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate ban đầu.")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay cho optimizer.")
    parser.add_argument("--warmup_epochs", type=int, default=2,
                        help="Số lượng epoch warmup.")
    parser.add_argument("--lr_milestones", type=int, nargs='+', default=[10, 20, 30],
                        help="Các epoch để giảm LR (ví dụ: 10 20 30).")
    parser.add_argument("--lr_mult", type=float, default=0.1,
                        help="Hệ số nhân LR tại các milestone.")
    parser.add_argument("--embed_size", type=int, default=64,
                        help="Kích thước của embedding đầu ra từ các encoder.")
    parser.add_argument("--unfreeze_n_blocks", type=int, default=4,
                        help="Số lượng khối cuối cùng để bỏ đóng băng trong các encoder.")
    parser.add_argument("--img_model", type=str, default='dinov2_vits14',
                        choices=ImageEncoder.SUPPORTED_MODELS,
                        help="Tên mô hình ảnh DINOv2 để sử dụng.")
    parser.add_argument("--txt_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        choices=TextEncoder.SUPPORTED_MODELS,
                        help="Tên mô hình văn bản Sentence-Transformers để sử dụng.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Số lượng epoch chờ đợi EarlyStopping nếu recall@5 không cải thiện.")
    parser.add_argument("--enable_rich_progress_bar", action="store_true",
                        help="Bật RichProgressBar thay vì thanh tiến trình mặc định.")

    args = parser.parse_args()

    # Thiết lập môi trường
    setup_environment(
        base_dir=os.getcwd(), # Hàm setup_environment sẽ xử lý thay đổi CWD bên trong
        install_open_clip=True, # Đã cài đặt open_clip_torch
        install_faiss_gpu=args.install_faiss_gpu,
        skip_pip_install=args.skip_setup
    )

    seed_everything(args.random_seed, workers=True) # Seed cho tính tái lập

    # Tải tokenizer một lần
    tokenizer = AutoTokenizer.from_pretrained(args.txt_model)

    train_transform, valid_transform = get_transforms()

    dataset = ImageTextDataset(args.train_csv_path, img_root_dir=args.image_root_dir, split='train')
    
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_seed)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n--- Fold {fold+1}/{args.num_folds} is starting ---")
        
        train_dataset = ImageTextDataset(args.train_csv_path, img_root_dir=args.image_root_dir, 
                                         split='train', img_transform=train_transform, indices=train_idx)
        val_dataset = ImageTextDataset(args.train_csv_path, img_root_dir=args.image_root_dir, 
                                       split='val', img_transform=valid_transform, indices=val_idx)

        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=args.num_workers, 
            pin_memory=True, 
            collate_fn=CollateImageText(tokenizer)
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers, 
            pin_memory=True,
            collate_fn=CollateImageText(tokenizer)
        )

        tensorboard_logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=f"nano_clip_fold{fold}",
            default_hp_metric=False
        )

        checkpoint_cb = ModelCheckpoint(
            dirpath=args.checkpoint_dir, # Thư mục lưu checkpoint
            monitor="recall@5",
            filename=f"fold{fold+1}_epoch{{epoch:02d}}_recall@5",
            auto_insert_metric_name=False,
            save_weights_only=True, # Chỉ lưu trọng số mô hình
            save_top_k=1,
            mode="max",
        )
        
        early_stopping_cb = EarlyStopping(
            monitor="recall@5",
            patience=args.patience,
            mode="max",
            verbose=False, # Đặt thành True nếu muốn thông báo dừng sớm
        )

        model = NanoCLIP(
            txt_model=args.txt_model,
            img_model=args.img_model,
            unfreeze_n_blocks=args.unfreeze_n_blocks,
            embed_size=args.embed_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_epochs=args.warmup_epochs,
            milestones=args.lr_milestones,
            lr_mult=args.lr_mult,
        )
        
        callbacks = [checkpoint_cb, early_stopping_cb]
        if args.enable_rich_progress_bar:
            callbacks.append(RichProgressBar())

        trainer = Trainer(
            accelerator="auto",
            devices="auto",
            logger=tensorboard_logger,
            precision="16-mixed",
            max_epochs=args.max_epochs,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            callbacks=callbacks,
            log_every_n_steps=10,
            fast_dev_run=False,
            enable_model_summary=True,
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        best_score = checkpoint_cb.best_model_score
        if best_score is not None:
            fold_scores.append(best_score.item())
            print(f"[Fold {fold+1}] Best Recall@5: {best_score.item():.4f}")
        else:
            print(f"[Fold {fold+1}] Không tìm thấy điểm Recall@5 tốt nhất (có thể do lỗi validation).")

    if fold_scores:
        print(f"\nAverage Recall@5 across {len(fold_scores)} folds: {np.mean(fold_scores):.4f}")
    else:
        print("\nKhông có điểm Recall@5 nào được ghi nhận cho các fold.")

if __name__ == "__main__":
    main()