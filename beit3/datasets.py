import os
import subprocess
import numpy as np
import pandas as pd
import glob
import torch
import faiss
from tqdm.auto import tqdm
from PIL import Image, ImageFile
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from transformers import XLMRobertaTokenizer
from timm import create_model # timm.create_model
import modeling_finetune # modeling_finetune.py, assuming it's in unilm/beit3
import argparse
import sys

# Đảm bảo PIL xử lý các ảnh bị cắt/hỏng
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Định nghĩa các hằng số MEAN/STD cho chuẩn hóa ảnh (như trong mã gốc)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)

# --- Các hằng số cấu hình mặc định ---
# Kiểm tra nếu đang chạy trong môi trường Kaggle để thiết lập đường dẫn mặc định
IS_KAGGLE = os.path.exists('/kaggle/input')

# Đường dẫn mặc định cho mô hình/tokenizer SAU KHI thay đổi thư mục làm việc sang 'unilm/beit3'
DEFAULT_MODEL_CHECKPOINT_LOCAL = './beit3_large_patch16_384_coco_retrieval.pth'
DEFAULT_TOKENIZER_MODEL_LOCAL = './beit3.spm'

# Thư mục ảnh mặc định để tạo embedding
DEFAULT_IMAGE_FOLDER = '/kaggle/input/database-eventa/database_images_compressed90_scaled05' if IS_KAGGLE else './database_images_compressed90_scaled05'

# Thư mục đầu ra mặc định cho các mảnh index FAISS (tương đối với thư mục chạy script ban đầu)
DEFAULT_OUTPUT_INDEX_DIR = './faiss_index_shards'

# Tệp đầu ra metadata mặc định (tương đối với thư mục chạy script ban đầu)
DEFAULT_METADATA_OUTPUT_PATH = './image_paths_processed.npy'

# Biến toàn cục để lưu trữ các ảnh bị hỏng
corrupted_images_list = []

# =================================================================
# === PHẦN 1: CÀI ĐẶT VÀ THIẾT LẬP MÔI TRƯỜNG ===
# =================================================================

def setup_environment(
    base_dir, # Thư mục nơi 'unilm' nên được clone/đã được đặt
    skip_git_clone=False,
    skip_pip_install=False,
    skip_wget_models=False,
    install_faiss_gpu=False
):
    """
    Tải mã nguồn, cài đặt thư viện và tải mô hình.
    Thay đổi thư mục làm việc hiện tại sang 'unilm/beit3' trong quá trình thiết lập,
    sau đó trở về thư mục làm việc ban đầu.
    """
    print("--- Bắt đầu thiết lập môi trường ---")
    
    original_cwd = os.getcwd() # Lưu thư mục làm việc ban đầu
    os.chdir(base_dir) # Chuyển đến thư mục gốc để clone 'unilm'

    try:
        # Clone unilm repository nếu chưa bỏ qua và chưa tồn tại
        if not skip_git_clone:
            if not os.path.exists('unilm'):
                print("Đang clone kho lưu trữ unilm...")
                subprocess.run(['git', 'clone', 'https://github.com/fonzi22/unilm.git'], check=True, capture_output=True, text=True)
            else:
                print("Kho lưu trữ 'unilm' đã tồn tại. Bỏ qua việc clone.")
        
        # Thay đổi thư mục làm việc hiện tại sang 'unilm/beit3' để cài đặt và tải mô hình
        beit3_dir = os.path.join(base_dir, 'unilm', 'beit3')
        if not os.path.exists(beit3_dir):
            print(f"Lỗi: Thư mục '{beit3_dir}' không tìm thấy. Đảm bảo 'unilm' đã được clone và có thư mục 'beit3' bên trong.", file=sys.stderr)
            sys.exit(1)
        os.chdir(beit3_dir)
        print(f"Đã thay đổi thư mục làm việc hiện tại sang: {os.getcwd()}")

        # Cài đặt các gói cần thiết
        if not skip_pip_install:
            print("Đang cài đặt các gói cần thiết...")
            try:
                # Sử dụng --break-system-packages cho cảnh báo môi trường ảo trên một số hệ thống
                subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q', '--break-system-packages'], check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"Cảnh báo: Lỗi khi cài đặt từ requirements.txt. Thử cài đặt thủ công. Error: {e.stderr}", file=sys.stderr)
                # Dự phòng nếu requirements.txt thất bại
                subprocess.run(['pip', 'install', 'timm', 'transformers', 'tokenizers', 'fvcore', 'omegaconf', 'einops', '-q', '--break-system-packages'], check=True, capture_output=True, text=True)
            except Exception as e:
                print(f"Lỗi không mong muốn khi cài đặt requirements: {e}", file=sys.stderr)

            if install_faiss_gpu:
                print("Đang cài đặt faiss-gpu...")
                subprocess.run(['pip', 'install', 'faiss-gpu', '-q', '--break-system-packages'], check=True, capture_output=True, text=True)
            else:
                print("Đang cài đặt faiss-cpu...")
                subprocess.run(['pip', 'install', 'faiss-cpu', '-q', '--break-system-packages'], check=True, capture_output=True, text=True)
        else:
            print("Bỏ qua cài đặt pip.")

        # Tải checkpoint mô hình BEiT3 và tokenizer
        if not skip_wget_models:
            if not os.path.exists(DEFAULT_MODEL_CHECKPOINT_LOCAL):
                print("Đang tải checkpoint mô hình BEiT3...")
                subprocess.run(['wget', 'https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth', '-q'], check=True, capture_output=True, text=True)
            else:
                print(f"Checkpoint mô hình '{DEFAULT_MODEL_CHECKPOINT_LOCAL}' đã tồn tại. Bỏ qua tải xuống.")

            if not os.path.exists(DEFAULT_TOKENIZER_MODEL_LOCAL):
                print("Đang tải mô hình tokenizer BEiT3...")
                subprocess.run(['wget', 'https://github.com/addf400/files/download/beit3/beit3.spm', '-q'], check=True, capture_output=True, text=True)
            else:
                print(f"Mô hình tokenizer '{DEFAULT_TOKENIZER_MODEL_LOCAL}' đã tồn tại. Bỏ qua tải xuống.")
        else:
            print("Bỏ qua tải xuống mô hình và tokenizer.")
            
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình thiết lập môi trường (Lệnh '{' '.join(e.cmd)}' thất bại): {e.stderr}", file=sys.stderr)
        os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu trước khi thoát
        sys.exit(1)
    except Exception as e:
        print(f"Một lỗi không mong muốn xảy ra trong quá trình thiết lập: {e}", file=sys.stderr)
        os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu trước khi thoát
        sys.exit(1)
    
    print("--- Thiết lập môi trường hoàn tất ---")
    os.chdir(original_cwd) # Trở lại thư mục làm việc ban đầu sau khi thiết lập xong

# =================================================================
# === PHẦN 2: TẢI MÔ HÌNH VÀ TOKENIZER ===
# =================================================================
def load_beit3_model(model_checkpoint_path, tokenizer_model_path, device):
    """Tải mô hình BEiT3 và tokenizer."""
    print("\n--- Đang tải mô hình BEiT3 và Tokenizer ---")
    print(f"Sử dụng thiết bị: {device}")

    try:
        # Các import này yêu cầu 'unilm/beit3' phải có trong sys.path
        # Chúng ta sẽ thêm nó vào sys.path để đảm bảo các module được tìm thấy
        # Giả định script được chạy từ thư mục gốc chứa thư mục 'unilm'
        beit3_module_path = os.path.join(os.getcwd(), 'unilm', 'beit3')
        if os.path.exists(beit3_module_path) and beit3_module_path not in sys.path:
            sys.path.append(beit3_module_path)
            print(f"Đã thêm '{beit3_module_path}' vào sys.path.")

        tokenizer = XLMRobertaTokenizer(tokenizer_model_path)
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu', weights_only=True)
        model = create_model('beit3_large_patch16_384_retrieval')
        model.load_state_dict(checkpoint['model'])
        model.eval().to(device)
        print("--- Tải mô hình thành công ---")
        return model, tokenizer
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file mô hình hoặc tokenizer: {e}. Vui lòng kiểm tra đường dẫn '{model_checkpoint_path}' và '{tokenizer_model_path}'. Đảm bảo bạn đã chạy thiết lập môi trường hoặc các tệp này đã được tải xuống và nằm trong thư mục 'unilm/beit3'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải mô hình BEiT3: {e}", file=sys.stderr)
        sys.exit(1)

# =================================================================
# === PHẦN 3: CÁC HÀM HỖ TRỢ ===
# =================================================================

def create_image_transform(image_size=384):
    """Tạo đối tượng biến đổi ảnh."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
    ])

def process_image(image_path, transform, device):
    """
    Tải, biến đổi ảnh và xử lý trường hợp ảnh bị lỗi.
    Nếu ảnh lỗi, trả về None.
    """
    try:
        image = default_loader(image_path)
        if image is None:
            raise IOError("Loader returned None for the image.")
        image_tensor = transform(image).unsqueeze(0).to(device)
        return image_tensor
    except (IOError, OSError) as e:
        # Bắt các lỗi liên quan đến file (hỏng, không đọc được,...)
        print(f"\n[!] Cảnh báo: Bỏ qua ảnh hỏng hoặc không đọc được: {os.path.basename(image_path)}. Lỗi: {e}", file=sys.stderr)
        global corrupted_images_list # Truy cập danh sách toàn cục
        corrupted_images_list.append(image_path)
        return None

# =================================================================
# === PHẦN 4: TẠO INDEX ===
# =================================================================

def create_faiss_index_shards(
    model, device, transform,
    image_folder, limit_images, chunk_size, output_index_dir,
    metadata_output_path
):
    """
    Thu thập ảnh, tính toán embedding theo chunk, và lưu các shard của index FAISS.
    """
    print("\n--- Bắt đầu tạo Index FAISS và Metadata ---")

    all_image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    print(f"Tìm thấy {len(all_image_paths)} tổng số ảnh trong thư mục '{image_folder}'.")

    if limit_images > 0:
        keyframe_paths = all_image_paths[:min(limit_images, len(all_image_paths))]
        print(f"Đang xử lý {len(keyframe_paths)} ảnh đầu tiên (từ chỉ mục 0 đến {len(keyframe_paths) - 1}).")
    else:
        keyframe_paths = all_image_paths
        print(f"Đang xử lý tất cả {len(keyframe_paths)} ảnh.")

    os.makedirs(output_index_dir, exist_ok=True)

    num_chunks = (len(keyframe_paths) + chunk_size - 1) // chunk_size
    final_processed_paths = [] # Danh sách để lưu các đường dẫn đã được xử lý thành công

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(keyframe_paths))
        
        chunk_paths = keyframe_paths[start_idx:end_idx]
        if not chunk_paths:
            continue

        print(f"\n--- Đang xử lý Chunk {i+1}/{num_chunks} (Ảnh từ {start_idx} đến {end_idx-1}) ---")
        
        chunk_embeddings = []
        chunk_processed_paths = [] # Lưu các path đã xử lý thành công của chunk này

        with torch.no_grad():
            for img_path in tqdm(chunk_paths, desc=f"Tính Embedding cho chunk {i}"):
                img_tensor = process_image(img_path, transform, device)
                if img_tensor is not None: # Chỉ xử lý nếu ảnh không bị lỗi
                    img_embedding, _ = model(img_tensor, only_infer=True)
                    img_embedding = torch.nn.functional.normalize(img_embedding, dim=-1)
                    chunk_embeddings.append(img_embedding.cpu().numpy())
                    chunk_processed_paths.append(img_path) # Thêm path đã xử lý thành công
        
        if not chunk_embeddings:
            print(f"Cảnh báo: Chunk {i} không có ảnh hợp lệ nào để xử lý. Bỏ qua việc tạo index.", file=sys.stderr)
            continue

        final_processed_paths.extend(chunk_processed_paths) # Thêm các path thành công của chunk vào danh sách tổng

        chunk_embeddings = np.vstack(chunk_embeddings).astype('float32') # Đảm bảo float32 cho FAISS
        
        # Tạo và lưu index cho chunk này
        d = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(d) # Sử dụng khoảng cách L2 cho các embedding đã chuẩn hóa
        index.add(chunk_embeddings)
        
        index_filename = os.path.join(output_index_dir, f"keyframe_part_{i}.index")
        faiss.write_index(index, index_filename)
        print(f"Đã lưu index cho chunk {i} với {index.ntotal} vector vào {index_filename}")

    # Lưu danh sách các đường dẫn đã được XỬ LÝ THÀNH CÔNG
    np.save(metadata_output_path, np.array(final_processed_paths))
    print(f"\nHoàn tất tạo tất cả các shard index. Tổng số ảnh đã xử lý thành công: {len(final_processed_paths)}")
    print(f"Tổng số ảnh lỗi đã bỏ qua: {len(corrupted_images_list)}")
    if corrupted_images_list:
        print("Danh sách các ảnh lỗi đã được lưu trong biến 'corrupted_images_list'.")

# =================================================================
# === PHẦN CHÍNH: XỬ LÝ SCRIPT BẰNG ĐỐI SỐ DÒNG LỆNH ===
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo Index FAISS cho Embedding ảnh BEiT3.")
    
    # Đối số cho việc thiết lập môi trường
    parser.add_argument("--skip_setup", action="store_true",
                        help="Bỏ qua việc clone repo unilm, cài đặt các gói và tải mô hình/tokenizer. Giả định môi trường đã được thiết lập.")
    parser.add_argument("--skip_git_clone", action="store_true",
                        help="Bỏ qua việc clone kho lưu trữ unilm nếu nó đã tồn tại. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")
    parser.add_argument("--skip_pip_install", action="store_true",
                        help="Bỏ qua việc cài đặt các gói python. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")
    parser.add_argument("--skip_wget_models", action="store_true",
                        help="Bỏ qua việc tải xuống mô hình BEiT3 và tokenizer. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")
    parser.add_argument("--install_faiss_gpu", action="store_true",
                        help="Cài đặt faiss-gpu thay vì faiss-cpu.")

    # Đối số cho đường dẫn mô hình và tokenizer (tương đối với CWD của beit3_dir)
    parser.add_argument("--model_checkpoint_path", type=str, 
                        default=DEFAULT_MODEL_CHECKPOINT_LOCAL,
                        help=f"Đường dẫn đến checkpoint mô hình BEiT3 (ví dụ: {DEFAULT_MODEL_CHECKPOINT_LOCAL}). Đường dẫn này sẽ được tìm trong thư mục unilm/beit3.")
    parser.add_argument("--tokenizer_model_path", type=str, 
                        default=DEFAULT_TOKENIZER_MODEL_LOCAL,
                        help=f"Đường dẫn đến mô hình tokenizer BEiT3 (ví dụ: {DEFAULT_TOKENIZER_MODEL_LOCAL}). Đường dẫn này sẽ được tìm trong thư mục unilm/beit3.")

    # Đối số cho dữ liệu ảnh và đầu ra
    parser.add_argument("--image_folder", type=str, 
                        default=DEFAULT_IMAGE_FOLDER,
                        help=f"Đường dẫn đến thư mục chứa các ảnh để tạo embedding. Mặc định: {DEFAULT_IMAGE_FOLDER}")
    parser.add_argument("--limit_images", type=int, default=250000,
                        help="Số lượng ảnh tối đa để xử lý. Đặt 0 để xử lý tất cả ảnh.")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Số lượng ảnh để xử lý trong mỗi chunk khi tạo index.")
    parser.add_argument("--output_index_dir", type=str, 
                        default=DEFAULT_OUTPUT_INDEX_DIR,
                        help=f"Thư mục để lưu các shard của index FAISS. Mặc định: {DEFAULT_OUTPUT_INDEX_DIR}")
    parser.add_argument("--metadata_output_path", type=str, 
                        default=DEFAULT_METADATA_OUTPUT_PATH,
                        help=f"Đường dẫn để lưu tệp NumPy chứa đường dẫn của các ảnh đã xử lý. Mặc định: {DEFAULT_METADATA_OUTPUT_PATH}")
    
    args = parser.parse_args()

    # Xác định thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Bước 1: Thiết lập Môi trường (có điều kiện)
    initial_cwd = os.getcwd() # Lưu CWD ban đầu
    if not args.skip_setup:
        # Hàm setup_environment sẽ thay đổi CWD đến unilm/beit3 và sau đó quay lại initial_cwd
        setup_environment(
            base_dir=initial_cwd, # Truyền thư mục gốc nơi unilm nên được đặt
            skip_git_clone=args.skip_git_clone,
            skip_pip_install=args.skip_pip_install,
            skip_wget_models=args.skip_wget_models,
            install_faiss_gpu=args.install_faiss_gpu
        )
    else:
        print("--- Đã bỏ qua thiết lập môi trường theo yêu cầu ---")
        # Đảm bảo 'unilm/beit3' nằm trong sys.path nếu bỏ qua thiết lập, vì việc tải mô hình phụ thuộc vào nó
        beit3_module_path = os.path.join(initial_cwd, 'unilm', 'beit3')
        if os.path.exists(beit3_module_path) and beit3_module_path not in sys.path:
            sys.path.append(beit3_module_path)
            print(f"Đã thêm '{beit3_module_path}' vào sys.path.")
        elif not os.path.exists(beit3_module_path):
            print(f"Cảnh báo: Thư mục '{beit3_module_path}' không tìm thấy. Việc tải mô hình có thể thất bại nếu các tệp không ở vị trí dự kiến.", file=sys.stderr)

    # Bước 2: Tải Mô hình và Tokenizer
    # Đường dẫn mô hình và tokenizer là tương đối với 'unilm/beit3', được xử lý bằng cách thêm vào sys.path
    model, tokenizer = load_beit3_model(
        model_checkpoint_path=args.model_checkpoint_path,
        tokenizer_model_path=args.tokenizer_model_path,
        device=device
    )

    # Bước 3: Tạo Biến đổi Ảnh
    transform = create_image_transform()

    # Bước 4: Tạo các mảnh Index FAISS
    create_faiss_index_shards(
        model=model,
        device=device,
        transform=transform,
        image_folder=args.image_folder,
        limit_images=args.limit_images,
        chunk_size=args.chunk_size,
        output_index_dir=args.output_index_dir,
        metadata_output_path=args.metadata_output_path
    )
    
    print("\n--- Quá trình tạo Index hoàn tất ---")
    if corrupted_images_list:
        print(f"Các ảnh bị hỏng/lỗi đã được bỏ qua: {len(corrupted_images_list)} ảnh.")
        # Tùy chọn lưu danh sách các ảnh bị hỏng
        # with open("corrupted_images.txt", "w") as f:
        #     for img_path in corrupted_images_list:
        #         f.write(f"{img_path}\n")