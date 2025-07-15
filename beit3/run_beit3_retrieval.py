import os
import subprocess
import glob
import numpy as np
import pandas as pd
import torch
import faiss
from tqdm.auto import tqdm
from PIL import Image
import argparse # Import argparse
import sys # Import sys for stderr messages

# =================================================================
# === CẤU HÌNH ĐƯỜNG DẪN MẶC ĐỊNH & KIỂM TRA MÔI TRƯỜNG KAGGLE ===
# =================================================================

# Đường dẫn mặc định cho tệp mô hình và tokenizer (sẽ tải xuống trong 'unilm/beit3')
DEFAULT_MODEL_CHECKPOINT_LOCAL = './beit3_large_patch16_384_coco_retrieval.pth'
DEFAULT_TOKENIZER_MODEL_LOCAL = './beit3.spm'

# Đường dẫn mặc định cho môi trường Kaggle
KAGGLE_FINAL_INDEX_PATH = '/kaggle/input/beit-concat/beit3_final_index.index'
KAGGLE_FINAL_METADATA_PATH = '/kaggle/input/beit-concat/beit3_final_metadata.npy'
KAGGLE_QUERY_CSV_PATH = '/kaggle/input/refine-query-0-to-1000/submission.csv'
KAGGLE_SUBMISSION_OUTPUT_PATH = '/kaggle/working/submission.csv'

# Xác định nếu đang ở môi trường Kaggle (để thiết lập đường dẫn mặc định)
IS_KAGGLE = os.path.exists('/kaggle/input')

# =================================================================
# === PHẦN 1: CÀI ĐẶT VÀ THIẾT LẬP MÔI TRƯỜNG ===
# =================================================================

def setup_environment(skip_git_clone=False, skip_pip_install=False, skip_wget_models=False):
    """
    Tải mã nguồn, cài đặt thư viện và tải mô hình.
    Thay đổi thư mục làm việc hiện tại sang 'unilm/beit3'.
    """
    print("--- Bắt đầu thiết lập môi trường ---")
    
    original_cwd = os.getcwd() # Lưu thư mục làm việc ban đầu

    try:
        # Clone unilm repository nếu chưa có và không bỏ qua
        if not skip_git_clone:
            if not os.path.exists('unilm'):
                print("Cloning unilm repository...")
                subprocess.run(['git', 'clone', 'https://github.com/fonzi22/unilm.git'], check=True)
            else:
                print("Kho lưu trữ 'unilm' đã tồn tại. Bỏ qua việc clone.")
        
        # Thay đổi thư mục làm việc sang 'unilm/beit3'
        beit3_dir = os.path.join(original_cwd, 'unilm', 'beit3')
        if not os.path.exists(beit3_dir):
            print(f"Lỗi: Thư mục '{beit3_dir}' không tìm thấy sau khi cố gắng clone 'unilm'. Thoát.", file=sys.stderr)
            sys.exit(1)
        os.chdir(beit3_dir)
        print(f"Đã thay đổi thư mục làm việc hiện tại sang: {os.getcwd()}")

        # Cài đặt các gói cần thiết
        if not skip_pip_install:
            print("Đang cài đặt các gói cần thiết...")
            try:
                subprocess.run(['pip', 'install', '-r', 'requirements.txt', '-q'], check=True)
            except Exception as e:
                print(f"Cảnh báo: Không thể cài đặt các yêu cầu từ requirements.txt: {e}. Thử cài đặt từng gói.", file=sys.stderr)
                # Dự phòng nếu requirements.txt thất bại
                subprocess.run(['pip', 'install', 'timm', 'transformers', 'tokenizers', 'fvcore', 'omegaconf', 'einops', '-q'], check=True)
            
            print("Đang cài đặt faiss-cpu...")
            subprocess.run(['pip', 'install', 'faiss-cpu', '-q'], check=True)
        else:
            print("Bỏ qua cài đặt pip.")

        # Tải checkpoint mô hình BEiT3
        if not skip_wget_models:
            if not os.path.exists(DEFAULT_MODEL_CHECKPOINT_LOCAL):
                print("Đang tải checkpoint mô hình BEiT3...")
                subprocess.run(['wget', 'https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth', '-q'], check=True)
            else:
                print(f"Checkpoint mô hình '{DEFAULT_MODEL_CHECKPOINT_LOCAL}' đã tồn tại. Bỏ qua tải xuống.")

            # Tải mô hình tokenizer BEiT3
            if not os.path.exists(DEFAULT_TOKENIZER_MODEL_LOCAL):
                print("Đang tải mô hình tokenizer BEiT3...")
                subprocess.run(['wget', 'https://github.com/addf400/files/releases/download/beit3/beit3.spm', '-q'], check=True)
            else:
                print(f"Mô hình tokenizer '{DEFAULT_TOKENIZER_MODEL_LOCAL}' đã tồn tại. Bỏ qua tải xuống.")
        else:
            print("Bỏ qua tải xuống mô hình và tokenizer.")
            
    except subprocess.CalledProcessError as e:
        print(f"Lỗi trong quá trình thiết lập môi trường (Lệnh thất bại): {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Một lỗi không mong muốn xảy ra trong quá trình thiết lập: {e}", file=sys.stderr)
        sys.exit(1)
    print("--- Thiết lập môi trường hoàn tất ---")

# =================================================================
# === PHẦN 2: TẢI MÔ HÌNH VÀ TOKENIZER ===
# =================================================================
def load_beit3_model(model_checkpoint_path, tokenizer_model_path):
    """Tải mô hình BEiT3 và tokenizer."""
    print("\n--- Đang tải mô hình BEiT3 và Tokenizer ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    try:
        # Các import này phụ thuộc vào việc thư mục 'beit3' nằm trong sys.path hoặc là CWD
        from timm import create_model
        from transformers import XLMRobertaTokenizer
        import modeling_finetune # Đảm bảo file này tồn tại trong CWD (unilm/beit3)

        tokenizer = XLMRobertaTokenizer(tokenizer_model_path)
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model = create_model('beit3_large_patch16_384_retrieval')
        model.load_state_dict(checkpoint['model'])
        model.eval().to(device)
        print("--- Tải mô hình thành công ---")
        return model, tokenizer, device
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file mô hình hoặc tokenizer: {e}. Vui lòng kiểm tra đường dẫn '{model_checkpoint_path}' và '{tokenizer_model_path}', hoặc chạy `setup_environment`.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải mô hình BEiT3: {e}", file=sys.stderr)
        sys.exit(1)


# =================================================================
# === PHẦN 3: CÁC HÀM HỖ TRỢ ===
# =================================================================

def text_to_embedding(model, tokenizer, text, device):
    """Hàm chuyển đổi văn bản thành vector embedding."""
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        _, language_cls = model.forward(
            text_description=inputs['input_ids'],
            only_infer=True
        )
    return language_cls.cpu().numpy()

def query_image_by_text(model, tokenizer, text, index, metadata, device, top_k=10):
    """Hàm tìm kiếm ảnh dựa trên văn bản."""
    text_embedding = text_to_embedding(model, tokenizer, text, device)
    _, indices = index.search(text_embedding, top_k)
    
    retrieved_paths = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata): # Đảm bảo chỉ số hợp lệ
            retrieved_paths.append(metadata[idx])
        else:
            # Xử lý trường hợp chỉ số không hợp lệ (ví dụ: Faiss trả về -1)
            print(f"Cảnh báo: Chỉ số không hợp lệ {idx} được trả về từ Faiss. Bỏ qua.", file=sys.stderr)
            # Có thể thêm placeholder nếu cần
            # retrieved_paths.append("#INVALID_PATH#") 
    return retrieved_paths


# =================================================================
# === PHẦN CHÍNH: XỬ LÝ SCRIPT BẰNG ĐỐI SỐ DÒNG LỆNH ===
# =================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Retrieval sử dụng mô hình BEiT3.")
    
    # Đối số cho việc thiết lập môi trường
    parser.add_argument("--skip_setup", action="store_true",
                        help="Bỏ qua việc clone repo unilm, cài đặt các gói và tải mô hình/tokenizer. Giả định môi trường đã được thiết lập.")
    parser.add_argument("--skip_git_clone", action="store_true",
                        help="Bỏ qua việc clone kho lưu trữ unilm nếu nó đã tồn tại. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")
    parser.add_argument("--skip_pip_install", action="store_true",
                        help="Bỏ qua việc cài đặt các gói python. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")
    parser.add_argument("--skip_wget_models", action="store_true",
                        help="Bỏ qua việc tải xuống mô hình BEiT3 và tokenizer. Chỉ có hiệu lực nếu --skip_setup không được sử dụng.")

    # Đối số cho đường dẫn mô hình và tokenizer (tương đối với 'unilm/beit3' sau khi chdir)
    parser.add_argument("--model_checkpoint_path", type=str, 
                        default=DEFAULT_MODEL_CHECKPOINT_LOCAL,
                        help=f"Đường dẫn đến checkpoint mô hình BEiT3 (ví dụ: {DEFAULT_MODEL_CHECKPOINT_LOCAL}). Đường dẫn này là tương đối với 'unilm/beit3' sau khi thiết lập.")
    parser.add_argument("--tokenizer_model_path", type=str, 
                        default=DEFAULT_TOKENIZER_MODEL_LOCAL,
                        help=f"Đường dẫn đến mô hình tokenizer BEiT3 (ví dụ: {DEFAULT_TOKENIZER_MODEL_LOCAL}). Đường dẫn này là tương đối với 'unilm/beit3' sau khi thiết lập.")

    # Đối số cho đường dẫn dữ liệu
    parser.add_argument("--final_index_path", type=str, 
                        default=KAGGLE_FINAL_INDEX_PATH if IS_KAGGLE else './beit3_final_index.index',
                        help=f"Đường dẫn đến tệp index FAISS đã được xây dựng trước. Mặc định (Kaggle): {KAGGLE_FINAL_INDEX_PATH}, (Local): ./beit3_final_index.index")
    parser.add_argument("--final_metadata_path", type=str, 
                        default=KAGGLE_FINAL_METADATA_PATH if IS_KAGGLE else './beit3_final_metadata.npy',
                        help=f"Đường dẫn đến tệp NumPy chứa metadata ảnh (đường dẫn). Mặc định (Kaggle): {KAGGLE_FINAL_METADATA_PATH}, (Local): ./beit3_final_metadata.npy")
    parser.add_argument("--query_csv_path", type=str, 
                        default=KAGGLE_QUERY_CSV_PATH if IS_KAGGLE else './query_private.csv',
                        help=f"Đường dẫn đến tệp CSV chứa các truy vấn. Mặc định (Kaggle): {KAGGLE_QUERY_CSV_PATH}, (Local): ./query_private.csv")
    parser.add_argument("--submission_output_path", type=str, 
                        default=KAGGLE_SUBMISSION_OUTPUT_PATH if IS_KAGGLE else './submission.csv',
                        help=f"Đường dẫn để lưu tệp CSV submission được tạo ra. Mặc định (Kaggle): {KAGGLE_SUBMISSION_OUTPUT_PATH}, (Local): ./submission.csv")
    
    # Đối số cho tham số truy xuất
    parser.add_argument("--top_k", type=int, default=10,
                        help="Số lượng ảnh hàng đầu để truy xuất cho mỗi truy vấn.")

    args = parser.parse_args()

    # Lưu thư mục làm việc ban đầu trước khi bất kỳ thay đổi CWD nào xảy ra
    original_start_cwd = os.getcwd()

    # Bước 1: Thiết lập môi trường (có điều kiện)
    if not args.skip_setup:
        setup_environment(
            skip_git_clone=args.skip_git_clone,
            skip_pip_install=args.skip_pip_install,
            skip_wget_models=args.skip_wget_models
        )
    else:
        print("--- Đã bỏ qua thiết lập môi trường theo yêu cầu ---")
        # Đảm bảo chúng ta đang ở trong thư mục chính xác nếu bỏ qua thiết lập
        # nhưng các phần tiếp theo mong đợi nó.
        beit3_dir_check = os.path.join(original_start_cwd, 'unilm', 'beit3')
        if os.path.exists(beit3_dir_check):
            os.chdir(beit3_dir_check)
            print(f"Đã thay đổi thư mục làm việc hiện tại sang: {os.getcwd()} (để tải mô hình BEiT3)")
        else:
            print(f"Cảnh báo: '{beit3_dir_check}' không tìm thấy. Việc tải mô hình/tokenizer có thể thất bại nếu đường dẫn là tương đối.", file=sys.stderr)


    # Bước 2: Tải Mô hình và Tokenizer
    # Các đường dẫn này là tương đối với CWD, hiện tại phải là 'unilm/beit3'
    model, tokenizer, device = load_beit3_model(args.model_checkpoint_path, args.tokenizer_model_path)

    # Bước 3: Tải Index và Metadata
    print("\n--- Chuẩn bị Index và Metadata cho việc tìm kiếm ---")
    try:
        all_metadata = np.load(args.final_metadata_path, allow_pickle=True)
        final_index = faiss.read_index(args.final_index_path)
        print(f"Đã tải thành công Index từ {args.final_index_path} và Metadata từ {args.final_metadata_path}")
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp Index hoặc Metadata: {e}. Vui lòng kiểm tra đường dẫn '{args.final_index_path}' và '{args.final_metadata_path}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải Index hoặc Metadata: {e}", file=sys.stderr)
        sys.exit(1)


    # Bước 4: Xử lý Query và Tạo Submission CSV
    print(f"\n--- Bắt đầu xử lý tệp query: {args.query_csv_path} ---")
    try:
        queries_df = pd.read_csv(args.query_csv_path)
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy tệp query CSV: {e}. Vui lòng kiểm tra đường dẫn '{args.query_csv_path}'.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Tìm thấy {len(queries_df)} câu query.")

    results = []

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Đang xử lý các query"):
        query_index = row['query_index']
        # Ưu tiên 'query_context' nếu có, nếu không thì dùng 'query_text'.
        # Nếu cả hai không có, cảnh báo và bỏ qua hoặc dùng chuỗi rỗng.
        if 'query_context' in row and pd.notna(row['query_context']):
            query_text = row['query_context']
        elif 'query_text' in row and pd.notna(row['query_text']):
            query_text = row['query_text']
        else:
            print(f"Cảnh báo: Không tìm thấy cột 'query_text' hoặc 'query_context' hợp lệ cho query_index {query_index}. Bỏ qua truy vấn này.", file=sys.stderr)
            results.append([query_index] + ['#'] * args.top_k) # Thêm hàng rỗng để giữ định dạng
            continue

        try:
            retrieved_paths = query_image_by_text(
                model=model,
                tokenizer=tokenizer,
                text=query_text,
                index=final_index,
                metadata=all_metadata,
                device=device,
                top_k=args.top_k
            )
            
            # Lấy ID ảnh (tên file không bao gồm đuôi .jpg)
            image_ids = [os.path.basename(p).split('.')[0] for p in retrieved_paths]
            
            # Đệm bằng '#' nếu có ít hơn top_k kết quả được truy xuất
            image_ids += ['#'] * (args.top_k - len(image_ids))
            
            result_row = [query_index] + image_ids[:args.top_k] # Đảm bảo chỉ lấy top_k
            results.append(result_row)
        except Exception as e:
            print(f"Lỗi khi xử lý query_index {query_index} ('{query_text}'): {e}. Bỏ qua truy vấn này.", file=sys.stderr)
            # Thêm một hàng với placeholder nếu có lỗi
            results.append([query_index] + ['#'] * args.top_k)


    print("\n--- Đang tạo tệp submission ---")
    columns = ['query_id'] + [f'image_id_{i}' for i in range(1, args.top_k + 1)]
    submission_df = pd.DataFrame(results, columns=columns)

    # Đảm bảo thư mục đầu ra tồn tại cho tệp submission
    output_dir = os.path.dirname(args.submission_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    submission_df.to_csv(args.submission_output_path, index=False)
    print(f"Tệp submission đã được lưu thành công tại: {args.submission_output_path}")