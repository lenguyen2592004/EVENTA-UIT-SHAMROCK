import os
import sys # Thêm import sys để dùng sys.stderr
import pandas as pd
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image # Có thể không cần nếu không dùng xử lý ảnh trực tiếp
from tqdm import tqdm # Thêm import tqdm

# Định nghĩa MASTER_PROMPT (nếu nó được sử dụng, cần phải có giá trị cụ thể)
# Nếu MASTER_PROMPT thay đổi hoặc được lấy từ file khác, bạn cần cập nhật phần này.
MASTER_PROMPT = "Refine the following query for an image generation AI. Focus on making it descriptive, concise, and suitable for text-to-image models. Remove any conversational elements or instructions that are not part of the core image description. Keep the core meaning of the original query."


def refine_queries(pipe, input_file, output_file, num_rows_to_process=None):
    """
    Đọc file CSV đầu vào, tinh chỉnh từng query và lưu vào file CSV đầu ra.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{input_file}'. Vui lòng kiểm tra lại đường dẫn.", file=sys.stderr)
        return

    df_to_process = df
    if num_rows_to_process is not None and num_rows_to_process > 0:
        df_to_process = df.head(num_rows_to_process)
        print(f"Tổng số hàng trong file: {len(df)}. Sẽ xử lý {len(df_to_process)} hàng đầu tiên.")
    else:
        print(f"Tổng số hàng trong file: {len(df)}. Sẽ xử lý tất cả {len(df_to_process)} hàng.")


    results = []
    
    for _, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="Đang tinh chỉnh các query"):
        query_index = row['query_index']
        original_query_text = str(row['query_text'])

        full_prompt = f"{MASTER_PROMPT}\n{original_query_text}"

        try:
            response = pipe(full_prompt)
            refined_text = response.text
            
            # --- LƯU Ý: Phần này có thể cần nếu lmdeploy trả về cả prompt hoặc token đặc biệt ---
            # Nếu output của model vẫn chứa prompt hoặc các token như "<|im_end|>", bạn có thể
            # uncomment và chỉnh sửa logic dưới đây để làm sạch nó.
            # Ví dụ:
            # if isinstance(refined_text, str):
            #     # Loại bỏ phần prompt nếu nó lặp lại trong output
            #     if full_prompt in refined_text:
            #         refined_text = refined_text.replace(full_prompt, "").strip()
            #     # Loại bỏ token đặc biệt lmdeploy thường dùng
            #     refined_text = refined_text.replace("<|im_end|>", "").strip()
            #     refined_text = refined_text.replace("Text-to-Image AI Prompt:", "").strip()
            #     refined_text = refined_text.replace("Refined Query:", "").strip()
            #     refined_text = refined_text.strip('"').strip() # Xóa dấu ngoặc kép thừa
            # ---------------------------------------------------------------------------------

            print(f'REFINED-TEXT (query_index {query_index}): {refined_text}\n')
            results.append({'query_index': query_index, 'query_text': refined_text})

        except Exception as e:
            print(f"\nLỗi khi xử lý query_index {query_index}: {e}", file=sys.stderr)
            results.append({'query_index': query_index, 'query_text': original_query_text})

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_file, index=False)
    print(f"\nHoàn tất! Các query đã được tinh chỉnh và lưu vào file: {output_file}")

# === Khối thực thi chính ===
if __name__ == "__main__":
    import argparse # Thêm import argparse

    parser = argparse.ArgumentParser(description="Tinh chỉnh các truy vấn sử dụng mô hình LMDeploy InternVL3-8B.")
    
    parser.add_argument("--model_path", type=str, default='OpenGVLab/InternVL3-8B',
                        help="Đường dẫn hoặc tên mô hình để tải.")
    parser.add_argument("--input_csv", type=str, 
                        default='/kaggle/input/eventa-track-2/query_private.csv' if os.path.exists('/kaggle/input') else 'query_private.csv',
                        help="Đường dẫn đến file CSV đầu vào chứa các query gốc.")
    parser.add_argument("--output_csv", type=str, 
                        default='/kaggle/working/submission.csv' if os.path.exists('/kaggle/input') else 'submission.csv',
                        help="Đường dẫn đến file CSV đầu ra để lưu các query đã tinh chỉnh.")
    parser.add_argument("--tp", type=int, default=2,
                        help="Số lượng GPU (tensor parallelism) để sử dụng cho TurbomindEngineConfig.")
    parser.add_argument("--session_len", type=int, default=16384,
                        help="Độ dài phiên tối đa cho TurbomindEngineConfig.")
    parser.add_argument("--num_rows", type=int, default=None,
                        help="Số lượng hàng (query) cần xử lý từ file CSV đầu vào. Mặc định là tất cả.")

    args = parser.parse_args()

    # Cấu hình backend với tp để sử dụng nhiều GPU
    backend_config = TurbomindEngineConfig(
        session_len=args.session_len, 
        tp=args.tp
    )

    # Cấu hình template chat
    chat_template_config = ChatTemplateConfig(model_name='internvl2_5')

    # Khởi tạo pipeline với cấu hình mới
    print(f"Đang tải mô hình: {args.model_path} với TP={args.tp}, Session Len={args.session_len}...")
    pipe = pipeline(
        args.model_path, 
        backend_config=backend_config, 
        chat_template_config=chat_template_config
    )
    print("Mô hình đã được tải.")

    # Đảm bảo thư mục đầu ra tồn tại
    output_dir = os.path.dirname(args.output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Đọc dữ liệu từ: {args.input_csv}")
    print(f"Lưu kết quả vào: {args.output_csv}")
    
    refine_queries(pipe, args.input_csv, args.output_csv, args.num_rows)