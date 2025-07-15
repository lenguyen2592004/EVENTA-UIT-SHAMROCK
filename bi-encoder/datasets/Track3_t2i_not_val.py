import os
import json
import csv
import random
import pandas as pd
import argparse # Thêm import argparse
import sys # Thêm import sys để in lỗi ra stderr

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dữ liệu cho nhiệm vụ Text-to-Image (T2I) và Image-to-Text (I2T).")
    
    # Định nghĩa các đối số dòng lệnh
    parser.add_argument("--image_root", type=str, 
                        default="/kaggle/input/train-eventa/train_compressed_scaled_images/train_images_compressed90_scaled05",
                        help="Đường dẫn gốc đến thư mục chứa ảnh train.")
    parser.add_argument("--json_path", type=str, 
                        default="/kaggle/input/t2i-eventa/t2i.json",
                        help="Đường dẫn đến tệp JSON chứa ánh xạ text-to-image.")
    parser.add_argument("--t2i_csv_path", type=str, 
                        default="/kaggle/input/eventa-track-2/query_private.csv",
                        help="Đường dẫn đến tệp CSV chứa các query (dùng để tạo test.csv).")
    parser.add_argument("--output_dir", type=str, 
                        default="./",
                        help="Thư mục để lưu các tệp CSV đầu ra (train.csv, val.csv, test.csv).")
    parser.add_argument("--val_ratio", type=float, 
                        default=0.1,
                        help="Tỷ lệ dữ liệu validation (0.0 đến 1.0). Mặc định là 0.1.")
    parser.add_argument("--random_seed", type=int, 
                        default=42,
                        help="Seed ngẫu nhiên để đảm bảo tính tái lập cho việc chia tập dữ liệu.")
    parser.add_argument("--split_train_val", action="store_true",
                        help="Sử dụng cờ này để chia dữ liệu thành train và validation. Nếu không, chỉ tạo train.csv (không val.csv).")

    args = parser.parse_args()

    # Thiết lập seed ngẫu nhiên
    random.seed(args.random_seed)

    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Bắt đầu chuẩn bị dữ liệu ---")
    print(f"Đường dẫn gốc ảnh train: {args.image_root}")
    print(f"Đường dẫn JSON T2I: {args.json_path}")
    print(f"Đường dẫn CSV Query: {args.t2i_csv_path}")
    print(f"Thư mục đầu ra: {args.output_dir}")
    if args.split_train_val:
        print(f"Tỷ lệ Validation: {args.val_ratio}")
    else:
        print("Sẽ không chia tập validation (val_ratio sẽ bị bỏ qua).")
    print(f"Seed ngẫu nhiên: {args.random_seed}")

    # Bước 1: Đọc tệp JSON và thu thập dữ liệu
    try:
        with open(args.json_path, 'r') as f:
            labels = json.load(f)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file JSON '{args.json_path}'. Vui lòng kiểm tra lại đường dẫn.", file=sys.stderr)
        return
    except json.JSONDecodeError:
        print(f"Lỗi: Không thể đọc file JSON '{args.json_path}'. Kiểm tra định dạng file.", file=sys.stderr)
        return

    data = []
    missing_files_count = 0
    for text, filename in labels.items():
        img_path = os.path.join(args.image_root, filename)
        if os.path.exists(img_path):
            data.append((img_path, text))
        else:
            print(f"⚠️ Cảnh báo: File ảnh không tồn tại: {img_path}", file=sys.stderr)
            missing_files_count += 1
    
    if missing_files_count > 0:
        print(f"Tổng cộng có {missing_files_count} file ảnh bị thiếu.", file=sys.stderr)
    
    if not data:
        print("Lỗi: Không có dữ liệu hợp lệ nào được tìm thấy sau khi kiểm tra ảnh tồn tại. Vui lòng kiểm tra đường dẫn ảnh gốc và JSON.", file=sys.stderr)
        return

    # Bước 2: Chia dữ liệu (nếu được yêu cầu) và ghi ra CSV
    def write_csv(data_list, path, header=["image_path", "text"]):
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row_data in data_list:
                writer.writerow(row_data)

    if args.split_train_val:
        if args.val_ratio > 0 and args.val_ratio < 1:
            train_data, val_data = train_test_split(data, test_size=args.val_ratio, random_state=args.random_seed)
            write_csv(train_data, os.path.join(args.output_dir, "train.csv"))
            write_csv(val_data, os.path.join(args.output_dir, "val.csv"))
            print(f"✅ Đã tạo train.csv ({len(train_data)} mẫu) và val.csv ({len(val_data)} mẫu).")
        else:
            print("Cảnh báo: Tỷ lệ validation không hợp lệ (phải > 0 và < 1). Sẽ chỉ tạo train.csv.", file=sys.stderr)
            write_csv(data, os.path.join(args.output_dir, "train.csv"))
            print(f"✅ Đã tạo train.csv ({len(data)} mẫu) (không có tập validation).")
    else:
        write_csv(data, os.path.join(args.output_dir, "train.csv"))
        print(f"✅ Đã tạo train.csv ({len(data)} mẫu) (không có tập validation).")


    # Bước 3: Đọc tệp CSV t2i và ghi ra test.csv
    print(f"\n--- Đang tạo test.csv từ '{args.t2i_csv_path}' ---")
    try:
        df_test = pd.read_csv(args.t2i_csv_path)
        test_csv_path = os.path.join(args.output_dir, "test.csv")
        # Giả định query_private.csv đã có các cột cần thiết cho test.csv
        # Nếu chỉ muốn cột 'query_text', bạn có thể bỏ comment dòng dưới
        # df_test = df_test[['query_index', 'query_text']] # Ví dụ nếu bạn cần chọn cột
        df_test.to_csv(test_csv_path, index=False) # index=False để không ghi cột index của pandas
        print(f"✅ Đã tạo test.csv ({len(df_test)} mẫu) tại: {test_csv_path}")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file CSV '{args.t2i_csv_path}' cho test set. Bỏ qua tạo test.csv.", file=sys.stderr)
    except Exception as e:
        print(f"Lỗi khi xử lý file CSV '{args.t2i_csv_path}': {e}. Bỏ qua tạo test.csv.", file=sys.stderr)

    print("\n--- Hoàn tất chuẩn bị dữ liệu ---")

if __name__ == "__main__":
    main()