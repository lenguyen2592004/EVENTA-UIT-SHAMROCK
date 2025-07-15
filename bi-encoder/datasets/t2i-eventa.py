import numpy as np # linear algebra (Mặc dù không được sử dụng trực tiếp, giữ lại nếu có ý định sử dụng sau này)
import csv
import json
import os # Thêm import os để kiểm tra đường dẫn
import argparse # Thêm import argparse
import sys # Thêm import sys để in lỗi ra stderr

def create_t2i_json_from_csv(csv_path, json_path):
    """
    Đọc file CSV và tạo file JSON theo định dạng text-to-image.

    Trong đó:
    - Key của JSON là giá trị từ cột 'caption'.
    - Value của JSON là giá trị từ cột 'image_index' + đuôi '.jpg'.

    Args:
        csv_path (str): Đường dẫn đến file CSV đầu vào.
        json_path (str): Đường dẫn đến file JSON đầu ra.
    """
    t2i_data = {}
    
    print(f"--- Bắt đầu tạo JSON từ CSV ---")
    print(f"Đọc dữ liệu từ: {csv_path}")
    print(f"Ghi kết quả ra: {json_path}")

    try:
        # Mở và đọc file CSV với encoding utf-8 để hỗ trợ tiếng Việt
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            # Sử dụng DictReader để dễ dàng truy cập các cột bằng tên
            reader = csv.DictReader(infile)
            
            # Kiểm tra xem các cột cần thiết có tồn tại không
            if 'caption' not in reader.fieldnames or 'image_index' not in reader.fieldnames:
                print(f"Lỗi: File CSV '{csv_path}' phải chứa cả hai cột 'caption' và 'image_index'.", file=sys.stderr)
                return False

            line_count = 0
            skipped_rows = 0
            for row in reader:
                line_count += 1
                # Lấy dữ liệu từ các cột tương ứng
                caption = row.get('caption')
                image_index = row.get('image_index')

                # Chỉ xử lý nếu cả hai giá trị đều tồn tại và không rỗng
                if caption and str(caption).strip() and image_index and str(image_index).strip():
                    # Tạo tên file ảnh bằng cách thêm đuôi '.jpg'
                    filename = f"{image_index}.jpg"
                    
                    # Thêm cặp key-value vào từ điển
                    t2i_data[caption] = filename
                else:
                    skipped_rows += 1
                    # In cảnh báo nếu một dòng bị thiếu dữ liệu
                    print(f"Cảnh báo: Bỏ qua dòng {line_count} không hợp lệ hoặc thiếu dữ liệu: {row}", file=sys.stderr)
            
            if skipped_rows > 0:
                print(f"Tổng cộng {skipped_rows} dòng đã bị bỏ qua do thiếu dữ liệu.", file=sys.stderr)

        # Đảm bảo thư mục đầu ra tồn tại trước khi ghi file JSON
        output_dir = os.path.dirname(json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Ghi từ điển vào file JSON
        with open(json_path, mode='w', encoding='utf-8') as outfile:
            # ensure_ascii=False để hiển thị đúng các ký tự không phải ASCII (như tiếng Việt)
            # indent=4 để file JSON có định dạng đẹp, dễ đọc
            json.dump(t2i_data, outfile, ensure_ascii=False, indent=4)

        print(f"Hoàn thành! Đã tạo file '{json_path}' thành công từ '{csv_path}'.")
        print(f"Tổng cộng {len(t2i_data)} cặp key-value đã được ghi.")
        return True

    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{csv_path}'. Vui lòng kiểm tra lại đường dẫn.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Đã xảy ra một lỗi không mong muốn khi xử lý file: {e}", file=sys.stderr)
        return False

# --- Khối thực thi chính ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển đổi file CSV chứa caption và image_index thành định dạng JSON Text-to-Image.")
    
    # Kiểm tra nếu đang chạy trong môi trường Kaggle để thiết lập đường dẫn mặc định
    is_kaggle = os.path.exists('/kaggle/input')

    # Định nghĩa các đối số dòng lệnh
    parser.add_argument("--csv_path", type=str, 
                        default='/kaggle/input/train-eventa/gt_train.csv' if is_kaggle else 'gt_train.csv',
                        help=f"Đường dẫn đến file CSV đầu vào (chứa cột 'caption' và 'image_index'). Mặc định (Kaggle): /kaggle/input/train-eventa/gt_train.csv, (Local): gt_train.csv")
    parser.add_argument("--json_path", type=str, 
                        default='t2i.json',
                        help=f"Đường dẫn đến file JSON đầu ra. Mặc định: t2i.json")

    args = parser.parse_args()

    # Gọi hàm chính với các đối số đã nhận từ terminal
    success = create_t2i_json_from_csv(csv_path=args.csv_path, json_path=args.json_path)
    
    if not success:
        sys.exit(1) # Thoát với mã lỗi nếu không thành công