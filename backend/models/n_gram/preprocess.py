import os
import re
from pyvi import ViTokenizer

# --- CẤU HÌNH ---
# Save data files to project root /data directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

input_raw_file = os.path.join(DATA_DIR, "dataset.txt")
output_norm_file = os.path.join(DATA_DIR, "dataset_normalized.txt")
output_final_file = os.path.join(DATA_DIR, "train_data_seg.txt")

# Tạo dữ liệu giả (Xóa khi chạy thật)
if not os.path.exists(input_raw_file):
    os.makedirs(os.path.dirname(input_raw_file), exist_ok=True)
    with open(input_raw_file, "w", encoding="utf-8") as f:
        f.write("Thân em như tấm lụa đào\nPhất phơ giữa chợ biết vào tay ai.\n\nCông cha như núi Thái Sơn\nNghĩa mẹ như nước trong nguồn chảy ra.")
    print(f"[OK] Created sample data at: {input_raw_file}")

def normalize_step(input_path, output_path):
    """Normalize raw Vietnamese text"""
    print(f"\n{'='*60}")
    print(f"[Step 1] Normalizing data")
    print(f"{'='*60}")
    print(f"Input: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
    except FileNotFoundError:
        print("[ERROR] Lỗi: Không tìm thấy file đầu vào.")
        return False

    # 1. Tách bài dựa trên dòng trống
    poems = re.split(r'\n\s*\n', raw_content.strip())
    clean_lines = []

    for poem in poems:
        verses = poem.strip().split('\n')
        # Xóa khoảng trắng thừa, chuyển chữ thường
        clean_verses = [v.strip().lower() for v in verses if v.strip()]

        if clean_verses:
            # Nối các câu thơ bằng dấu chấm
            merged_line = ". ".join(clean_verses)

            # Đảm bảo kết thúc bằng dấu chấm
            if not merged_line.endswith('.'):
                merged_line += "."
            merged_line = merged_line.replace("..", ".")

            clean_lines.append(merged_line)

    # Ghi file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(clean_lines))

    print(f"[OK] Normalized {len(clean_lines)} lines.")
    print(f"[FILE] Saved to: {output_path}")
    if clean_lines:
        print(f"[SAMPLE] {clean_lines[0][:50]}...")
    
    return True

def tokenize_step(input_path, output_path):
    """Tokenize Vietnamese text using PyVi"""
    print(f"\n{'='*60}")
    print(f"[Step 2] Tokenizing & Adding sentence markers")
    print(f"{'='*60}")
    print(f"Input: {input_path}")

    if not os.path.exists(input_path):
        print(f"❌ Lỗi: Chưa chạy bước Chuẩn hóa (Không thấy file input).")
        return False

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    final_data = []
    errors = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if not line: 
            continue

        try:
            # 1. Tách từ bằng PyVi
            tokenized_line = ViTokenizer.tokenize(line)

            # 2. Thêm sentence markers
            complete_line = f"<s> {tokenized_line} </s>"
            final_data.append(complete_line)
        except Exception as e:
            print(f"[WARNING] Line {i+1} failed: {e}")
            errors += 1
            continue

    # Ghi file kết quả
    with open(output_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(final_data))

    print(f"[OK] Tokenized {len(final_data)} sentences successfully.")
    if errors > 0:
        print(f"[WARNING] {errors} sentences failed.")
    print(f"[FILE] Ready for KenLM training: {output_path}")
    if final_data:
        print(f"[SAMPLE] {final_data[0][:60]}...")
    
    return True

def main():
    """Run preprocessing pipeline"""
    print(f"""
    {'='*60}
    VIETNAMESE TEXT PREPROCESSING PIPELINE
    {'='*60}
    Data Directory: {DATA_DIR}
    """)
    
    # Step 1: Normalize
    if not normalize_step(input_raw_file, output_norm_file):
        print("[ERROR] Normalization failed!")
        return False
    
    # Step 2: Tokenize
    if not tokenize_step(output_norm_file, output_final_file):
        print("[ERROR] Tokenization failed!")
        return False
    
    print(f"""
    {'='*60}
    PREPROCESSING COMPLETED SUCCESSFULLY!
    {'='*60}
    Ready for KenLM training!
    """)
    return True

if __name__ == "__main__":
    main()
    
    
# cd n_gram
# python preprocess.py