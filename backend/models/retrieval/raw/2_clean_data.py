"""
SCRIPT 2: LÀM SẠCH DATASET (PHIÊN BẢN ĐƠN GIẢN CHO RETRIEVAL)
File: 2_clean_data_simple.py

Mục đích: Làm sạch dataset để dùng cho retrieval model
- Xóa dòng rỗng
- Chuẩn hóa Unicode
- Xóa ký tự đặc biệt không cần thiết
- Giữ nguyên câu đầy đủ (không tạo input/target)

Chạy:
  python 2_clean_data_simple.py
"""

import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter


class SimpleDatasetCleaner:
    """Làm sạch dataset ca dao cho retrieval model"""
    
    def __init__(self):
        self.stats = {
            'original': 0,
            'empty_removed': 0,
            'too_short_removed': 0,
            'duplicate_removed': 0,
            'final': 0
        }
    
    def normalize_unicode(self, text):
        """Chuẩn hóa dấu tiếng Việt (NFC normalization)"""
        return unicodedata.normalize('NFC', text)
    
    def clean_text(self, text):
        """
        Làm sạch text nhẹ nhàng
        - Xóa khoảng trắng thừa
        - Chuẩn hóa dấu câu
        - Giữ nguyên nội dung
        """
        # Xóa BOM nếu có
        text = text.replace('\ufeff', '')
        
        # Chuẩn hóa khoảng trắng
        text = ' '.join(text.split())
        
        # Xóa dấu cách đầu/cuối
        text = text.strip()
        
        return text
    
    def is_valid(self, text, min_words=3):
        """
        Kiểm tra câu có hợp lệ không
        - Không rỗng
        - Ít nhất min_words từ
        - Có ít nhất 50% chữ cái tiếng Việt
        """
        if not text or len(text.strip()) == 0:
            return False
        
        words = text.split()
        if len(words) < min_words:
            return False
        
        # Kiểm tra tỷ lệ chữ cái tiếng Việt
        vietnamese_letters = re.findall(
            r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', 
            text.lower()
        )
        
        text_without_space = text.replace(' ', '').replace(',', '').replace('.', '')
        if len(text_without_space) == 0:
            return False
        
        letter_ratio = len(vietnamese_letters) / len(text_without_space)
        
        return letter_ratio >= 0.5
    
    def clean(self, input_file, output_file, min_words=3):
        """Pipeline chính để làm sạch dataset"""
        
        print("\n" + "="*70)
        print("🧹 LÀM SẠCH DATASET CA DAO (CHO RETRIEVAL)")
        print("="*70)
        
        # Đọc file
        print(f"\n📂 Đọc file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_file}")
            return False
        
        self.stats['original'] = len(lines)
        print(f"✓ Đọc thành công {self.stats['original']:,} dòng")
        
        # Xử lý từng dòng
        print(f"\n{'─'*70}")
        print("BƯỚC 1: Làm sạch và lọc dữ liệu")
        
        cleaned_lines = []
        
        for line in lines:
            # Bỏ dòng rỗng
            if not line.strip():
                self.stats['empty_removed'] += 1
                continue
            
            # Chuẩn hóa Unicode
            line = self.normalize_unicode(line)
            
            # Làm sạch text
            line = self.clean_text(line)
            
            # Kiểm tra hợp lệ
            if not self.is_valid(line, min_words):
                self.stats['too_short_removed'] += 1
                continue
            
            cleaned_lines.append(line)
        
        print(f"✓ Xóa {self.stats['empty_removed']:,} dòng rỗng")
        print(f"✓ Xóa {self.stats['too_short_removed']:,} câu quá ngắn hoặc không hợp lệ")
        print(f"  Còn lại: {len(cleaned_lines):,} câu")
        
        # Xóa trùng lặp
        print(f"\n{'─'*70}")
        print("BƯỚC 2: Xóa trùng lặp")
        
        seen = set()
        unique_lines = []
        
        for line in cleaned_lines:
            # So sánh không phân biệt hoa thường
            line_lower = line.lower()
            if line_lower not in seen:
                seen.add(line_lower)
                unique_lines.append(line)
            else:
                self.stats['duplicate_removed'] += 1
        
        print(f"✓ Xóa {self.stats['duplicate_removed']:,} câu trùng lặp")
        print(f"  Còn lại: {len(unique_lines):,} câu")
        
        self.stats['final'] = len(unique_lines)
        
        # Lưu kết quả
        print(f"\n{'─'*70}")
        print("BƯỚC 3: Lưu kết quả")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in unique_lines:
                f.write(line + '\n')
        
        print(f"✓ Đã lưu: {output_file}")
        
        # Báo cáo
        self.print_report()
        
        # Hiển thị mẫu
        self.print_samples(unique_lines, n=10)
        
        return True
    
    def print_report(self):
        """In báo cáo tổng hợp"""
        print(f"\n{'='*70}")
        print("📊 BÁO CÁO TỔNG HỢP")
        print("="*70)
        
        print(f"\n📈 Thống kê:")
        print(f"   Dòng gốc:              {self.stats['original']:>6,}")
        print(f"   ├─ Xóa rỗng:           {self.stats['empty_removed']:>6,}")
        print(f"   ├─ Xóa không hợp lệ:   {self.stats['too_short_removed']:>6,}")
        print(f"   └─ Xóa trùng lặp:      {self.stats['duplicate_removed']:>6,}")
        print(f"   {'─'*35}")
        print(f"   ✅ Còn lại:            {self.stats['final']:>6,}")
        
        if self.stats['original'] > 0:
            retention = (self.stats['final'] / self.stats['original']) * 100
            print(f"\n📊 Tỷ lệ giữ lại: {retention:.1f}%")
            
            if retention < 70:
                print(f"   ⚠️  Lưu ý: Mất {100-retention:.1f}% dữ liệu")
            else:
                print(f"   ✅ Tốt! Giữ được phần lớn dữ liệu")
    
    def print_samples(self, data, n=10):
        """Hiển thị mẫu dữ liệu sau khi làm sạch"""
        print(f"\n{'='*70}")
        print(f"📝 MẪU DỮ LIỆU SAU KHI LÀM SẠCH ({n} câu đầu)")
        print("="*70)
        
        for i, line in enumerate(data[:n], 1):
            word_count = len(line.split())
            # Cắt ngắn nếu câu quá dài
            display = line if len(line) <= 70 else line[:67] + "..."
            print(f"   {i:2d}. {display} ({word_count} từ)")


# ========== MAIN ==========
if __name__ == "__main__":
    # Đường dẫn - BẠN CẦN SỬA ĐỔI CHỖ NÀY
# SAU (phù hợp với cấu trúc của bạn)
    BASE_DIR = Path(__file__).parent.parent  # backend/
    INPUT_FILE = BASE_DIR / "data" / "raw" / "dataset_normalized.txt"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.txt"
    
    # Tham số
    MIN_WORDS = 3  # Tối thiểu 3 từ (câu ngắn nhất trong ca dao)
    
    print("\n🚀 BẮT ĐẦU LÀM SẠCH DATASET")
    print(f"📥 Input:  {INPUT_FILE}")
    print(f"📤 Output: {OUTPUT_FILE}")
    print(f"⚙️  Cấu hình: Tối thiểu {MIN_WORDS} từ")
    
    # Chạy cleaner
    cleaner = SimpleDatasetCleaner()
    success = cleaner.clean(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        min_words=MIN_WORDS
    )
    
    if success:
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH!")
        print("="*70)
        print(f"\n📌 Bước tiếp theo:")
        print(f"   1. Kiểm tra file {OUTPUT_FILE}")
        print(f"   2. Chạy 3_create_splits_simple.py để tạo train/val/test")
        print(f"   3. Train retrieval model\n")
    else:
        print(f"\n❌ Làm sạch thất bại!")
        sys.exit(1)