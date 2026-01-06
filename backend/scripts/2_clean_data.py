import re
import sys
import unicodedata
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))


class DatasetCleaner:
    """Class xử lý làm sạch dataset ca dao"""
    
    def __init__(self):
        self.stats = {
            'original': 0,
            'empty_removed': 0,
            'short_removed': 0,
            'long_removed': 0,
            'invalid_removed': 0,
            'duplicate_removed': 0,
            'final': 0
        }
        self.removed_samples = {
            'empty': [],
            'short': [],
            'long': [],
            'invalid': [],
            'duplicate': []
        }
    
    # ========== BƯỚC 1: XÓA DÒNG RỖNG ==========
    def remove_empty_lines(self, lines):
        """Xóa dòng trống và chỉ có khoảng trắng"""
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped:
                cleaned.append(stripped)
            else:
                self.stats['empty_removed'] += 1
                self.removed_samples['empty'].append(line)
        
        return cleaned
    
    # ========== BƯỚC 2: CHUẨN HÓA UNICODE ==========
    def normalize_unicode(self, text):
        """
        Chuẩn hóa dấu tiếng Việt (NFC normalization)
        VD: 'á' có thể là 1 ký tự hoặc 'a' + dấu sắc → chuẩn hóa về 1 dạng
        """
        return unicodedata.normalize('NFC', text)
    
    # ========== BƯỚC 3: XÓA KÝ TỰ KHÔNG HỢP LỆ ==========
    def clean_text(self, text):
        """
        Xóa số, ký tự đặc biệt, giữ lại chữ cái tiếng Việt
        """
        original = text
        
        # 1. Xóa số đầu dòng (VD: "1. Ăn quả..." → "Ăn quả...")
        text = re.sub(r'^\d+[\.\):\s]+', '', text)
        
        # 2. Xóa dấu ngoặc, gạch ngang
        text = re.sub(r'[\(\)\[\]{}\-–—_]', ' ', text)
        
        # 3. Giữ chữ cái, số, dấu phẩy, chấm, khoảng trắng
        text = re.sub(r'[^\w\s,\.]', '', text)
        
        # 4. Xóa số (nếu còn)
        text = re.sub(r'\d+', '', text)
        
        # 5. Xóa dấu câu đầu/cuối
        text = text.strip(',. ')
        
        # 6. Chuẩn hóa khoảng trắng (nhiều space → 1 space)
        text = ' '.join(text.split())
        
        return text
    
    # ========== BƯỚC 4: CHUYỂN CHỮ THƯỜNG ==========
    def normalize_case(self, text):
        """
        Chuyển về chữ thường
        Ca dao/tục ngữ thường không có tên riêng nên an toàn
        """
        return text.lower()
    
    # ========== BƯỚC 5: KIỂM TRA HỢP LỆ ==========
    def is_valid_proverb(self, text):
        """
        Kiểm tra câu có phải ca dao/tục ngữ hợp lệ không
        
        Tiêu chí:
        - Có ít nhất 50% chữ cái tiếng Việt
        - Không chứa URL, email
        - Không toàn ký tự đặc biệt
        """
        if not text or len(text.strip()) == 0:
            return False, "rỗng"
        
        # Kiểm tra URL
        if re.search(r'http[s]?://|www\.', text):
            return False, "có URL"
        
        # Kiểm tra email
        if re.search(r'\S+@\S+\.\S+', text):
            return False, "có email"
        
        # Kiểm tra tỷ lệ chữ cái tiếng Việt
        vietnamese_letters = re.findall(
            r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', 
            text.lower()
        )
        
        text_without_space = text.replace(' ', '').replace(',', '').replace('.', '')
        if len(text_without_space) == 0:
            return False, "chỉ có khoảng trắng"
        
        letter_ratio = len(vietnamese_letters) / len(text_without_space)
        
        if letter_ratio < 0.5:
            return False, f"ít chữ cái ({letter_ratio:.0%})"
        
        return True, None
    
    # ========== BƯỚC 6: LỌC THEO ĐỘ DÀI ==========
    def filter_by_length(self, text, min_words=4, max_words=30):
        """
        Kiểm tra độ dài hợp lệ
        Ca dao thường 5-15 từ
        """
        words = text.split()
        word_count = len(words)
        
        if word_count < min_words:
            return False, f"quá ngắn ({word_count} từ)"
        
        if word_count > max_words:
            return False, f"quá dài ({word_count} từ)"
        
        return True, None
    
    # ========== PIPELINE CHÍNH ==========
    def clean(self, input_file, output_file, min_words=4, max_words=30):
        """Chạy toàn bộ pipeline làm sạch"""
        
        print("\n" + "="*70)
        print("🧹 LÀM SẠCH DATASET CA DAO")
        print("="*70)
        
        # Đọc file
        print(f"\n📁 Đọc file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_file}")
            return False
        
        self.stats['original'] = len(lines)
        print(f"✓ Đọc thành công {self.stats['original']:,} dòng")
        
        # ========== STEP 1: Xóa dòng rỗng ==========
        print(f"\n{'─'*70}")
        print("BƯỚC 1: Xóa dòng rỗng")
        lines = self.remove_empty_lines(lines)
        print(f"✓ Xóa {self.stats['empty_removed']:,} dòng rỗng")
        print(f"  Còn lại: {len(lines):,} dòng")
        
        # ========== STEP 2-5: Xử lý từng dòng ==========
        print(f"\n{'─'*70}")
        print("BƯỚC 2-5: Chuẩn hóa văn bản")
        
        cleaned_lines = []
        
        for line in lines:
            # 2. Chuẩn hóa Unicode
            line = self.normalize_unicode(line)
            
            # 3. Xóa ký tự không hợp lệ
            line = self.clean_text(line)
            
            # 4. Chuyển chữ thường
            line = self.normalize_case(line)
            
            # 5. Kiểm tra hợp lệ
            is_valid, reason = self.is_valid_proverb(line)
            if not is_valid:
                self.stats['invalid_removed'] += 1
                self.removed_samples['invalid'].append((line, reason))
                continue
            
            # 6. Kiểm tra độ dài
            length_ok, reason = self.filter_by_length(line, min_words, max_words)
            if not length_ok:
                if "ngắn" in reason:
                    self.stats['short_removed'] += 1
                    self.removed_samples['short'].append((line, reason))
                else:
                    self.stats['long_removed'] += 1
                    self.removed_samples['long'].append((line, reason))
                continue
            
            cleaned_lines.append(line)
        
        print(f"✓ Xóa {self.stats['invalid_removed']:,} câu không hợp lệ")
        print(f"✓ Xóa {self.stats['short_removed']:,} câu quá ngắn")
        print(f"✓ Xóa {self.stats['long_removed']:,} câu quá dài")
        print(f"  Còn lại: {len(cleaned_lines):,} câu")
        
        # ========== STEP 7: Xóa trùng lặp ==========
        print(f"\n{'─'*70}")
        print("BƯỚC 6: Xóa trùng lặp")
        
        seen = set()
        unique_lines = []
        
        for line in cleaned_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
            else:
                self.stats['duplicate_removed'] += 1
                self.removed_samples['duplicate'].append(line)
        
        print(f"✓ Xóa {self.stats['duplicate_removed']:,} câu trùng lặp")
        print(f"  Còn lại: {len(unique_lines):,} câu")
        
        self.stats['final'] = len(unique_lines)
        
        # ========== LƯU KẾT QUẢ ==========
        print(f"\n{'─'*70}")
        print("BƯỚC 7: Lưu kết quả")
        
        # Tạo thư mục nếu chưa có
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in unique_lines:
                f.write(line + '\n')
        
        print(f"✓ Đã lưu: {output_file}")
        
        # ========== BÁO CÁO CUỐI CÙNG ==========
        self.print_report()
        
        # ========== MẪU DỮ LIỆU ==========
        self.print_samples(unique_lines)
        
        return True
    
    def print_report(self):
        """In báo cáo tổng hợp"""
        print(f"\n{'='*70}")
        print("📊 BÁO CÁO TỔNG HỢP")
        print("="*70)
        
        print(f"\n📈 Thống kê:")
        print(f"   Dòng gốc:              {self.stats['original']:>6,}")
        print(f"   ├─ Xóa rỗng:           {self.stats['empty_removed']:>6,}")
        print(f"   ├─ Xóa không hợp lệ:   {self.stats['invalid_removed']:>6,}")
        print(f"   ├─ Xóa quá ngắn:       {self.stats['short_removed']:>6,}")
        print(f"   ├─ Xóa quá dài:        {self.stats['long_removed']:>6,}")
        print(f"   └─ Xóa trùng lặp:      {self.stats['duplicate_removed']:>6,}")
        print(f"   {'─'*35}")
        print(f"   ✅ Còn lại:            {self.stats['final']:>6,}")
        
        # Tỷ lệ giữ lại
        if self.stats['original'] > 0:
            retention = (self.stats['final'] / self.stats['original']) * 100
            print(f"\n📊 Tỷ lệ giữ lại: {retention:.1f}%")
            
            if retention < 50:
                print(f"   ⚠️  CẢNH BÁO: Mất hơn 50% dữ liệu!")
                print(f"       Cân nhắc nới lỏng tiêu chí (min_words, max_words)")
            elif retention < 70:
                print(f"   ⚠️  Lưu ý: Mất {100-retention:.1f}% dữ liệu")
            else:
                print(f"   ✅ Tốt! Giữ được phần lớn dữ liệu")
        
        # Chi tiết các mẫu bị xóa
        print(f"\n📝 Chi tiết các mẫu bị xóa:")
        
        if self.removed_samples['short']:
            print(f"\n   🔍 Top 3 câu quá ngắn:")
            for line, reason in self.removed_samples['short'][:3]:
                print(f"      • '{line}' - {reason}")
        
        if self.removed_samples['invalid']:
            print(f"\n   🔍 Top 3 câu không hợp lệ:")
            for line, reason in self.removed_samples['invalid'][:3]:
                preview = line[:50] + '...' if len(line) > 50 else line
                print(f"      • '{preview}' - {reason}")
    
    def print_samples(self, data, n=10):
        """Hiển thị mẫu dữ liệu sau khi làm sạch"""
        print(f"\n{'='*70}")
        print(f"📝 MẪU DỮ LIỆU SAU KHI LÀM SẠCH ({n} câu đầu)")
        print("="*70)
        
        for i, line in enumerate(data[:n], 1):
            word_count = len(line.split())
            print(f"   {i:2d}. {line} ({word_count} từ)")


# ========== MAIN ==========
if __name__ == "__main__":
    # Đường dẫn
    BASE_DIR = Path(__file__).parent.parent
    INPUT_FILE = BASE_DIR / "data" / "raw" / "dataset.txt"
    OUTPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.txt"
    
    # Tham số làm sạch
    MIN_WORDS = 4    # Tối thiểu 4 từ
    MAX_WORDS = 30   # Tối đa 30 từ
    
    print("\n🚀 BẮT ĐẦU LÀM SẠCH DATASET")
    print(f"📥 Input:  {INPUT_FILE}")
    print(f"📤 Output: {OUTPUT_FILE}")
    print(f"⚙️  Cấu hình: {MIN_WORDS}-{MAX_WORDS} từ")
    
    # Chạy cleaner
    cleaner = DatasetCleaner()
    success = cleaner.clean(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        min_words=MIN_WORDS,
        max_words=MAX_WORDS
    )
    
    if success:
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH!")
        print("="*70)
        print(f"\n📌 Bước tiếp theo: Chạy 3_create_splits.py để tạo train/val/test\n")
    else:
        print(f"\n❌ Làm sạch thất bại!")
        sys.exit(1)