"""
PREPROCESSING SCRIPT: LÀM SẠCH + TẠO SPLITS CHO RETRIEVAL MODEL
File: preprocessing.py

Mục đích: Một pipeline hoàn chỉnh để:
1. Làm sạch dataset (xóa rỗng, chuẩn hóa, xóa trùng)
2. Chia train/val/test splits
3. Lưu output ở dạng JSON + TXT

Chạy:
  python preprocessing.py

Input:  data/dataset_normalized.txt
Output: data/processed/
        ├── cleaned_dataset.txt
        ├── train.json / train.txt
        ├── val.json / val.txt
        └── test.json / test.txt
"""

import json
import random
import sys
import re
import unicodedata
import hashlib
from pathlib import Path
from collections import Counter


class DataCleaner:
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
    
    def clean(self, input_file, min_words=3):
        """
        Pipeline làm sạch dataset
        
        Args:
            input_file: File input (dataset_normalized.txt)
            min_words: Tối thiểu số từ cho mỗi câu
            
        Returns:
            List các câu đã được làm sạch
        """
        
        print(f"\n{'─'*70}")
        print("🧹 LÀM SẠCH DATASET")
        print(f"{'─'*70}")
        
        # Đọc file
        print(f"📂 Đọc file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_file}")
            return None
        
        self.stats['original'] = len(lines)
        print(f"✓ Đọc thành công {self.stats['original']:,} dòng")
        
        # Xử lý từng dòng
        print(f"\nBƯỚC 1: Làm sạch và lọc dữ liệu")
        
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
        print(f"\nBƯỚC 2: Xóa trùng lặp")
        
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
        
        return unique_lines
    
    def print_report(self):
        """In báo cáo tổng hợp"""
        print(f"\n{'─'*70}")
        print("📊 BÁO CÁO LÀMM SẠCH")
        print("─"*70)
        
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
    
    def compute_checksum(self, data):
        """Tính checksum của cleaned data để verify reproducibility"""
        text = '\n'.join(data)
        return hashlib.md5(text.encode()).hexdigest()


class DataSplitter:
    """Chia dataset cho retrieval model - train 100%, test lấy 30% (có thể trùng)"""
    
    def __init__(self, test_ratio=0.3):
        """
        Args:
            test_ratio: Tỷ lệ lấy test set từ toàn bộ data (mặc định 30%)
                       Train sẽ là toàn bộ 100% data
                       Test được lấy từ cùng pool (có thể trùng train)
        """
        assert 0 < test_ratio < 1
        self.test_ratio = test_ratio
        
        # Đặt seed để reproducible
        random.seed(42)
    
    def split_data(self, data):
        """
        Chia dataset:
        - Train: 100% toàn bộ data
        - Test: 30% lấy ngẫu nhiên từ data (có thể trùng train)
        """
        # Shuffle data để lấy test random
        data_shuffled = data.copy()
        random.shuffle(data_shuffled)
        
        n = len(data)
        test_size = int(n * self.test_ratio)
        
        return {
            'train': data,  # 100% toàn bộ data
            'test': data_shuffled[:test_size]  # 30% lấy từ data (có thể trùng)
        }
    
    def analyze_lengths(self, data):
        """Phân tích độ dài câu"""
        lengths = [len(sentence.split()) for sentence in data]
        return {
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'avg': sum(lengths) / len(lengths) if lengths else 0,
            'total': len(data)
        }
    
    def split_and_save(self, cleaned_data, output_dir):
        """
        Pipeline chia dataset và lưu
        
        Args:
            cleaned_data: List các câu đã làm sạch
            output_dir: Thư mục output
            
        Returns:
            Dict chứa train/val/test splits
        """
        print(f"\n{'─'*70}")
        print("📦 TẠO DATASET TRAIN/TEST")
        print(f"{'─'*70}")
        
        # Phân tích độ dài
        length_stats = self.analyze_lengths(cleaned_data)
        print(f"\nBƯỚC 1: Phân tích độ dài")
        print(f"   Số câu:        {length_stats['total']:,}")
        print(f"   Ngắn nhất:     {length_stats['min']} từ")
        print(f"   Dài nhất:      {length_stats['max']} từ")
        print(f"   Trung bình:    {length_stats['avg']:.1f} từ")
        
        # Chia dataset
        print(f"\nBƯỚC 2: Chia train/test")
        print(f"   Train: 100% (toàn bộ data)")
        print(f"   Test:  {self.test_ratio*100:.0f}% (lấy từ data, có thể trùng train)")
        
        splits = self.split_data(cleaned_data)
        
        # Tạo thư mục output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu files
        print(f"\nBƯỚC 3: Lưu files")
        
        stats = {}
        for split_name, split_data in splits.items():
            # Lưu dạng JSON
            file_path = output_path / f"{split_name}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # Phân tích split này
            stats[split_name] = self.analyze_lengths(split_data)
            
            print(f"   ✓ {split_name:5s}: {len(split_data):>5,} câu" +
                  f" (avg: {stats[split_name]['avg']:>5.1f} từ)")
        
        # Kiểm tra chất lượng
        self.validate_splits(splits)
        
        # Cảnh báo nếu dataset quá nhỏ
        if len(splits['train']) < 200:
            print(f"\n⚠️  CẢNH BÁO: Train set chỉ có {len(splits['train'])} câu")
            print(f"   Khuyến nghị:")
            print(f"   • Thu thập thêm ca dao (tối thiểu 500+ câu)")
        elif len(splits['train']) < 500:
            print(f"\n⚠️  Lưu ý: Train set có {len(splits['train'])} câu")
            print(f"   Có thể cần thêm dữ liệu để model hoạt động tốt hơn")
        else:
            print(f"\n✅ Dataset size OK: {len(splits['train'])} câu train")
        
        return splits
    
    def validate_splits(self, splits):
        """Kiểm tra chất lượng split"""
        print(f"\nBƯỚC 4: Kiểm tra chất lượng")
        
        print(f"   ✅ Train: 100% data")
        print(f"   ✅ Test: {len(splits['test']):,} câu (có thể trùng train)")
        
        # Kiểm tra câu không rỗng
        issues = 0
        for split_name, split_data in splits.items():
            for i, sentence in enumerate(split_data):
                if not sentence.strip():
                    print(f"   ⚠️  {split_name}[{i}]: Câu rỗng!")
                    issues += 1
        
        if issues == 0:
            print(f"   ✅ Tất cả câu đều hợp lệ")
        else:
            print(f"   ⚠️  Tìm thấy {issues} câu có vấn đề")
    
    def print_examples(self, splits):
        """Hiển thị ví dụ từ mỗi split"""
        print(f"\n{'─'*70}")
        print("📝 VÍ DỤ TỪ MỖI SPLIT (3 câu mỗi loại)")
        print("─"*70)
        
        for split_name in ['train', 'test']:
            print(f"\n{split_name.upper()}:")
            for i, sentence in enumerate(splits[split_name][:3], 1):
                word_count = len(sentence.split())
                # Cắt ngắn nếu quá dài
                display = sentence if len(sentence) <= 60 else sentence[:57] + "..."
                print(f"   {i}. {display}")
                print(f"      ({word_count} từ)")


# ========== MAIN PIPELINE ==========
if __name__ == "__main__":
    # ⭐ SET SEED NGAY TỪ ĐẦU để đảm bảo reproducibility
    random.seed(42)
    
    # Đường dẫn
    PROJECT_ROOT = Path(__file__).parent.parent.parent  # /NLP_v01/
    DATA_DIR = PROJECT_ROOT / "data"
    INPUT_FILE = DATA_DIR / "dataset_normalized.txt"
    OUTPUT_DIR = DATA_DIR / "processed"
    METADATA_FILE = OUTPUT_DIR / "metadata.json"
    
    # Cấu hình
    MIN_WORDS = 3
    TEST_RATIO = 0.3  # Lấy 30% làm test set (có thể trùng train)
    
    print("\n" + "="*70)
    print("🚀 PREPROCESSING PIPELINE (Cho Retrieval Model)")
    print("="*70)
    print(f"\n📥 Input:  {INPUT_FILE}")
    print(f"📤 Output: {OUTPUT_DIR}/")
    print(f"\n⚙️  Cấu hình:")
    print(f"   • Min words per sentence: {MIN_WORDS}")
    print(f"   • Train: 100% (toàn bộ data)")
    print(f"   • Test:  {TEST_RATIO*100:.0f}% (lấy từ data, có thể trùng train)")
    
    # Bước 1: Làm sạch dataset
    print(f"\n{'='*70}")
    print("PHASE 1: DATA CLEANING")
    print("="*70)
    
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean(
        input_file=INPUT_FILE,
        min_words=MIN_WORDS
    )
    
    if cleaned_data is None:
        print(f"\n❌ Làm sạch thất bại!")
        sys.exit(1)
    
    # Lưu cleaned data
    cleaned_file = OUTPUT_DIR / "cleaned_dataset.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cleaned_file, 'w', encoding='utf-8') as f:
        for line in cleaned_data:
            f.write(line + '\n')
    print(f"\n✓ Đã lưu cleaned data: {cleaned_file}")
    
    # Tính checksum của cleaned data
    checksum = cleaner.compute_checksum(cleaned_data)
    print(f"\n🔐 Checksum cleaned data: {checksum}")
    
    # Bước 2: Chia train/test
    print(f"\n{'='*70}")
    print("PHASE 2: CREATE SPLITS")
    print("="*70)
    
    splitter = DataSplitter(test_ratio=TEST_RATIO)
    
    splits = splitter.split_and_save(
        cleaned_data=cleaned_data,
        output_dir=OUTPUT_DIR
    )
    
    # Tóm tắt
    print(f"\n{'='*70}")
    print("✅ PREPROCESSING HOÀN THÀNH!")
    print("="*70)
    print(f"\n📊 Kết quả:")
    print(f"   • Train: {len(splits['train']):,} câu (100%)")
    print(f"   • Test:  {len(splits['test']):,} câu ({TEST_RATIO*100:.0f}%, có thể trùng train)")
    print(f"\n📁 Output files:")
    print(f"   ✓ {cleaned_file}")
    print(f"   ✓ {OUTPUT_DIR / 'train.json'}")
    print(f"   ✓ {OUTPUT_DIR / 'test.json'}")
    
    # Lưu metadata để verify reproducibility
    metadata = {
        'timestamp': str(Path(cleaned_file).stat().st_mtime),
        'cleaned_data_checksum': checksum,
        'total_cleaned_sentences': len(cleaned_data),
        'train_size': len(splits['train']),
        'test_size': len(splits['test']),
        'test_ratio': TEST_RATIO,
        'min_words': MIN_WORDS,
        'random_seed': 42
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"   ✓ {METADATA_FILE}")
    
    print(f"\n🔐 REPRODUCIBILITY:")
    print(f"   • Random seed: 42 (cố định)")
    print(f"   • Cleaned data checksum: {checksum}")
    print(f"   • Metadata saved: {METADATA_FILE}")
    print(f"\n   ℹ️  Các lần chạy tiếp theo sẽ tạo kết quả giống hệt")
    print(f"       nếu input file không thay đổi!")
    
    print(f"\n📌 Bước tiếp theo: Train retrieval model với 100% dữ liệu")
    print()
