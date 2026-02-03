"""
SCRIPT 3: TẠO TRAIN/VAL/TEST SPLIT (CHO RETRIEVAL)
File: 3_create_splits_simple.py

Mục đích: Chia dataset thành train/val/test cho retrieval model
- Mỗi câu giữ nguyên dạng đầy đủ
- Không tạo input/target variants
- Chỉ lưu danh sách câu đầy đủ

Chạy:
  python 3_create_splits_simple.py
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter


class SimpleDatasetSplitter:
    """Chia dataset đơn giản cho retrieval model"""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Đặt seed để reproducible
        random.seed(42)
    
    def split_data(self, data):
        """Chia dataset theo tỷ lệ train/val/test"""
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        
        return {
            'train': data[:train_end],
            'val': data[train_end:val_end],
            'test': data[val_end:]
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
    
    def create(self, input_file, output_dir):
        """
        Pipeline chính
        
        Args:
            input_file: File đã làm sạch (cleaned_dataset.txt)
            output_dir: Thư mục output
        """
        print("\n" + "="*70)
        print("📦 TẠO DATASET TRAIN/VAL/TEST (CHO RETRIEVAL)")
        print("="*70)
        
        # Đọc file
        print(f"\n📂 Đọc file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_file}")
            print(f"   Hãy chạy 2_clean_data_simple.py trước!")
            return False
        
        print(f"✓ Đọc thành công {len(sentences):,} câu ca dao")
        
        # Phân tích độ dài
        length_stats = self.analyze_lengths(sentences)
        print(f"\n📊 Phân tích độ dài:")
        print(f"   Số câu:        {length_stats['total']:,}")
        print(f"   Ngắn nhất:     {length_stats['min']} từ")
        print(f"   Dài nhất:      {length_stats['max']} từ")
        print(f"   Trung bình:    {length_stats['avg']:.1f} từ")
        
        # Chia dataset
        print(f"\n{'─'*70}")
        print("BƯỚC 1: Chia train/val/test")
        print(f"   Train: {self.train_ratio*100:.0f}%")
        print(f"   Val:   {self.val_ratio*100:.0f}%")
        print(f"   Test:  {self.test_ratio*100:.0f}%")
        
        splits = self.split_data(sentences)
        
        # Tạo thư mục output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu files
        print(f"\n{'─'*70}")
        print("BƯỚC 2: Lưu files")
        
        stats = {}
        for split_name, split_data in splits.items():
            # Lưu dạng list đơn giản
            file_path = output_path / f"{split_name}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            # Lưu cả dạng text để dễ đọc
            text_path = output_path / f"{split_name}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                for sentence in split_data:
                    f.write(sentence + '\n')
            
            # Phân tích split này
            stats[split_name] = self.analyze_lengths(split_data)
            
            print(f"   ✓ {split_name:5s}: {len(split_data):>5,} câu" +
                  f" (avg: {stats[split_name]['avg']:>5.1f} từ)")
        
        # Kiểm tra chất lượng
        self.validate_splits(splits)
        
        # Hiển thị ví dụ
        self.print_examples(splits)
        
        # Cảnh báo nếu dataset quá nhỏ
        if len(splits['train']) < 200:
            print(f"\n⚠️  CẢNH BÁO: Train set chỉ có {len(splits['train'])} câu")
            print(f"   Model có thể không học tốt. Khuyến nghị:")
            print(f"   • Thu thập thêm ca dao (tối thiểu 500+ câu)")
        elif len(splits['train']) < 500:
            print(f"\n⚠️  Lưu ý: Train set có {len(splits['train'])} câu")
            print(f"   Có thể cần thêm dữ liệu để model hoạt động tốt hơn")
        else:
            print(f"\n✅ Dataset size OK: {len(splits['train'])} câu train")
        
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH!")
        print("="*70)
        print(f"\n📌 Bước tiếp theo: Train retrieval model")
        print(f"   python retrieval.py\n")
        
        return True
    
    def validate_splits(self, splits):
        """Kiểm tra chất lượng split"""
        print(f"\n{'─'*70}")
        print("BƯỚC 3: Kiểm tra chất lượng")
        
        # Kiểm tra không có câu trùng giữa train/val/test
        train_set = set(s.lower() for s in splits['train'])
        val_set = set(s.lower() for s in splits['val'])
        test_set = set(s.lower() for s in splits['test'])
        
        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        
        total_overlap = len(overlap_train_val) + len(overlap_train_test) + len(overlap_val_test)
        
        if total_overlap == 0:
            print(f"   ✅ Không có câu trùng giữa train/val/test")
        else:
            print(f"   ⚠️  Phát hiện {total_overlap} câu trùng!")
            if overlap_train_val:
                print(f"      • Train-Val: {len(overlap_train_val)} câu")
            if overlap_train_test:
                print(f"      • Train-Test: {len(overlap_train_test)} câu")
            if overlap_val_test:
                print(f"      • Val-Test: {len(overlap_val_test)} câu")
        
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
        
        for split_name in ['train', 'val', 'test']:
            print(f"\n{split_name.upper()}:")
            for i, sentence in enumerate(splits[split_name][:3], 1):
                word_count = len(sentence.split())
                # Cắt ngắn nếu quá dài
                display = sentence if len(sentence) <= 60 else sentence[:57] + "..."
                print(f"   {i}. {display}")
                print(f"      ({word_count} từ)")


# ========== MAIN ==========
if __name__ == "__main__":
    # Đường dẫn - BẠN CẦN SỬA ĐỔI CHỖ NÀY
    BASE_DIR = Path(__file__).parent.parent
    INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.txt"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    
    # Cấu hình
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print("\n🚀 BẮT ĐẦU TẠO DATASET SPLITS")
    print(f"📥 Input:  {INPUT_FILE}")
    print(f"📤 Output: {OUTPUT_DIR}/")
    print(f"⚙️  Cấu hình:")
    print(f"   • Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    
    # Chạy splitter
    splitter = SimpleDatasetSplitter(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    success = splitter.create(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR
    )
    
    if not success:
        sys.exit(1)