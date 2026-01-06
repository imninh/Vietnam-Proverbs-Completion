"""
SCRIPT 3: TẠO TRAIN/VAL/TEST SPLIT
File: backend/scripts/3_create_splits.py

Chạy:
  cd backend/scripts
  python 3_create_splits.py
"""

import json
import random
import sys
from pathlib import Path
from collections import Counter

sys.path.append(str(Path(__file__).parent.parent))


class DatasetSplitter:
    """Tạo dataset train/val/test với partial inputs"""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def create_partial_variants(self, full_text, num_variants=2):
        """
        Tạo nhiều variants của partial input từ 1 câu
        
        VD: "ăn quả nhớ kẻ trồng cây" (5 từ)
        → Variants:
          1. input: "ăn quả" (40%)           target: "nhớ kẻ trồng cây"
          2. input: "ăn quả nhớ" (60%)       target: "kẻ trồng cây"
        
        Args:
            full_text: Câu đầy đủ
            num_variants: Số variants (2-3 là hợp lý)
        """
        words = full_text.split()
        total_words = len(words)
        
        # Nếu câu quá ngắn, chỉ tạo 1 variant
        if total_words < 4:
            split_point = max(1, total_words // 2)
            return [{
                'full': full_text,
                'input': ' '.join(words[:split_point]),
                'target': ' '.join(words[split_point:]),
                'split_ratio': split_point / total_words
            }]
        
        variants = []
        
        # Tạo các điểm cắt khác nhau
        # VD: Câu 8 từ với 2 variants:
        #   - Variant 1: cắt ở 40% (3 từ input)
        #   - Variant 2: cắt ở 60% (5 từ input)
        
        for i in range(num_variants):
            # Tính ratio: 0.3, 0.4, 0.5, 0.6...
            ratio = 0.3 + (i * 0.15)  # 30%, 45%, 60%...
            
            split_point = max(2, int(total_words * ratio))
            split_point = min(split_point, total_words - 2)  # Ít nhất 2 từ còn lại
            
            variant = {
                'full': full_text,
                'input': ' '.join(words[:split_point]),
                'target': ' '.join(words[split_point:]),
                'split_ratio': round(ratio, 2),
                'input_words': split_point,
                'target_words': total_words - split_point
            }
            
            variants.append(variant)
        
        return variants
    
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
    
    def create(self, input_file, output_dir, variants_per_proverb=2):
        """
        Pipeline chính
        
        Args:
            input_file: File đã làm sạch (cadao_cleaned.txt)
            output_dir: Thư mục output (data/processed/)
            variants_per_proverb: Số variants mỗi câu (2-3)
        """
        print("\n" + "="*70)
        print("📦 TẠO DATASET TRAIN/VAL/TEST")
        print("="*70)
        
        # Đọc file
        print(f"\n📁 Đọc file: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                proverbs = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Không tìm thấy file: {input_file}")
            print(f"   Hãy chạy 2_clean_data.py trước!")
            return False
        
        print(f"✓ Đọc thành công {len(proverbs):,} câu ca dao")
        
        # Tạo variants
        print(f"\n{'─'*70}")
        print(f"BƯỚC 1: Tạo {variants_per_proverb} variants cho mỗi câu")
        
        all_samples = []
        for proverb in proverbs:
            variants = self.create_partial_variants(proverb, num_variants=variants_per_proverb)
            all_samples.extend(variants)
        
        print(f"✓ Tạo được {len(all_samples):,} samples")
        print(f"  ({len(proverbs):,} câu × {variants_per_proverb} variants)")
        
        # Phân tích độ dài input
        input_lengths = [s['input_words'] for s in all_samples]
        target_lengths = [s['target_words'] for s in all_samples]
        
        print(f"\n📊 Thống kê variants:")
        print(f"   Độ dài input:")
        print(f"      Trung bình: {sum(input_lengths)/len(input_lengths):.1f} từ")
        print(f"      Min-Max:    {min(input_lengths)}-{max(input_lengths)} từ")
        print(f"   Độ dài target:")
        print(f"      Trung bình: {sum(target_lengths)/len(target_lengths):.1f} từ")
        print(f"      Min-Max:    {min(target_lengths)}-{max(target_lengths)} từ")
        
        # Chia dataset
        print(f"\n{'─'*70}")
        print("BƯỚC 2: Chia train/val/test")
        print(f"   Train: {self.train_ratio*100:.0f}%")
        print(f"   Val:   {self.val_ratio*100:.0f}%")
        print(f"   Test:  {self.test_ratio*100:.0f}%")
        
        splits = self.split_data(all_samples)
        
        # Tạo thư mục output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Lưu files
        print(f"\n{'─'*70}")
        print("BƯỚC 3: Lưu files JSON")
        
        for split_name, split_data in splits.items():
            file_path = output_path / f"{split_name}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            
            print(f"   ✓ {split_name}.json: {len(split_data):>5,} samples")
        
        # Kiểm tra chất lượng split
        self.validate_splits(splits)
        
        # Hiển thị ví dụ
        self.print_examples(splits['train'])
        
        # Cảnh báo nếu dataset quá nhỏ
        if len(splits['train']) < 200:
            print(f"\n⚠️  CẢNH BÁO: Train set chỉ có {len(splits['train'])} samples")
            print(f"   Mô hình có thể không học tốt. Khuyến nghị:")
            print(f"   • Thu thập thêm ca dao (tối thiểu 300+ câu)")
            print(f"   • Hoặc tăng variants_per_proverb lên 3-4")
        
        print(f"\n{'='*70}")
        print("✅ HOÀN THÀNH!")
        print("="*70)
        print(f"\n📌 Bước tiếp theo: Train models với data đã chuẩn bị\n")
        
        return True
    
    def validate_splits(self, splits):
        """Kiểm tra chất lượng split"""
        print(f"\n{'─'*70}")
        print("BƯỚC 4: Kiểm tra chất lượng")
        
        # Kiểm tra không có câu trùng giữa train/val/test
        train_fulls = set(s['full'] for s in splits['train'])
        val_fulls = set(s['full'] for s in splits['val'])
        test_fulls = set(s['full'] for s in splits['test'])
        
        overlap_train_val = train_fulls & val_fulls
        overlap_train_test = train_fulls & test_fulls
        overlap_val_test = val_fulls & test_fulls
        
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
        
        # Kiểm tra input không rỗng
        issues = 0
        for split_name, split_data in splits.items():
            for i, sample in enumerate(split_data):
                if not sample['input'].strip() or not sample['target'].strip():
                    print(f"   ⚠️  {split_name}[{i}]: Input hoặc target rỗng!")
                    issues += 1
        
        if issues == 0:
            print(f"   ✅ Tất cả samples đều hợp lệ")
        else:
            print(f"   ⚠️  Tìm thấy {issues} samples có vấn đề")
    
    def print_examples(self, train_data, n=5):
        """Hiển thị ví dụ"""
        print(f"\n{'─'*70}")
        print(f"📝 VÍ DỤ TỪ TRAIN SET ({n} samples)")
        print("─"*70)
        
        for i, sample in enumerate(train_data[:n], 1):
            print(f"\n   {i}. Full:   {sample['full']}")
            print(f"      Input:  {sample['input']}")
            print(f"      Target: {sample['target']}")
            print(f"      Split:  {sample['split_ratio']*100:.0f}% " +
                  f"({sample['input_words']} từ input → {sample['target_words']} từ target)")


# ========== MAIN ==========
if __name__ == "__main__":
    # Đường dẫn
    BASE_DIR = Path(__file__).parent.parent
    INPUT_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.txt"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    
    # Cấu hình
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    VARIANTS_PER_PROVERB = 2  # Tăng lên 3 nếu dataset < 400 câu
    
    print("\n🚀 BẮT ĐẦU TẠO DATASET SPLITS")
    print(f"📥 Input:  {INPUT_FILE}")
    print(f"📤 Output: {OUTPUT_DIR}/")
    print(f"⚙️  Cấu hình:")
    print(f"   • Train/Val/Test: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"   • Variants per proverb: {VARIANTS_PER_PROVERB}")
    
    # Chạy splitter
    splitter = DatasetSplitter(
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    success = splitter.create(
        input_file=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        variants_per_proverb=VARIANTS_PER_PROVERB
    )
    
    if not success:
        sys.exit(1)