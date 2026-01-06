"""
SCRIPT 4: KIỂM TRA CHẤT LƯỢNG DATASET
Chạy: python 4_validate_dataset.py
"""

import json
from pathlib import Path

def validate_dataset(file_path):
    """Kiểm tra dataset có lỗi không"""
    print(f"\n🔍 Kiểm tra: {file_path.name}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   Số samples: {len(data):,}")
    
    issues = []
    
    for i, sample in enumerate(data):
        # 1. Kiểm tra input không rỗng
        if not sample['input'].strip():
            issues.append(f"Sample {i}: Input rỗng")
        
        # 2. Kiểm tra target không rỗng
        if not sample['target'].strip():
            issues.append(f"Sample {i}: Target rỗng")
        
        # 3. Kiểm tra input + target = full
        reconstructed = sample['input'] + ' ' + sample['target']
        if reconstructed.strip() != sample['full'].strip():
            issues.append(f"Sample {i}: Input+Target ≠ Full")
        
        # 4. Kiểm tra input ngắn hơn full
        if len(sample['input'].split()) >= len(sample['full'].split()):
            issues.append(f"Sample {i}: Input dài hơn full")
    
    if issues:
        print(f"   ❌ Phát hiện {len(issues)} lỗi:")
        for issue in issues[:5]:  # Hiển thị 5 lỗi đầu
            print(f"      • {issue}")
    else:
        print(f"   ✅ Không có lỗi!")
    
    return len(issues) == 0

# MAIN
if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    
    print("\n" + "="*70)
    print("🔍 KIỂM TRA CHẤT LƯỢNG DATASET")
    print("="*70)
    
    all_ok = True
    for file_name in ['train.json', 'val.json', 'test.json']:
        file_path = PROCESSED_DIR / file_name
        if file_path.exists():
            ok = validate_dataset(file_path)
            all_ok = all_ok and ok
        else:
            print(f"\n⚠️  Không tìm thấy {file_name}")
            all_ok = False
    
    print("\n" + "="*70)
    if all_ok:
        print("✅ TẤT CẢ FILES ĐỀU HỢP LỆ!")
    else:
        print("❌ CÓ VẤN ĐỀ VỚI DATASET!")
    print("="*70 + "\n")