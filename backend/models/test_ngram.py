"""
Test N-gram model thủ công
"""

from ngram import NgramModel
from pathlib import Path
import json

# Load trained model
BASE_DIR = Path(__file__).parent.parent
model_path = BASE_DIR / "trained_models" / "ngram_model.pkl"

model = NgramModel()
model.load(model_path)

print("\n" + "="*60)
print("🧪 TEST N-GRAM MODEL")
print("="*60)

# Test cases
test_cases = [
    "ăn",                    # Rất mơ hồ
    "ăn quả",                # Ít mơ hồ
    "ăn quả nhớ",            # Rõ ràng
    "có công",               # Mơ hồ
    "gần mực",               # Rõ ràng
    "học thầy không",        # Thiếu 1 từ
    "xyz abc"                # Không có trong dataset
]

for inp in test_cases:
    print(f"\n📝 Input: '{inp}'")
    
    try:
        candidates = model.predict_multiple(inp, top_k=3)
        
        if not candidates:
            print("   ❌ Không tìm thấy kết quả")
        else:
            for i, cand in enumerate(candidates, 1):
                print(f"   {i}. {cand['text']}")
                print(f"      📊 Confidence: {cand['confidence']:.1%}")
                print(f"      🔧 Method: {cand['method']}")
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")

print("\n" + "="*60)