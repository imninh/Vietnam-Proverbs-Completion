"""
INFERENCE SCRIPT - Sử dụng trained retrieval model

Cách chạy:
  python models/retrieval/inference.py
"""

from pathlib import Path
from train import RetrievalModel

# Đường dẫn
PROJECT_ROOT = Path(__file__).parent.parent.parent  # /NLP_v01/
MODEL_PATH = PROJECT_ROOT / "models" / "retrieval" / "checkpoint" / "retrieval.pkl"

print("\n" + "="*70)
print("🔍 RETRIEVAL MODEL INFERENCE")
print("="*70)

# Load model
if not MODEL_PATH.exists():
    print(f"\n❌ Không tìm thấy model: {MODEL_PATH}")
    print(f"\n💡 Hãy train model trước:")
    print(f"   python models/retrieval/train.py")
    exit(1)

print(f"\n📂 Loading model từ: {MODEL_PATH}")
model = RetrievalModel()
model.load(str(MODEL_PATH))

# Test inference
test_queries = [
    "ăn quả",
    "ăn quả nhớ",
    "có công",
    "gần mực",
    "học thầy"
]

print(f"\n{'─'*70}")
print("🧪 TEST QUERIES")
print(f"{'─'*70}")

for query in test_queries:
    print(f"\n📝 Query: '{query}'")
    results = model.predict(query, top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['text']}")
        print(f"      Score: {result['score']:.3f} | Confidence: {result['confidence']:.1%}")

print(f"\n{'='*70}\n")