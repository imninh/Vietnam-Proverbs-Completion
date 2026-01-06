"""
SEMANTIC FILL-IN-THE-BLANK MODEL
File: backend/models/semantic_fill_blank.py

Inspired by SP-GPT2 paper (2110.15723v1)
Sử dụng sentence embeddings để tìm câu có semantic similarity cao nhất
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import pickle
from pathlib import Path


class SemanticFillBlankModel:
    """
    Semantic-based Fill-in-Blank model
    
    Cách hoạt động (Inspired by SP-GPT2):
    1. Encode tất cả câu trong database thành semantic vectors
    2. Khi có input, encode input thành vector
    3. Tính cosine similarity giữa input và tất cả câu
    4. Trả về top-k câu có semantic similarity cao nhất
    
    Khác với Retrieval (TF-IDF):
    - Retrieval: Dựa trên từ khóa (keyword matching)
    - Semantic: Dựa trên ý nghĩa (semantic meaning)
    
    VD:
    - Input: "ăn quả nhớ"
    - Retrieval có thể match: "ăn cháo đá bát" (có từ "ăn")
    - Semantic sẽ match: "uống nước nhớ nguồn" (cùng ý nghĩa biết ơn)
    """
    
    def __init__(self, model_name='keepitreal/vietnamese-sbert'):
        """
        Args:
            model_name: Pre-trained sentence transformer model
                       'keepitreal/vietnamese-sbert' - Model tiếng Việt tốt
        """
        print(f"🔄 Loading SentenceTransformer: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Lỗi load model: {e}")
            print(f"💡 Fallback sang 'paraphrase-multilingual-MiniLM-L12-v2'")
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        self.database = []  # List of sentences
        self.embeddings = None  # Numpy array of embeddings
        self.is_trained = False
    
    def train(self, train_data):
        """
        Encode tất cả câu trong database
        
        Args:
            train_data: List of dicts [{'full': '...', ...}]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING SEMANTIC FILL-BLANK MODEL")
        print(f"{'─'*60}")
        
        # Lấy unique sentences
        seen = set()
        for item in train_data:
            sentence = item['full']
            if sentence not in seen:
                self.database.append(sentence)
                seen.add(sentence)
        
        print(f"📊 Database: {len(self.database)} sentences")
        
        # Encode tất cả câu (có thể mất vài phút)
        print(f"🔄 Encoding sentences (có thể mất 1-2 phút)...")
        self.embeddings = self.model.encode(
            self.database,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Embeddings shape: {self.embeddings.shape}")
        print(f"✓ Vector dimension: {self.embeddings.shape[1]}")
        
        self.is_trained = True
    
    def semantic_similarity(self, query_embedding, database_embeddings):
        """
        Tính cosine similarity giữa query và database
        
        Returns:
            Numpy array of similarity scores
        """
        # Cosine similarity = dot product (vì vectors đã normalized)
        similarities = np.dot(database_embeddings, query_embedding)
        return similarities
    
    def predict_multiple(self, partial_input, top_k=3, min_similarity=0.3):
        """
        Trả về top-k candidates dựa trên semantic similarity
        
        Args:
            partial_input: Input string
            top_k: Số candidates
            min_similarity: Ngưỡng similarity tối thiểu
        
        Returns:
            List of dicts [{'text': '...', 'confidence': 0.9, 'model': 'semantic'}]
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
        
        # Encode query
        query_embedding = self.model.encode(
            partial_input.lower(),
            convert_to_numpy=True
        )
        
        # Tính similarity
        similarities = self.semantic_similarity(query_embedding, self.embeddings)
        
        # Lấy top-k indices
        top_indices = np.argsort(similarities)[-top_k*2:][::-1]  # Lấy nhiều hơn để filter
        
        candidates = []
        
        for idx in top_indices:
            similarity = float(similarities[idx])
            
            # Filter theo threshold
            if similarity < min_similarity:
                continue
            
            # Map similarity → confidence
            # Semantic similarity thường cao hơn TF-IDF
            confidence = min(0.99, similarity * 1.1)
            
            candidates.append({
                'text': self.database[idx],
                'confidence': round(confidence, 3),
                'model': 'semantic',
                'similarity': round(similarity, 3),
                'method': 'sentence_embedding'
            })
            
            if len(candidates) >= top_k:
                break
        
        # Fallback nếu không tìm thấy
        if not candidates:
            # Lấy top 1 dù similarity thấp
            best_idx = np.argmax(similarities)
            candidates = [{
                'text': self.database[best_idx],
                'confidence': 0.2,
                'model': 'semantic',
                'similarity': round(float(similarities[best_idx]), 3),
                'method': 'fallback'
            }]
        
        return candidates
    
    def predict(self, partial_input):
        """Wrapper trả về 1 kết quả"""
        candidates = self.predict_multiple(partial_input, top_k=1)
        return candidates[0]['text'] if candidates else partial_input
    
    def evaluate(self, test_data):
        """Đánh giá model"""
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING SEMANTIC FILL-BLANK MODEL")
        print(f"{'─'*60}")
        
        exact_correct = 0
        top3_correct = 0
        total = len(test_data)
        similarities = []
        
        for item in test_data:
            candidates = self.predict_multiple(item['input'], top_k=3)
            
            # Exact match
            if candidates and candidates[0]['text'] == item['full']:
                exact_correct += 1
            
            # Top-3
            if candidates:
                top3_texts = [c['text'] for c in candidates]
                if item['full'] in top3_texts:
                    top3_correct += 1
            
            # Similarity
            if candidates:
                similarities.append(candidates[0]['similarity'])
        
        exact_acc = exact_correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        avg_sim = sum(similarities) / len(similarities) if similarities else 0
        
        print(f"Test samples: {total}")
        print(f"Exact matches: {exact_correct} ({exact_acc:.1%})")
        print(f"Top-3 matches: {top3_correct} ({top3_acc:.1%})")
        print(f"Avg similarity: {avg_sim:.3f}")
        
        return {
            'exact_accuracy': exact_acc,
            'top3_accuracy': top3_acc,
            'avg_similarity': avg_sim,
            'exact_correct': exact_correct,
            'top3_correct': top3_correct,
            'total': total
        }
    
    def save(self, file_path):
        """Lưu model"""
        data = {
            'database': self.database,
            'embeddings': self.embeddings,
            'model_name': 'keepitreal/vietnamese-sbert'
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.database = data['database']
        self.embeddings = data['embeddings']
        self.is_trained = True
        
        print(f"✓ Model loaded from {file_path}")


# ========== TRAINING SCRIPT ==========
def train_semantic_model():
    """Script train và test"""
    
    print("\n" + "="*70)
    print("🚀 SEMANTIC FILL-BLANK MODEL TRAINING")
    print("="*70)
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data...")
    
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Test:  {len(test_data)} samples")
    
    # Train
    model = SemanticFillBlankModel()
    model.train(train_data)
    
    # Test predictions
    print(f"\n{'─'*60}")
    print("🧪 TEST PREDICTIONS")
    print(f"{'─'*60}")
    
    test_inputs = [
        "ăn",
        "ăn quả",
        "ăn quả nhớ",
        "có công",
        "gần mực",
        "học thầy",
        "uống nước"  # Test semantic: giống "ăn quả nhớ" về ý nghĩa
    ]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = model.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. {cand['text']}")
            print(f"      📊 Confidence: {cand['confidence']:.1%} | Similarity: {cand['similarity']:.3f}")
    
    # Evaluate
    metrics = model.evaluate(test_data[:100])
    
    # Save
    model_path = MODEL_DIR / "semantic_model.pkl"
    model.save(model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Database size: {len(model.database):,} sentences")
    print(f"   • Embedding dim: {model.embeddings.shape[1]}")
    print(f"   • Exact accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   • Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"   • Avg similarity: {metrics['avg_similarity']:.3f}")
    print(f"   • Model saved: {model_path}")
    print()


# ========== MAIN ==========
if __name__ == "__main__":
    train_semantic_model()