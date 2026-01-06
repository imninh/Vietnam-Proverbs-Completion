from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pickle
from pathlib import Path


class RetrievalModel:
    """
    Retrieval-based model sử dụng TF-IDF + Cosine Similarity
    
    Cách hoạt động:
    1. Vectorize tất cả câu trong dataset thành TF-IDF vectors
    2. Khi có input, vectorize input
    3. Tính cosine similarity giữa input và tất cả câu
    4. Trả về top-k câu có similarity cao nhất
    
    Ưu điểm:
    - Luôn trả về câu hoàn chỉnh (không generate)
    - Xử lý tốt input mơ hồ
    - Nhanh (vectorize 1 lần, inference nhanh)
    
    Nhược điểm:
    - Không tạo câu mới (chỉ retrieve)
    - Phụ thuộc vào dataset có đủ đa dạng
    """
    
    def __init__(self, ngram_range=(1, 3), max_features=5000):
        """
        Args:
            ngram_range: (min, max) n-grams để extract
                        (1,3) = unigrams + bigrams + trigrams
            max_features: Số features tối đa cho TF-IDF
        """
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents=None,  # Giữ nguyên dấu tiếng Việt
            token_pattern=r'\b\w+\b'
        )
        
        self.database = []  # List of full sentences
        self.vectors = None  # TF-IDF matrix
        self.is_trained = False
    
    def train(self, train_data):
        """
        Huấn luyện model = Xây dựng database + vectorize
        
        Args:
            train_data: List of dicts [{'full': '...', ...}]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING RETRIEVAL MODEL")
        print(f"{'─'*60}")
        
        # Lấy unique sentences
        seen = set()
        for item in train_data:
            sentence = item['full']
            if sentence not in seen:
                self.database.append(sentence)
                seen.add(sentence)
        
        print(f"📊 Database: {len(self.database)} unique sentences")
        
        # Vectorize tất cả câu
        print(f"🔄 Vectorizing with TF-IDF...")
        self.vectors = self.vectorizer.fit_transform(self.database)
        
        print(f"✓ Vector shape: {self.vectors.shape}")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        # Phân tích top features
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"\n📝 Top 10 features (từ quan trọng nhất):")
        
        # Tính IDF scores
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[:10]  # IDF thấp = xuất hiện nhiều
        
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i:2d}. '{feature_names[idx]}' (IDF: {idf_scores[idx]:.2f})")
        
        self.is_trained = True
    
    def retrieve(self, query, top_k=10):
        """
        Tìm top-k câu giống nhất
        
        Args:
            query: Input string
            top_k: Số câu trả về
        
        Returns:
            List of (sentence, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query.lower()])
        
        # Tính similarity với tất cả câu
        similarities = cosine_similarity(query_vec, self.vectors)[0]
        
        # Lấy top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Trả về (sentence, score)
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Chỉ lấy nếu có similarity > 0
                results.append((self.database[idx], float(similarities[idx])))
        
        return results
    
    def predict_multiple(self, partial_input, top_k=3, min_similarity=0.05):
        """
        Trả về top-k candidates cho API
        
        Args:
            partial_input: Input string
            top_k: Số candidates
            min_similarity: Ngưỡng similarity tối thiểu
        
        Returns:
            List of dicts [{'text': '...', 'confidence': 0.9, 'model': 'retrieval'}]
        """
        # Retrieve top candidates
        retrieved = self.retrieve(partial_input, top_k=top_k*2)  # Lấy nhiều hơn để filter
        
        candidates = []
        
        for sentence, similarity in retrieved:
            # Filter theo threshold
            if similarity < min_similarity:
                continue
            
            # Map similarity → confidence (0-1)
            # Similarity thường trong khoảng 0.1-0.8
            # Scale lên để confidence rõ ràng hơn
            confidence = min(0.99, similarity * 1.2)
            
            candidates.append({
                'text': sentence,
                'confidence': round(confidence, 3),
                'model': 'retrieval',
                'similarity': round(similarity, 3)
            })
            
            if len(candidates) >= top_k:
                break
        
        # Fallback nếu không tìm thấy gì
        if not candidates:
            import random
            random_sentence = random.choice(self.database) if self.database else partial_input
            candidates = [{
                'text': random_sentence,
                'confidence': 0.05,
                'model': 'retrieval',
                'similarity': 0.0,
                'method': 'fallback'
            }]
        
        return candidates
    
    def predict(self, partial_input):
        """
        Trả về 1 kết quả tốt nhất (wrapper)
        """
        candidates = self.predict_multiple(partial_input, top_k=1)
        return candidates[0]['text'] if candidates else partial_input
    
    def evaluate(self, test_data):
        """
        Đánh giá model trên test set
        
        Metrics:
        - Exact match accuracy
        - Top-3 accuracy (câu đúng có trong top 3 không)
        - Average similarity score
        """
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING RETRIEVAL MODEL")
        print(f"{'─'*60}")
        
        exact_correct = 0
        top3_correct = 0
        total = len(test_data)
        similarities = []
        
        for item in test_data:
            # Predict
            candidates = self.predict_multiple(item['input'], top_k=3)
            
            # Check exact match
            if candidates[0]['text'] == item['full']:
                exact_correct += 1
            
            # Check top-3
            top3_texts = [c['text'] for c in candidates]
            if item['full'] in top3_texts:
                top3_correct += 1
            
            # Similarity score
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
            'vectorizer': self.vectorizer,
            'database': self.database,
            'vectors': self.vectors
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.database = data['database']
        self.vectors = data['vectors']
        self.is_trained = True
        
        print(f"✓ Model loaded from {file_path}")


# ========== SCRIPT TRAINING ==========
def train_retrieval_model():
    """Script để train và test model"""
    
    print("\n" + "="*70)
    print("🚀 RETRIEVAL MODEL TRAINING")
    print("="*70)
    
    # Đường dẫn
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
    
    # Train model
    model = RetrievalModel(ngram_range=(1, 3), max_features=5000)
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
        "có công mài sắt",
        "gần mực",
        "học thầy không"
    ]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = model.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. {cand['text']}")
            print(f"      📊 Confidence: {cand['confidence']:.1%} | Similarity: {cand['similarity']:.3f}")
    
    # Evaluate
    metrics = model.evaluate(test_data[:100])
    
    # Save model
    model_path = MODEL_DIR / "retrieval_model.pkl"
    model.save(model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Database size: {len(model.database):,} sentences")
    print(f"   • Vector dimension: {model.vectors.shape[1]:,}")
    print(f"   • Exact accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   • Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"   • Avg similarity: {metrics['avg_similarity']:.3f}")
    print(f"   • Model saved: {model_path}")
    print()


# ========== MAIN ==========
if __name__ == "__main__":
    train_retrieval_model()