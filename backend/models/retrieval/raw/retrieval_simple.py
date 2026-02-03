"""
RETRIEVAL MODEL - PHIÊN BẢN ĐƠN GIẢN
File: retrieval_simple.py

Mô tả:
- Sử dụng BM25 ranking (tốt hơn TF-IDF cosine)
- Prefix matching boost: +10 điểm nếu câu bắt đầu chính xác bằng query
- Trả về câu đầy đủ (không cắt input/target)

Cách dùng:
1. Train: python retrieval_simple.py
2. Trong code khác:
   from retrieval_simple import RetrievalModel
   model = RetrievalModel()
   model.load('retrieval_model.pkl')
   result = model.predict("ăn quả nhớ")
   # Output: "ăn quả nhớ kẻ trồng cây. uống nước nhớ người đào giếng."
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import json
import pickle
from pathlib import Path


class RetrievalModel:
    """
    Retrieval model sử dụng BM25 với prefix matching boost
    
    Ưu điểm:
    - BM25: Tốt hơn TF-IDF cho text retrieval
    - Prefix boost: Ưu tiên câu bắt đầu giống query
    - Đơn giản: Không cần training phức tạp
    - Nhanh: Chỉ cần vector search
    """
    
    def __init__(self, analyzer='char_wb', ngram_range=(2, 4), max_features=10000, 
                 bm25_k1=1.5, bm25_b=0.75):
        """
        Args:
            analyzer: 'char_wb' = character n-grams trong word boundaries
            ngram_range: (2, 4) = bigram đến 4-gram
            max_features: Kích thước vocabulary
            bm25_k1: TF saturation parameter (1.2-2.0)
            bm25_b: Length normalization (0.75 typical)
        """
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents=None,  # Giữ dấu tiếng Việt
        )
        
        self.database = []  # Danh sách câu đầy đủ
        self.term_freqs = None  # Document-term matrix
        self.idf = None  # IDF values
        self.doc_lengths = None  # Độ dài mỗi doc
        self.avg_doc_len = 0
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.is_trained = False
    
    def train(self, train_data):
        """
        Train model
        
        Args:
            train_data: List of strings (danh sách câu đầy đủ)
                       VD: ["ăn quả nhớ kẻ trồng cây", "có công mài sắt", ...]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING RETRIEVAL MODEL (BM25)")
        print(f"{'─'*60}")
        
        # Lọc trùng lặp
        seen = set()
        for sentence in train_data:
            if sentence not in seen:
                self.database.append(sentence)
                seen.add(sentence)
        
        print(f"📊 Database: {len(self.database):,} câu unique")
        
        # Vectorize để lấy term frequencies
        count_vec = CountVectorizer(
            analyzer=self.vectorizer.analyzer,
            ngram_range=self.vectorizer.ngram_range,
            max_features=self.vectorizer.max_features,
            lowercase=True,
            strip_accents=None,
        )
        self.term_freqs = count_vec.fit_transform(self.database)
        
        # Tính IDF
        self.idf = self.vectorizer.fit(self.database).idf_
        
        # Tính độ dài doc (số từ)
        self.doc_lengths = np.array([len(doc.split()) for doc in self.database])
        self.avg_doc_len = np.mean(self.doc_lengths) if len(self.doc_lengths) > 0 else 0
        
        print(f"✓ Term matrix shape: {self.term_freqs.shape}")
        print(f"✓ Vocabulary size: {len(count_vec.vocabulary_):,}")
        print(f"✓ Average doc length: {self.avg_doc_len:.1f} từ")
        
        self.is_trained = True
    
    def compute_bm25_scores(self, query):
        """
        Tính BM25 scores cho tất cả documents
        
        BM25 formula:
        score(d, q) = Σ IDF(t) × [TF(t,d) × (k1 + 1)] / [TF(t,d) + k1 × (1 - b + b × len(d)/avg_len)]
        """
        if not self.is_trained:
            raise ValueError("Model chưa được train!")
        
        # Transform query
        from sklearn.feature_extraction.text import CountVectorizer
        count_vec = CountVectorizer(vocabulary=self.vectorizer.vocabulary_)
        query_tf = count_vec.fit_transform([query]).toarray()[0]
        
        scores = np.zeros(len(self.database))
        
        # Tính score cho mỗi term trong query
        for term_idx in np.nonzero(query_tf)[0]:
            # TF trong docs cho term này
            tf_docs = self.term_freqs[:, term_idx].toarray().flatten()
            
            # IDF của term
            idf_term = self.idf[term_idx]
            
            # BM25 score
            numerator = tf_docs * (self.bm25_k1 + 1)
            denominator = tf_docs + self.bm25_k1 * (
                1 - self.bm25_b + self.bm25_b * (self.doc_lengths / self.avg_doc_len)
            )
            term_scores = idf_term * (numerator / denominator)
            
            scores += term_scores
        
        return scores
    
    def retrieve(self, query, top_k=10):
        """
        Tìm top-k câu giống nhất với query
        
        Args:
            query: Input string (VD: "ăn quả")
            top_k: Số lượng kết quả trả về
        
        Returns:
            List of (sentence, score) tuples
        """
        # Tính BM25 scores
        scores = self.compute_bm25_scores(query.lower())
        
        # Thêm prefix matching boost
        query_words = query.lower().split()
        query_len = len(query_words)
        
        for i, sentence in enumerate(self.database):
            sent_words = sentence.lower().split()
            # Nếu sentence bắt đầu chính xác bằng query → boost
            if sent_words[:query_len] == query_words:
                scores[i] += 10.0
        
        # Lấy top-k
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.database[idx], float(scores[idx])))
        
        return results
    
    def predict(self, partial_input, top_k=1):
        """
        Dự đoán câu đầy đủ từ input một phần
        
        Args:
            partial_input: Input string (VD: "ăn quả")
            top_k: Số lượng kết quả (mặc định 1 = chỉ trả về câu tốt nhất)
        
        Returns:
            Nếu top_k=1: String (câu đầy đủ)
            Nếu top_k>1: List of dicts với info chi tiết
        """
        retrieved = self.retrieve(partial_input, top_k=top_k)
        
        if not retrieved:
            # Fallback: trả về random
            import random
            return random.choice(self.database) if self.database else partial_input
        
        if top_k == 1:
            return retrieved[0][0]  # Chỉ trả về text
        else:
            # Trả về list với details
            max_score = max([s[1] for s in retrieved])
            results = []
            for sentence, score in retrieved:
                confidence = min(0.99, score / max_score) if max_score > 0 else 0.05
                results.append({
                    'text': sentence,
                    'score': round(score, 3),
                    'confidence': round(confidence, 3)
                })
            return results
    
    def predict_multiple(self, partial_input, top_k=3):
        """
        Trả về nhiều candidates cho API
        Giống predict nhưng luôn trả về list
        """
        return self.predict(partial_input, top_k=top_k)
    
    def evaluate(self, test_data, test_queries=None):
        """
        Đánh giá model
        
        Args:
            test_data: List of full sentences để test
            test_queries: (Optional) List of tuples (query, expected_sentence)
                         Nếu None, sẽ tự tạo queries từ test_data
        """
        print(f"\n{'─'*60}")
        print(f"📊 ĐÁNH GIÁ MODEL")
        print(f"{'─'*60}")
        
        # Nếu không có test_queries, tự tạo
        if test_queries is None:
            test_queries = []
            for sentence in test_data[:50]:  # Test 50 câu đầu
                words = sentence.split()
                if len(words) >= 3:
                    # Lấy 2-4 từ đầu làm query
                    query_len = min(len(words) // 2, 4)
                    query = ' '.join(words[:query_len])
                    test_queries.append((query, sentence))
        
        print(f"Test queries: {len(test_queries)}")
        
        exact_correct = 0
        top3_correct = 0
        
        for query, expected in test_queries:
            # Predict top-3
            results = self.predict(query, top_k=3)
            
            # Check exact match (top-1)
            if results[0]['text'] == expected:
                exact_correct += 1
            
            # Check top-3
            top3_texts = [r['text'] for r in results]
            if expected in top3_texts:
                top3_correct += 1
        
        total = len(test_queries)
        exact_acc = exact_correct / total if total > 0 else 0
        top3_acc = top3_correct / total if total > 0 else 0
        
        print(f"✓ Exact match (top-1): {exact_correct}/{total} = {exact_acc:.1%}")
        print(f"✓ Top-3 match:         {top3_correct}/{total} = {top3_acc:.1%}")
        
        return {
            'exact_accuracy': exact_acc,
            'top3_accuracy': top3_acc,
            'exact_correct': exact_correct,
            'top3_correct': top3_correct,
            'total': total
        }
    
    def save(self, file_path):
        """Lưu model"""
        data = {
            'vectorizer': self.vectorizer,
            'database': self.database,
            'term_freqs': self.term_freqs,
            'idf': self.idf,
            'doc_lengths': self.doc_lengths,
            'avg_doc_len': self.avg_doc_len,
            'bm25_k1': self.bm25_k1,
            'bm25_b': self.bm25_b
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved: {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.database = data['database']
        self.term_freqs = data['term_freqs']
        self.idf = data['idf']
        self.doc_lengths = data['doc_lengths']
        self.avg_doc_len = data['avg_doc_len']
        self.bm25_k1 = data['bm25_k1']
        self.bm25_b = data['bm25_b']
        self.is_trained = True
        
        print(f"✓ Model loaded: {file_path}")


# ========== SCRIPT TRAINING ==========
def train_and_test():
    """Script chính để train và test model"""
    
    print("\n" + "="*70)
    print("🚀 RETRIEVAL MODEL TRAINING")
    print("="*70)
    
    # Đường dẫn - BẠN CẦN SỬA CHỖ NÀY
    BASE_DIR = Path(__file__).parent.parent
    TRAIN_FILE = BASE_DIR / "data" / "processed" / "train.json"
    TEST_FILE = BASE_DIR / "data" / "processed" / "test.json"
    MODEL_FILE = BASE_DIR / "trained_models" / "retrieval_model.pkl"
    # Load data
    print(f"\n📂 Loading data...")
    
    try:
        with open(TRAIN_FILE, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(TEST_FILE, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file: {e}")
        print(f"\n💡 Hãy chạy các script theo thứ tự:")
        print(f"   1. python 2_clean_data_simple.py")
        print(f"   2. python 3_create_splits_simple.py")
        print(f"   3. python retrieval_simple.py")
        return
    
    print(f"✓ Train: {len(train_data):,} câu")
    print(f"✓ Test:  {len(test_data):,} câu")
    
    # Train model
    model = RetrievalModel()
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
        
        # Predict top-3
        results = model.predict(inp, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['text']}")
            print(f"      📊 Score: {result['score']:.3f} | Confidence: {result['confidence']:.1%}")
    
    # Evaluate
    metrics = model.evaluate(test_data)
    
    # Save model
    model.save(MODEL_FILE)
    
    print(f"\n{'='*70}")
    print("✅ HOÀN THÀNH!")
    print("="*70)
    print(f"\n📊 Tóm tắt:")
    print(f"   • Database: {len(model.database):,} câu")
    print(f"   • Vocabulary: {model.term_freqs.shape[1]:,} features")
    print(f"   • Exact accuracy: {metrics['exact_accuracy']:.1%}")
    print(f"   • Top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"   • Model saved: {MODEL_FILE}")
    
    print(f"\n💡 Cách sử dụng model:")
    print(f"   from retrieval_simple import RetrievalModel")
    print(f"   model = RetrievalModel()")
    print(f"   model.load('{MODEL_FILE}')")
    print(f"   result = model.predict('ăn quả nhớ')")
    print(f"   print(result)\n")


# ========== MAIN ==========
if __name__ == "__main__":
    train_and_test()