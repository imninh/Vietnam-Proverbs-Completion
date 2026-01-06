from collections import defaultdict, Counter
import json
import pickle
from pathlib import Path


class NgramModel:
    """
    Mô hình N-gram để dự đoán từ tiếp theo
    
    Cách hoạt động:
    1. Training: Đếm tần suất xuất hiện của các n-grams
    2. Prediction: Tìm từ có xác suất cao nhất sau context
    
    VD với trigram (n=3):
    - Context: ("ăn", "quả")
    - Từ tiếp theo có thể là: "nhớ" (80%), "ngọt" (20%)
    """
    
    def __init__(self, n=3):
        """
        Args:
            n: Độ dài context (3 = trigram là tốt nhất cho tiếng Việt)
        """
        self.n = n
        
        # Dictionary lưu n-grams
        # Key: tuple của n-1 từ (context)
        # Value: Counter của từ tiếp theo và tần suất
        # VD: {('ăn', 'quả'): Counter({'nhớ': 10, 'ngọt': 2})}
        self.ngrams = defaultdict(Counter)
        
        # Lưu toàn bộ câu để fallback khi không tìm thấy
        self.full_sentences = []
        
        # Thống kê
        self.vocab_size = 0
        self.total_ngrams = 0
    
    def train(self, train_data):
        """
        Huấn luyện mô hình từ dataset
        
        Args:
            train_data: List of dicts [{'full': '...', 'input': '...', 'target': '...'}]
        """
        print(f"\n{'─'*60}")
        print(f"🔄 TRAINING N-GRAM MODEL (n={self.n})")
        print(f"{'─'*60}")
        
        # Lưu tất cả câu đầy đủ
        seen_sentences = set()
        for item in train_data:
            sentence = item['full']
            if sentence not in seen_sentences:
                self.full_sentences.append(sentence)
                seen_sentences.add(sentence)
        
        print(f"📊 Dataset: {len(train_data)} samples, {len(self.full_sentences)} unique sentences")
        
        # Đếm n-grams
        vocabulary = set()
        
        for item in train_data:
            words = item['full'].split()
            vocabulary.update(words)
            
            # Tạo n-grams từ câu
            # VD: "ăn quả nhớ kẻ" với n=3
            # → contexts: [("ăn", "quả"), ("quả", "nhớ")]
            # → next_words: ["nhớ", "kẻ"]
            
            for i in range(len(words) - self.n):
                # Lấy n-1 từ làm context
                context = tuple(words[i:i+self.n-1])
                
                # Từ tiếp theo
                next_word = words[i+self.n-1]
                
                # Đếm
                self.ngrams[context][next_word] += 1
                self.total_ngrams += 1
        
        self.vocab_size = len(vocabulary)
        
        print(f"✓ Vocabulary size: {self.vocab_size:,} từ")
        print(f"✓ Total n-grams: {self.total_ngrams:,}")
        print(f"✓ Unique contexts: {len(self.ngrams):,}")
        
        # Thống kê phân bố
        context_sizes = [sum(counter.values()) for counter in self.ngrams.values()]
        avg_size = sum(context_sizes) / len(context_sizes) if context_sizes else 0
        
        print(f"✓ Avg words per context: {avg_size:.1f}")
        
        # Ví dụ n-grams
        print(f"\n📝 Ví dụ n-grams học được:")
        for i, (context, counter) in enumerate(list(self.ngrams.items())[:3]):
            context_str = ' '.join(context)
            top_3 = counter.most_common(3)
            print(f"   {i+1}. '{context_str}' →")
            for word, count in top_3:
                prob = count / sum(counter.values())
                print(f"      • '{word}' ({prob:.1%}, {count} lần)")
    
    def predict_next_word(self, context_words):
        """
        Dự đoán 1 từ tiếp theo
        
        Args:
            context_words: List of words (VD: ["ăn", "quả"])
        
        Returns:
            (word, confidence) hoặc (None, 0) nếu không tìm thấy
        """
        # Lấy n-1 từ cuối làm context
        context = tuple(context_words[-(self.n-1):])
        
        if context not in self.ngrams:
            return None, 0.0
        
        # Tìm từ xuất hiện nhiều nhất
        counter = self.ngrams[context]
        most_common_word, count = counter.most_common(1)[0]
        
        # Tính confidence (xác suất)
        total_count = sum(counter.values())
        confidence = count / total_count
        
        return most_common_word, confidence
    
    def predict(self, partial_input, max_words=15):
        """
        Dự đoán hoàn thiện câu (generate từng từ)
        
        Args:
            partial_input: Chuỗi input (VD: "ăn quả nhớ")
            max_words: Số từ tối đa để generate
        
        Returns:
            Câu hoàn chỉnh
        """
        words = partial_input.strip().split()
        result = words.copy()
        
        # Generate từng từ
        for _ in range(max_words):
            next_word, confidence = self.predict_next_word(result)
            
            if next_word is None:
                # Không tìm thấy → dừng
                break
            
            result.append(next_word)
            
            # Dừng nếu câu đã đủ dài (heuristic)
            if len(result) >= len(words) + 8:
                break
            
            # Dừng nếu confidence quá thấp (từ hiếm)
            if confidence < 0.1:
                break
        
        return ' '.join(result)
    
    def predict_multiple(self, partial_input, top_k=3):
        """
        Trả về nhiều candidates (dùng cho API) - IMPROVED VERSION
        
        Args:
            partial_input: Chuỗi input
            top_k: Số candidates trả về
        
        Returns:
            List of dicts [{'text': '...', 'confidence': 0.9, 'model': 'ngram'}]
        """
        # Normalize input
        words = partial_input.strip().lower().split()
        input_text = ' '.join(words)
        
        candidates = []
        
        # STRATEGY 1: Exact prefix match
        for sentence in self.full_sentences:
            sentence_lower = sentence.lower()
            
            # Kiểm tra câu có bắt đầu bằng input không
            if sentence_lower.startswith(input_text):
                # Tính confidence dựa trên độ overlap
                overlap_ratio = len(input_text) / len(sentence_lower)
                confidence = min(0.95, overlap_ratio + 0.2)  # Boost confidence
                
                candidates.append({
                    'text': sentence,
                    'confidence': round(confidence, 3),
                    'model': 'ngram',
                    'method': 'exact_match'
                })
        
        # STRATEGY 2: Fuzzy match (chứa các từ của input)
        if len(candidates) < top_k:
            for sentence in self.full_sentences:
                sentence_lower = sentence.lower()
                
                # Kiểm tra các từ input có trong câu không
                words_in_sentence = sum(1 for word in words if word in sentence_lower)
                match_ratio = words_in_sentence / len(words) if words else 0
                
                # Chỉ lấy nếu match >= 50% và chưa có trong candidates
                if match_ratio >= 0.5 and sentence not in [c['text'] for c in candidates]:
                    confidence = match_ratio * 0.6  # Lower confidence
                    
                    candidates.append({
                        'text': sentence,
                        'confidence': round(confidence, 3),
                        'model': 'ngram',
                        'method': 'fuzzy_match'
                    })
        
        # STRATEGY 3: Generate với n-gram
        if len(candidates) < top_k:
            generated = self.predict(partial_input)
            
            # Chỉ thêm nếu khác với input và chưa có
            if generated.lower() != input_text and generated not in [c['text'] for c in candidates]:
                candidates.append({
                    'text': generated,
                    'confidence': 0.4,
                    'model': 'ngram',
                    'method': 'generated'
                })
        
        # Sort theo confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Lấy top-k
        candidates = candidates[:top_k]
        
        # STRATEGY 4: Fallback nếu vẫn không có
        if not candidates:
            import random
            random_sentence = random.choice(self.full_sentences) if self.full_sentences else partial_input
            candidates = [{
                'text': random_sentence,
                'confidence': 0.1,
                'model': 'ngram',
                'method': 'fallback'
            }]
        
        return candidates
    
    def evaluate(self, test_data):
        """
        Đánh giá model trên test set
        
        Args:
            test_data: List of dicts [{'full': '...', 'input': '...', 'target': '...'}]
        
        Returns:
            Dict với các metrics
        """
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING N-GRAM MODEL")
        print(f"{'─'*60}")
        
        correct = 0
        total = len(test_data)
        
        for item in test_data:
            predicted = self.predict(item['input'])
            
            # Exact match
            if predicted == item['full']:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print(f"Test samples: {total}")
        print(f"Exact matches: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def save(self, file_path):
        """Lưu model"""
        data = {
            'n': self.n,
            'ngrams': dict(self.ngrams),
            'full_sentences': self.full_sentences,
            'vocab_size': self.vocab_size
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        """Load model"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        self.n = data['n']
        self.ngrams = defaultdict(Counter, data['ngrams'])
        self.full_sentences = data['full_sentences']
        self.vocab_size = data['vocab_size']
        
        print(f"✓ Model loaded from {file_path}")


# ========== SCRIPT TRAINING ==========
def train_ngram_model():
    """Script để train và test model"""
    
    print("\n" + "="*70)
    print("🚀 N-GRAM MODEL TRAINING")
    print("="*70)
    
    # Đường dẫn
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    # Tạo thư mục models nếu chưa có
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
    model = NgramModel(n=3)  # Trigram
    model.train(train_data)
    
    # Test predictions
    print(f"\n{'─'*60}")
    print("🧪 TEST PREDICTIONS")
    print(f"{'─'*60}")
    
    test_inputs = [
        "ăn quả",
        "có công mài sắt",
        "gần mực"
    ]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = model.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. {cand['text']}")
            print(f"      Confidence: {cand['confidence']:.1%} | Method: {cand['method']}")
    
    # Evaluate
    metrics = model.evaluate(test_data[:100])  # Test trên 100 samples
    
    # Save model
    model_path = MODEL_DIR / "ngram_model.pkl"
    model.save(model_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   • Vocabulary: {model.vocab_size:,} words")
    print(f"   • N-grams: {model.total_ngrams:,}")
    print(f"   • Accuracy: {metrics['accuracy']:.2%}")
    print(f"   • Model saved: {model_path}")
    print()


# ========== MAIN ==========
if __name__ == "__main__":
    train_ngram_model()