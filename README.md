# 🎯 Ca Dao & Tục Ngữ Autocomplete

> Hệ thống tự động hoàn thiện câu ca dao và tục ngữ Việt Nam sử dụng Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Development-yellow.svg)]()

---

## 📖 Table of Contents

- [Giới thiệu](#-giới-thiệu)
- [Demo](#-demo)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Models](#-models)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Kết quả](#-kết-quả)
- [API Documentation](#-api-documentation)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Giới thiệu

### Vấn đề

Ca dao và tục ngữ Việt Nam là di sản văn hóa quý báu, nhưng nhiều người trẻ không nhớ đủ hoặc nhớ sai. Project này giúp:

- ✅ Gợi ý hoàn thiện câu ca dao/tục ngữ khi nhập một phần
- ✅ Giáo dục và bảo tồn văn hóa truyền thống
- ✅ Hỗ trợ học sinh, giáo viên, người yêu văn hóa

### Giải pháp

Hệ thống sử dụng **ensemble của nhiều ML models** để đưa ra gợi ý thông minh:

```
User input: "ăn quả"
↓
System output:
  1. 🟢 [95%] ăn quả nhớ kẻ trồng cây
  2. 🟡 [70%] gieo nhân nào gặt quả nấy
  3. 🟡 [65%] ăn cháo đá bát
```

---

## 🎬 Demo

### Interactive Terminal

```bash
python backend/models/interactive_autocomplete.py
```

![Demo GIF](docs/demo.gif)

### Web Interface (Coming Soon)

```bash
cd frontend
npm start
```

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────┐
│                   USER INPUT                         │
│              "ăn quả" / "có công"                    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            SMART FALLBACK SYSTEM                     │
│  (Coordinate multiple strategies & models)           │
└─────────────────────┬───────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Strategy 1  │ │  Strategy 2  │ │  Strategy 3  │
│ Exact Prefix │ │ Fuzzy Match  │ │   Semantic   │
│  (95% conf)  │ │  (85% conf)  │ │  (70% conf)  │
└──────────────┘ └──────────────┘ └──────────────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              WEIGHTED VOTING                         │
│     (Combine scores from all strategies)             │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              TOP-K SUGGESTIONS                       │
│   1. [95%] ăn quả nhớ kẻ trồng cây                  │
│   2. [70%] gieo nhân nào gặt quả nấy                │
│   3. [65%] ăn cháo đá bát                           │
└─────────────────────────────────────────────────────┘
```

### System Workflow

1. **Input Processing**: Normalize và tokenize user input
2. **Multi-Strategy Search**: 
   - Exact prefix matching (fastest, most accurate)
   - Fuzzy matching (handle typos)
   - Semantic similarity (understand meaning)
   - Keyword retrieval (fallback)
   - Popular sentences (last resort)
3. **Weighted Voting**: Combine scores with confidence calibration
4. **Re-ranking**: Sort by confidence and strategy priority
5. **Output**: Return top-K suggestions with confidence scores

---

## 🤖 Models

### 1. **Retrieval Model** (TF-IDF + Cosine Similarity)

**Accuracy**: 55% | **Speed**: ⚡⚡⚡ Fast

```python
from backend.models.retrieval import RetrievalModel

model = RetrievalModel()
model.train(train_data)

# Usage
candidates = model.predict_multiple("ăn quả", top_k=3)
# → [{'text': '...', 'confidence': 0.85, 'similarity': 0.75}]
```

**Ưu điểm:**
- ✅ Nhanh (pre-computed TF-IDF vectors)
- ✅ Luôn trả về câu hoàn chỉnh
- ✅ Tốt cho keyword matching

**Nhược điểm:**
- ❌ Không hiểu semantic (ý nghĩa)
- ❌ Phụ thuộc vào exact keywords

---

### 2. **Semantic Model** (Sentence Embeddings)

**Accuracy**: 50-55% | **Speed**: ⚡⚡ Medium

```python
from backend.models.semantic_fill_blank import SemanticFillBlankModel

model = SemanticFillBlankModel()
model.train(train_data)

# Usage
candidates = model.predict_multiple("ăn quả", top_k=3)
# → Tìm câu có ý nghĩa tương tự
```

**Ưu điểm:**
- ✅ Hiểu semantic similarity
- ✅ Có thể match câu khác từ khóa nhưng cùng ý nghĩa
- ✅ Tốt cho inputs mơ hồ

**Nhược điểm:**
- ❌ Chậm hơn Retrieval
- ❌ Cần model pre-trained (vietnamese-sbert)

---

### 3. **N-gram Model** (Statistical Language Model)

**Accuracy**: 15% | **Speed**: ⚡⚡⚡ Fast

```python
from backend.models.ngram import NgramModel

model = NgramModel(n=3)  # Trigram
model.train(train_data)

# Usage
prediction = model.predict("ăn quả")
# → Generate từng từ tiếp theo
```

**Ưu điểm:**
- ✅ Simple, interpretable
- ✅ Nhanh

**Nhược điểm:**
- ❌ Accuracy rất thấp (15%)
- ❌ Không phù hợp với task này
- ⚠️ **KHÔNG khuyến nghị dùng**

---

### 4. **Transformer Mini** (Attention-based)

**Accuracy**: 40-50% | **Speed**: ⚡ Slow

```python
from backend.models.transformer_mini import TransformerCaDao

model = TransformerCaDao()
model.train(train_data, epochs=20)

# Usage
prediction = model.predict("ăn quả")
```

**Ưu điểm:**
- ✅ State-of-the-art architecture
- ✅ Có thể học patterns phức tạp

**Nhược điểm:**
- ❌ Cần nhiều data để train tốt
- ❌ Slow inference
- ❌ Overfitting với dataset nhỏ (2,265 samples)

---

### 5. **Improved Ensemble** (Retrieval + Semantic)

**Accuracy**: 62% ⭐ | **Speed**: ⚡⚡ Medium

```python
from backend.models.ensemble import ImprovedEnsembleModel
from backend.models.retrieval import RetrievalModel
from backend.models.semantic_fill_blank import SemanticFillBlankModel

# Load models
retrieval = RetrievalModel()
retrieval.load("trained_models/retrieval_model.pkl")

semantic = SemanticFillBlankModel()
semantic.load("trained_models/semantic_model.pkl")

# Create ensemble
ensemble = ImprovedEnsembleModel(retrieval, semantic)

# Usage
candidates = ensemble.predict_multiple("ăn quả", top_k=5)
```

**Strategy:**
- Weighted voting: Retrieval (60%) + Semantic (40%)
- Diversity bonus cho candidates xuất hiện ở cả 2 models
- Adaptive confidence calibration

**Ưu điểm:**
- ✅ **Accuracy cao nhất** (62%)
- ✅ Kết hợp keyword + semantic
- ✅ Robust với nhiều loại inputs

---

## 🚀 Cài đặt

### Prerequisites

- Python 3.8+
- pip
- (Optional) GPU cho training Transformer

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/yourusername/cadao-tucngu-nlp.git
cd cadao-tucngu-nlp

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download data (if not included)
python backend/data/download_data.py

# 5. Preprocess data
python backend/data/preprocess.py
```

### Requirements.txt

```txt
# Core
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# NLP
sentence-transformers>=2.2.0
torch>=1.10.0

# Utils
tqdm>=4.62.0
python-dotenv>=0.19.0

# Web (optional)
fastapi>=0.70.0
uvicorn>=0.15.0

# Testing
pytest>=7.0.0
```

---

## 📚 Sử dụng

### 1. Training Models

#### Train Retrieval Model

```bash
cd backend/models
python retrieval.py
```

Output:
```
🚀 RETRIEVAL MODEL TRAINING
════════════════════════════════════════════════════════════
✓ Train: 2265 samples
✓ Test:  486 samples
✓ Database: 1492 unique sentences
✓ Accuracy: 55.0%
✓ Model saved: trained_models/retrieval_model.pkl
```

#### Train Semantic Model

```bash
python semantic_fill_blank.py
```

#### Train Ensemble

```bash
python ensemble.py
```

---

### 2. Interactive Terminal

```bash
python interactive_autocomplete.py
```

**Example Session:**

```
🎯 INTERACTIVE AUTOCOMPLETE - Ca Dao & Tục Ngữ
══════════════════════════════════════════════════════════

>>> Nhập: ăn quả

   📋 Suggestions:
   1. 🟢 [95% HIGH] 🎯 ăn quả nhớ kẻ trồng cây
   2. 🟡 [70% MED ] 🔤 gieo nhân nào gặt quả nấy
   3. 🔴 [45% LOW ] ⭐ ăn cháo đá bát

>>> Nhập: có công

   📋 Suggestions:
   1. 🟢 [95% HIGH] 🎯 có công mài sắt có ngày nên kim
   2. 🟡 [70% MED ] 🔤 công cha như núi thái sơn

>>> Nhập: q
👋 Goodbye!
```

---

### 3. Python API

```python
from backend.models.retrieval import RetrievalModel
from backend.models.semantic_fill_blank import SemanticFillBlankModel
from backend.models.ensemble import ImprovedEnsembleModel

# Load models
retrieval = RetrievalModel()
retrieval.load("backend/trained_models/retrieval_model.pkl")

semantic = SemanticFillBlankModel()
semantic.load("backend/trained_models/semantic_model.pkl")

# Create ensemble
ensemble = ImprovedEnsembleModel(retrieval, semantic)

# Get suggestions
user_input = "ăn quả"
suggestions = ensemble.predict_multiple(user_input, top_k=5)

for i, sugg in enumerate(suggestions, 1):
    print(f"{i}. [{sugg['confidence']:.0%}] {sugg['text']}")
```

---

### 4. REST API (FastAPI)

```bash
cd backend
uvicorn api.main:app --reload
```

**Endpoint:**

```bash
POST /api/autocomplete
Content-Type: application/json

{
  "input": "ăn quả",
  "top_k": 5,
  "min_confidence": 0.5
}
```

**Response:**

```json
{
  "suggestions": [
    {
      "text": "ăn quả nhớ kẻ trồng cây",
      "confidence": 0.95,
      "strategy": "exact_prefix"
    },
    {
      "text": "gieo nhân nào gặt quả nấy",
      "confidence": 0.70,
      "strategy": "word_match"
    }
  ],
  "query_time_ms": 15.3
}
```

---

## 📊 Kết quả

### Model Comparison

| Model | Exact Acc | Top-3 Acc | Top-5 Acc | Speed | Production Ready |
|-------|-----------|-----------|-----------|-------|------------------|
| **N-gram** | 15.0% | 30.0% | - | ⚡⚡⚡ | ❌ |
| **Retrieval** | 55.0% | 70.0% | - | ⚡⚡⚡ | ✅ |
| **Semantic** | 50-55% | 65-70% | - | ⚡⚡ | ✅ |
| **Transformer** | 40-50% | 60-65% | - | ⚡ | ❌ |
| **Ensemble** | **62.0%** | **69.5%** | **71.0%** | ⚡⚡ | ✅ ⭐ |

### Strategy Performance

| Strategy | Accuracy | Use Cases | Confidence |
|----------|----------|-----------|------------|
| **Exact Prefix** | 90-95% | Input dài, rõ ràng | 95% |
| **Fuzzy Prefix** | 80-85% | Input có typo | 85% |
| **Semantic** | 60-70% | Input về ý nghĩa | 70% |
| **Retrieval** | 55-60% | Keyword matching | 60% |
| **Popular** | 30-40% | Fallback | 40% |

### Real-world Performance

**Test Cases:**

| Input | Expected | Model Output | Correct? |
|-------|----------|--------------|----------|
| "ăn quả" | ăn quả nhớ kẻ trồng cây | ✅ Same | ✅ |
| "có công" | có công mài sắt có ngày nên kim | ✅ Same | ✅ |
| "gần mực" | gần mực thì đen gần đèn thì sáng | ✅ Same | ✅ |
| "học thầy" | học thầy không tày học bạn | ❌ như thầy tăng thầy lộ | ❌ |

**Success Rate**: 75% (3/4)

---

## 🔌 API Documentation

### Endpoints

#### `POST /api/autocomplete`

Autocomplete câu ca dao/tục ngữ.

**Request:**
```json
{
  "input": "string",           // Required: User input
  "top_k": 5,                  // Optional: Number of suggestions (default: 5)
  "min_confidence": 0.5,       // Optional: Min confidence threshold (default: 0.5)
  "strategy": "auto"           // Optional: "auto" | "exact" | "semantic" | "retrieval"
}
```

**Response:**
```json
{
  "suggestions": [
    {
      "text": "string",
      "confidence": 0.95,
      "strategy": "exact_prefix",
      "rank": 1
    }
  ],
  "query_time_ms": 15.3,
  "total_candidates": 10
}
```

#### `GET /api/stats`

Lấy thống kê hệ thống.

**Response:**
```json
{
  "total_sentences": 1492,
  "models_loaded": ["retrieval", "semantic"],
  "cache_size": 1500,
  "uptime_seconds": 3600
}
```

---

## 📁 Project Structure

```
cadao-tucngu-nlp/
├── backend/
│   ├── data/
│   │   ├── raw/                    # Raw data files
│   │   │   └── cadao_tucngu.txt
│   │   ├── processed/              # Processed data
│   │   │   ├── train.json
│   │   │   ├── val.json
│   │   │   └── test.json
│   │   ├── preprocess.py           # Data preprocessing script
│   │   └── download_data.py
│   │
│   ├── models/
│   │   ├── retrieval.py            # TF-IDF Retrieval model
│   │   ├── semantic_fill_blank.py  # Semantic model
│   │   ├── ngram.py                # N-gram model
│   │   ├── transformer_mini.py     # Transformer model
│   │   ├── ensemble.py             # Ensemble model
│   │   ├── interactive_autocomplete.py  # Interactive terminal
│   │   └── smart_fallback_system.py     # Production system
│   │
│   ├── trained_models/             # Saved models
│   │   ├── retrieval_model.pkl
│   │   ├── semantic_model.pkl
│   │   └── ensemble_config.pkl
│   │
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   └── utils/
│       ├── metrics.py              # Evaluation metrics
│       └── helpers.py
│
├── frontend/                       # (Coming soon)
│   ├── src/
│   ├── public/
│   └── package.json
│
├── tests/
│   ├── test_retrieval.py
│   ├── test_semantic.py
│   └── test_ensemble.py
│
├── docs/
│   ├── model_comparison.md
│   ├── api_guide.md
│   └── training_guide.md
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Test Individual Models

```bash
# Test retrieval
python backend/models/retrieval.py

# Test semantic
python backend/models/semantic_fill_blank.py

# Test ensemble
python backend/models/ensemble.py
```

### Evaluation

```python
from backend.models.retrieval import RetrievalModel
import json

# Load test data
with open('backend/data/processed/test.json', 'r') as f:
    test_data = json.load(f)

# Load model
model = RetrievalModel()
model.load('backend/trained_models/retrieval_model.pkl')

# Evaluate
metrics = model.evaluate(test_data)

print(f"Accuracy: {metrics['exact_accuracy']:.1%}")
print(f"Top-3 Accuracy: {metrics['top3_accuracy']:.1%}")
```

---

## 📈 Performance Optimization

### Tips for Better Performance

1. **Cache frequently queried inputs**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_suggestions(input_text):
       return model.predict_multiple(input_text)
   ```

2. **Use batch inference**
   ```python
   # Instead of
   for text in texts:
       model.predict(text)
   
   # Use
   model.predict_batch(texts)
   ```

3. **Optimize model weights**
   ```python
   ensemble.optimize_weights(val_data, steps=10)
   ```

4. **Use appropriate top_k**
   - User-facing: `top_k=3-5`
   - Internal processing: `top_k=10-20`

---

## 🐛 Troubleshooting

### Common Issues

**1. Model file not found**
```
FileNotFoundError: [Errno 2] No such file or directory: 'trained_models/retrieval_model.pkl'
```
**Solution:** Train the model first:
```bash
python backend/models/retrieval.py
```

---

**2. Scikit-learn version warning**
```
InconsistentVersionWarning: Trying to unpickle estimator from version 1.7.2 when using version 1.6.1
```
**Solution:** Update scikit-learn:
```bash
pip install --upgrade scikit-learn
```

---

**3. Low accuracy on custom data**
```
Model accuracy: 30% (expected 55%+)
```
**Solution:** 
- Ensure data format is correct
- Increase training data size
- Check data quality (duplicates, typos)

---

**4. Slow inference**
```
Query time: 500ms (expected <50ms)
```
**Solution:**
- Enable caching
- Use lighter models (Retrieval instead of Semantic)
- Reduce top_k parameter

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README.md if needed

---

## 📝 License

This pimninh/cadao-tucngu-nlp)

---

## 🗺️ Roadmap

- [x] Basic retrieval model
- [x] Semantic model
- [x] Ensemble system
- [x] Interactive terminal
- [ ] Web interface (React)
- [ ] Mobile app (React Native)
- [ ] User feedback collection
- [ ] Model fine-tuning with user data
- [ ] Multi-language support (English sayings)
- [ ] Voice input support
- [ ] Educational games

---

## 📚 Further Reading

- [Model Comparison Guide](docs/model_comparison.md)
- [API Usage Guide](docs/api_guide.md)
- [Training Custom Models](docs/training_guide.md)
- [Deployment Guide](docs/deployment.md)

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ and ☕ by [Group 16]

</div>
