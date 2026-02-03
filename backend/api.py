import os
import sys
import time
import pickle
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. PATH SETUP  — tính từ vị trí của app.py (backend/)
#    NLP_v01/
#      backend/app.py        ← file này
#      data/                 ← train_data_seg.txt
#      models/
#        n_gram/             ← gen_mul_layer.py, checkpoint/model.bin
#        retrieval/          ← inference.py, checkpoint/retrieval.pkl
#        ensemble/           ← gen_ensemble.py, bm25_scorer.py
# ---------------------------------------------------------------------------
BACKEND_DIR  = Path(__file__).resolve().parent          # backend/
PROJECT_ROOT = BACKEND_DIR.parent                       # Cadao-Tucngu-NLP/

# Thêm paths để import các module
sys.path.insert(0, str(BACKEND_DIR / "models" / "n_gram"))
sys.path.insert(0, str(BACKEND_DIR / "models" / "ensemble"))
sys.path.insert(0, str(BACKEND_DIR / "models" / "retrieval"))

# Paths các file data / model - SỬA LẠI ĐÂY
RETRIEVAL_PKL = str(BACKEND_DIR / "models" / "retrieval" / "checkpoint" / "retrieval.pkl")
NGRAM_MODEL   = str(BACKEND_DIR / "models" / "n_gram" / "checkpoint" / "model.bin")
TRAIN_DATA    = str(BACKEND_DIR / "data" / "train_data_seg.txt")

# ---------------------------------------------------------------------------
# 2. FLASK APP
# ---------------------------------------------------------------------------
app  = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
])

# ---------------------------------------------------------------------------
# 3. LOAD MODELS (1 lần khi server start)
# ---------------------------------------------------------------------------

# ---- 3a. Retrieval model (BM25) — load từ retrieval.pkl ----
_ret = {}   # chứa vectorizer, database, term_freqs, idf, ...

def _load_retrieval():
    global _ret
    if _ret:
        return  # đã load
    if not os.path.exists(RETRIEVAL_PKL):
        print(f"⚠️  retrieval.pkl không tìm: {RETRIEVAL_PKL}")
        return
    print(f"[startup] Loading retrieval.pkl …")
    with open(RETRIEVAL_PKL, "rb") as f:
        _ret = pickle.load(f)
    print(f"[startup] ✅ Retrieval: {len(_ret['database'])} docs")

# ---- 3b. N-gram model (KenLM) ----
_kenlm_model = None   # kenlm.Model hoặc None
_corpus      = []     # train_data_seg.txt lines

def _load_ngram():
    global _kenlm_model, _corpus

    # load corpus (train_data_seg.txt)
    if not _corpus:
        if os.path.exists(TRAIN_DATA):
            with open(TRAIN_DATA, "r", encoding="utf-8") as f:
                _corpus = [l.strip() for l in f if l.strip()]
            print(f"[startup] ✅ Corpus: {len(_corpus)} sentences")
        else:
            print(f"⚠️  train_data_seg.txt không tìm: {TRAIN_DATA}")

    # load KenLM model
    if _kenlm_model is None:
        try:
            import kenlm                                       # type: ignore
            _kenlm_model = kenlm.Model(NGRAM_MODEL)
            print(f"[startup] ✅ KenLM loaded (order={_kenlm_model.order})")
        except ImportError:
            print("[startup] ⚠️  kenlm chưa install → n_gram dùng fallback overlap scoring")
        except Exception as e:
            print(f"[startup] ⚠️  KenLM load lỗi: {e}")

# ---- load cả hai khi app start ----
_load_retrieval()
_load_ngram()

# ---------------------------------------------------------------------------
# 4. CORE FUNCTIONS  — 3 model predictions
# ---------------------------------------------------------------------------

def _bm25_search(query: str, top_k: int = 3):
    """
    BM25 search trên toàn corpus.
    Return: [(score, text), ...]  sorted desc
    """
    if not _ret:
        return []

    vectorizer  = _ret["vectorizer"]
    term_freqs  = _ret["term_freqs"]
    idf         = _ret["idf"]
    doc_lengths = _ret["doc_lengths"]
    avg_dl      = float(_ret["avg_doc_len"])
    k1          = float(_ret["bm25_k1"])
    b           = float(_ret["bm25_b"])
    database    = _ret["database"]

    qv     = vectorizer.transform([query])
    scores = np.zeros(term_freqs.shape[0])
    for idx in qv.indices:
        tf      = np.asarray(term_freqs[:, idx].todense()).flatten()
        scores += idf[idx] * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_lengths / avg_dl))

    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), database[i]) for i in top_idx]


def _predict_retrieval(text: str, top_k: int = 3):
    """Retrieval model → BM25 search."""
    t0      = time.perf_counter()
    raw     = _bm25_search(text, top_k)
    elapsed = time.perf_counter() - t0

    max_s = max((s for s, _ in raw), default=1.0) or 1.0
    return {
        "model_name": "Retrieval Model",
        "model_type": "retrieval",
        "results": [
            {
                "text":       cadao,
                "confidence": round(score / max_s, 4),
                "score":      round(score, 2),
                "source":     "corpus",
                "metadata":   {"bm25_score": round(score, 2)}
            } for score, cadao in raw
        ],
        "metrics": {
            "inference_time": round(elapsed, 4),
            "total_results":  len(raw)
        }
    }


def _predict_ngram(text: str, top_k: int = 3):
    """
    N-gram model.
    - Nếu kenlm đã install: dùng BidirectionalBeamGenerator từ gen_mul_layer.py
    - Nếu không: fallback overlap scoring trên corpus
    """
    t0 = time.perf_counter()

    # --- Thử import BidirectionalBeamGenerator (gen_mul_layer.py có sẵn) ---
    try:
        from gen_mul_layer import BidirectionalBeamGenerator  # noqa: F811
        if _kenlm_model is not None:
            # Dùng beam generator thật
            gen     = BidirectionalBeamGenerator(NGRAM_MODEL, TRAIN_DATA, n_gram_order=5)
            raw     = gen.generate_best_cases(text, num_results=top_k)   # [(score, text), …]
            elapsed = time.perf_counter() - t0
            max_s   = max((s for s, _ in raw), default=1.0) or 1.0
            return {
                "model_name": "N-Gram Model",
                "model_type": "n_gram",
                "results": [
                    {
                        "text":       cadao,
                        "confidence": round(score / max_s, 4),
                        "score":      round(score, 4),
                        "source":     "generated",
                        "metadata":   {"lm_score": round(score, 4)}
                    } for score, cadao in raw
                ],
                "metrics": {
                    "inference_time": round(elapsed, 4),
                    "total_results":  len(raw)
                }
            }
    except Exception:
        pass  # fall through → overlap fallback

    # --- Fallback: token overlap scoring ---
    q_tokens = set(text.lower().split())
    scored   = []
    for sent in _corpus:
        s_tokens = set(sent.lower().split())
        overlap  = q_tokens & s_tokens
        if overlap:
            scored.append((len(overlap) / len(q_tokens | s_tokens), sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    raw     = scored[:top_k]
    elapsed = time.perf_counter() - t0

    max_s = max((s for s, _ in raw), default=1.0) or 1.0
    return {
        "model_name": "N-Gram Model",
        "model_type": "n_gram",
        "results": [
            {
                "text":       cadao,
                "confidence": round(score / max_s, 4),
                "score":      round(score, 4),
                "source":     "generated",
                "metadata":   {"overlap_score": round(score, 4)}
            } for score, cadao in raw
        ],
        "metrics": {
            "inference_time": round(elapsed, 4),
            "total_results":  len(raw)
        }
    }


def _predict_ensemble(text: str, top_k: int = 3):
    """
    Ensemble = gen_ensemble.py pipeline:
      Step 1: N-gram beam search → 5 candidates
      Step 2: BM25 score từng candidate (bm25_scorer logic)
      Step 3: Sort by BM25 → top-k

    Nếu kenlm có → dùng BidirectionalBeamGenerator cho Step 1
    Nếu không    → fallback overlap cho candidates, vẫn BM25 re-rank
    """
    t0 = time.perf_counter()

    # --- Step 1: lấy candidates (5) ---
    candidates = []

    # Thử BidirectionalBeamGenerator
    try:
        from gen_mul_layer import BidirectionalBeamGenerator  # noqa: F811
        if _kenlm_model is not None:
            gen        = BidirectionalBeamGenerator(NGRAM_MODEL, TRAIN_DATA, n_gram_order=5)
            ngram_raw  = gen.generate_best_cases(text, num_results=5)
            candidates = [t for _, t in ngram_raw]
    except Exception:
        pass

    # Fallback: overlap candidates
    if not candidates:
        q_tokens = set(text.lower().split())
        scored   = []
        for sent in _corpus:
            s_tokens = set(sent.lower().split())
            if q_tokens & s_tokens:
                scored.append((len(q_tokens & s_tokens) / len(q_tokens | s_tokens), sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        candidates = [t for _, t in scored[:5]]

    # --- Step 2: BM25 score mỗi candidate ---
    #   bm25_scorer.calculate_bm25 cũng dùng cùng logic:
    #   score candidate against corpus → lấy max score
    bm25_scored = []
    if _ret:
        vectorizer  = _ret["vectorizer"]
        term_freqs  = _ret["term_freqs"]
        idf         = _ret["idf"]
        doc_lengths = _ret["doc_lengths"]
        avg_dl      = float(_ret["avg_doc_len"])
        k1          = float(_ret["bm25_k1"])
        b           = float(_ret["bm25_b"])

        for cand in candidates:
            qv     = vectorizer.transform([cand])
            scores = np.zeros(term_freqs.shape[0])
            for idx in qv.indices:
                tf      = np.asarray(term_freqs[:, idx].todense()).flatten()
                scores += idf[idx] * (tf*(k1+1)) / (tf + k1*(1-b+b*doc_lengths/avg_dl))
            bm25_scored.append((float(np.max(scores)), cand))
    else:
        # không có retrieval → dùng score 0
        bm25_scored = [(0.0, c) for c in candidates]

    # --- Step 3: sort desc, top-k ---
    bm25_scored.sort(key=lambda x: x[0], reverse=True)
    raw     = bm25_scored[:top_k]
    elapsed = time.perf_counter() - t0

    max_s = max((s for s, _ in raw), default=1.0) or 1.0
    return {
        "model_name": "Ensemble Model",
        "model_type": "ensemble",
        "results": [
            {
                "text":       cadao,
                "confidence": round(score / max_s, 4),
                "score":      round(score, 2),
                "source":     "hybrid",
                "metadata":   {"combined_score": round(score, 2)}
            } for score, cadao in raw
        ],
        "metrics": {
            "inference_time": round(elapsed, 4),
            "total_results":  len(raw)
        }
    }


# ---------------------------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "retrieval_loaded": bool(_ret),
        "kenlm_loaded":     _kenlm_model is not None,
        "corpus_size":      len(_corpus),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Body rỗng"}), 400

    text   = (body.get("text") or "").strip()
    models = body.get("models", ["n_gram", "retrieval", "ensemble"])  # default: cả 3

    if not text:
        return jsonify({"error": "Trường 'text' rỗng"}), 400

    # --- gọi từng model theo danh sách request ---
    model_results = []

    if "retrieval" in models:
        model_results.append(_predict_retrieval(text))

    if "n_gram" in models:
        model_results.append(_predict_ngram(text))

    if "ensemble" in models:
        model_results.append(_predict_ensemble(text))

    return jsonify({
        "input_text":    text,
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "model_results": model_results,
    })


# ---------------------------------------------------------------------------
# 6. RUN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Ca dao & Tục ngữ API  →  http://localhost:{PORT}/api")
    print(f"   Docs                  →  http://localhost:{PORT}/api/health\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)