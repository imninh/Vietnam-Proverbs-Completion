# 🇻🇳 Vietnamese Proverb & Folk Verse Completion System

<div align="center">

<!-- Badges -->
<img src="https://img.shields.io/badge/Python-3.8%2B-3572A5?style=flat-square&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/KenLM-5--gram-4CAF50?style=flat-square" />
<img src="https://img.shields.io/badge/Dataset-13%2C062%20verses-FF9800?style=flat-square" />
<img src="https://img.shields.io/badge/Accuracy-40.2%25%20exact%20|%2073.9%25%20similarity-2196F3?style=flat-square" />
<img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square" />

<br/>

> **Ca dao và tục ngữ** là viên ngoc vô giá của văn hóa Việt Nam — nhưng đang dần bị lãng quên.  
> Project này dùng NLP để giúp mọi người **gợi nhớ, học thuộc, và bảo tồn** di sản văn hóa truyền thống.

</div>

---

## 📑 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models & Approaches](#models--approaches)
- [Results & Evaluation](#results--evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Web Deployment](#web-deployment)
- [Limitations & Future Work](#limitations--future-work)
- [Team & References](#team--references)

---

## Overview

Vietnamese folk poetry (**Ca Dao**) and proverbs (**Tục Ngữ**) carry centuries of cultural wisdom — from life lessons, moral values, to reflections on nature and human relationships. Yet in the digital era, younger generations are increasingly losing touch with these traditions.

**This project** builds an intelligent text completion system: given a partial Ca Dao or Tục Ngữ verse, the system predicts and suggests the complete verse. It serves as:

| Purpose | Description |
|---|---|
| 🎓 **Educational Tool** | Helps students and learners recall and memorize traditional verses |
| 📚 **Cultural Preservation** | Creates a digital, searchable repository of Vietnamese folk poetry |
| 🤖 **NLP Research Baseline** | Establishes the first systematic benchmark for Vietnamese poetry completion |
| 🌐 **Public Accessibility** | Deployed as a web app — no specialized tools needed |

---

## How It Works

The core idea is simple: **type a few words you remember, and the system completes the verse for you.**

```
Input:  "công cha như núi"
Output: "Công cha như núi Thái Sơn
         Nghĩa mẹ như nước trong nguồn chảy ra"

Input:  "gần mực thì"
Output: "Gần mực thì đen, gần đèn thì sáng"

Input:  "ăn quả nhớ"
Output: "Ăn quả nhớ kẻ trồng cây"
```

The system supports **three completion scenarios**:

| Scenario | Input Position | Example |
|---|---|---|
| **Forward** | Beginning of verse | `"công cha"` → completes the rest |
| **Backward** | End of verse | `"chảy ra"` → finds the beginning |
| **Bidirectional** | Middle of verse | `"núi thái sơn"` → reconstructs full verse |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Input                          │
│              (partial Ca Dao / Tục Ngữ)                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│               Vietnamese Preprocessing                  │
│  ┌──────────┐  ┌─────────────┐  ┌────────────────────┐  │
│  │  Unicode  │→│  PyVi Word  │→│  Special Tokens    │  │
│  │   NFC     │ │ Segmentation│ │  <s> ... </s>      │  │
│  └──────────┘  └─────────────┘  └────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┼──────────────┐
              ▼          ▼              ▼
    ┌───────────┐ ┌──────────┐  ┌────────────┐
    │  5-gram   │ │  TF-IDF  │  │ Ensemble   │
    │  KenLM    │ │ Retrieval│  │ (30/70)    │
    │  + Beam   │ │  + BM25  │  │ N-gram +   │
    │  Search   │ │          │  │ Retrieval  │
    └─────┬─────┘ └────┬─────┘  └─────┬──────┘
          │            │              │
          └──────────┬─┘              │
                     ▼                ▼
          ┌─────────────────────────────┐
          │    Ranked Suggestions       │
          │  (Score, Confidence, Text)  │
          └─────────────────────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │  Format & Display   │
          │  (Lục Bát layout)   │
          └─────────────────────┘
```

---

## Dataset

### Collection & Curation

The dataset was sourced from authoritative Vietnamese cultural archives and digitized repositories, then manually verified for authenticity and linguistic correctness.

### Statistics

| Metric | Value |
|---|---|
| Total verified verses | **13,062** |
| Vocabulary size (unique tokens) | **6,763** |
| Total tokens | **97,305** |
| Dominant structure | Lục Bát (6–8 syllable lines) |
| Sentence length range | 6–14 words (majority) |

### Preprocessing Pipeline

```
Raw Text
    │
    ▼
① Text Normalization          — Unicode NFC, lowercase, strip whitespace
    │                            merge multi-line verses with "."
    ▼
② Word Segmentation (PyVi)    — "thái sơn" → "thái_sơn"
    │                            accuracy: 94.8% on 500 sample verses
    ▼
③ Boundary Tagging            — prepend <s>, append </s>
    │
    ▼
④ Quality Validation          — length filter (4–35 tokens),
    │                            character validation, content check
    ▼
train_data_seg.txt            — 13,062 clean, tokenized verses
```

> ⚠️ **Critical**: Vietnamese tonal diacritics are **never** removed. `"ma"` (ghost) ≠ `"má"` (mother) ≠ `"mà"` (but) — each tone carries distinct meaning.

---

## Models & Approaches

Three approaches were designed and compared. Each leverages different strengths of the data:

### 1. 5-gram Language Model (Primary) — KenLM + Bidirectional Beam Search

**The core model.** A 5-gram order was chosen because Vietnamese Ca Dao lines typically have 6–8 syllables, so a 4-word context window captures most of a half-line.

| Component | Details |
|---|---|
| **Toolkit** | KenLM (`lmplz`, `build_binary`) |
| **Smoothing** | Modified Kneser-Ney (handles unseen n-grams gracefully) |
| **Search** | Bidirectional Beam Search (beam width = 10 backward, 5 forward) |
| **Vocabulary Maps** | Pre-built `fwd_map` and `bwd_map` for O(1) context lookup |
| **Inference** | < 100ms per completion |

**Beam Search Flow:**
```
Phase 1 — Backward Expansion:
  seed_words → bwd_map → expand toward <s>
  (reconstruct the beginning of the verse)

Phase 2 — Forward Expansion:
  recovered prefix → fwd_map + KenLM scoring → expand toward </s>
  (complete the rest of the verse)

Post-processing:
  Deduplicate → Re-rank by total log-probability → Format output
```

**N-gram Statistics from training:**

| N-gram Order | Unique N-grams |
|---|---|
| 1-gram | 6,816 |
| 2-gram | 48,941 |
| 3-gram | 70,121 |
| 4-gram | 74,188 |
| 5-gram | 67,191 |

### 2. TF-IDF Retrieval Baseline

A **retrieval-based** approach that treats completion as an information retrieval problem.

- Vectorize all training verses using `TfidfVectorizer` (scikit-learn)
- Compute **cosine similarity** between input query and all stored verses
- Return top-K most similar complete verses

**Strengths:** Always returns valid, real verses from the corpus. Fast and simple.  
**Weaknesses:** Cannot generate novel completions — limited to lexical matching.

### 3. Ensemble Model (Best Performance)

Combines N-gram generation with retrieval for the best of both worlds.

```
┌─ N-gram model  → top-5 candidates  ──┐
│                                       ▼
│                              Combined Scoring:
│                              score = 0.3 × ngram + 0.7 × retrieval
│                              + 0.15 bonus if both methods agree
│                                       │
└─ TF-IDF retrieval → top-5 results ───┘
                                       │
                                       ▼
                              Final ranked output
```

| Model | Role |
|---|---|
| TF-IDF (70%) | Anchor — high precision for known patterns |
| N-gram (30%) | Flexibility — handles partial matches & variations |
| Agreement bonus (+0.15) | Confidence boost when both models agree |

---

## Results & Evaluation

### Test Setup

- **239,883 test cases** generated exhaustively from 1,306 held-out verses
- Three positional categories: Start / Mid / End
- Input lengths: 1 word → 70% of verse length
- Metrics: **Exact Match Accuracy** and **Character-level Similarity** (`SequenceMatcher`)

### Overall Performance (5-gram Beam Search)

| Metric | Score |
|---|---|
| Exact Match Accuracy | **40.20%** |
| Average Similarity | **73.89%** |
| Similarity ≥ 70% | **74.0%** of all predictions |
| Similarity ≥ 90% | **45.3%** of all predictions |

### Performance by Input Position

| Position | Accuracy | Notes |
|---|---|---|
| **End** | Highest (~46.9%) | Verse endings are more formulaic & distinctive |
| **Start** | Mid-range | Forward generation benefits from longer context |
| **Mid** | Lowest | Requires both backward + forward expansion — hardest |

### Performance by Input Length

| Input Words | Behavior |
|---|---|
| 1 word | Low accuracy (high ambiguity) |
| 2–3 words | **Sweet spot** — accuracy 46–68% |
| 4+ words | Exact match drops (too specific), but similarity stays high |

### Similarity Distribution

| Similarity Range | % of Cases | Interpretation |
|---|---|---|
| 90–100% | 45.3% | Near-perfect or exact match |
| 70–89% | 28.7% | Semantically related, partial match |
| 50–69% | 16.2% | Some overlap, wrong verse |
| 0–49% | 9.8% | Completely wrong or failure |

> 💡 **Key Insight:** The model rarely outputs nonsense. Even when not exactly correct, 74% of predictions are at least 70% similar to the ground truth — meaning most "failures" are near-misses.

---

## Installation

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.8+ |
| KenLM | latest |
| PyVi | ≥ 0.1.1 |
| scikit-learn | ≥ 0.24.0 |
| numpy | ≥ 1.19.0 |
| Flask | (for web app) |

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/ca-dao-completion.git
cd ca-dao-completion

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install KenLM (requires C++ compiler)
pip install kenlm
# If build fails, install cmake and a C++ compiler first:
# sudo apt install cmake g++        (Ubuntu)
# brew install cmake                (Mac)
```

### Train the Model

```bash
# Preprocess raw data → tokenized training file
python ngram/preprocess.py

# Train 5-gram KenLM model
python ngram/train.py
# Output: checkpoint/model.bin
```

## Web Deployment: [https://vietnam-proverbs-completion-7f5b.vercel.app]

```

### Deployment Checklist

- [ ] Use `model.bin` (binary) — loads in milliseconds vs. seconds for ARPA
- [ ] Enable caching for TF-IDF vectors (computed once at startup)
- [ ] Set `FLASK_ENV=production`
- [ ] Response target: **< 100ms** per completion

---

## Limitations & Future Work

### Current Limitations

The following are known constraints of the current system — each represents a concrete direction for future improvement.

| # | Area | Limitation |
|---|---|---|
| 1 | 🪟 **Context Window** | 5-gram captures only 4 words of context. Long-range poetic structure (e.g. thematic coherence across a full verse) may be missed. |
| 2 | 📦 **Dataset Size** | 13,062 verses is sufficient for N-gram modeling, but too small to fine-tune neural models like PhoBERT without overfitting. |
| 3 | 🧠 **Semantic Understanding** | The model is purely statistical — it matches surface-level word patterns, not meaning. It cannot reason about theme or emotion. |
| 4 | 🗺️ **Regional Variation** | The corpus may over-represent certain regions or historical periods, under-representing rarer Ca Dao dialects. |
| 5 | 📊 **Evaluation** | All metrics are automated (exact match, similarity). No expert literary evaluation has been conducted yet. |

---

### Roadmap

```
NOW ──────────────► SHORT-TERM ──────────► MEDIUM-TERM ──────────► LONG-TERM
                    (next 3–6 mo)          (6–12 mo)               (12+ mo)

  ✅ 5-gram          📦 Expand dataset      🤖 Fine-tune            🎨 Poetry
  ✅ TF-IDF          🔧 VnCoreNLP           PhoBERT / ViT5         generation
  ✅ Ensemble        📐 Confidence          📐 Lục Bát meter       🎮 Gamified
  ✅ Web app         🔄 User feedback       🔗 Multi-task          🌏 Multi-
  ✅ Evaluation                             learning               cultural
```

**Short-term** — foundation improvements

- [ ] Expand dataset to **50,000+ verses** from additional cultural archives (target: +10–15% accuracy)
- [ ] Swap in `VnCoreNLP` for higher-accuracy Vietnamese word segmentation (currently PyVi at 94.8%)
- [ ] Add confidence calibration via temperature scaling so scores better reflect true accuracy
- [ ] Wire up a user feedback loop in the web app — let users correct wrong completions

**Medium-term** — move toward neural & structural modeling

- [ ] Fine-tune **PhoBERT** or **ViT5** on the Ca Dao corpus once dataset is large enough
- [ ] Explicitly model **Lục Bát** meter constraints (6–8 syllable structure + rhyme scheme) as a generation filter
- [ ] Explore multi-task learning: joint training on completion, verse classification, and paraphrasing

**Long-term** — beyond completion

- [ ] Controlled **poetry generation** — create brand-new Ca Dao matching a user-specified theme or structure
- [ ] **Multimodal** inputs: generate verses inspired by images (e.g. Vietnamese landscapes) or pair with traditional melodies
- [ ] Build a **gamified learning platform** with progressive difficulty, quizzes, and streak tracking
- [ ] Expand into a cross-cultural poetry system bridging Vietnamese, Chinese, and Japanese traditions

---

## Team & References

### 👥 Project Team — Group 16

| Name | Student ID | Role |
|---|---|---|
| Vũ Xuân Anh | 20233832 | 
| Đào Hữu Mao | 20233865 | 
| Trần Thế Ninh | 20233873 | 
| Lê Thi Thảo | 20233877 | 

| | |
|---|---|
| **Supervisor** | Dr. Đỗ Thị Ngọc Diệp |
| **Institution** | Hanoi University of Science and Technology |
| **Department** | School of Electrical and Electronic Engineering |
| **Course** | Natural Language Processing — Class 161838 |

---

### 📚 Key References

1. Jurafsky, D. & Martin, J. H. (2023). *Speech and Language Processing*, 3rd ed. — Pearson.
2. Nguyen, P. T. & Nguyen, L. M. (2020). *PhoBERT: Pre-trained language models for Vietnamese.* Findings of EMNLP 2020.
3. Heafield, K. (2011). *KenLM: Faster and Smaller Language Model Queries.* — WMT 2011.
4. Kneser, R. & Ney, H. (1995). *Improved backing-off techniques for estimating n-gram probabilities.*
5. Vu, M. H., Hoang, A. C. & Nguyen, T. T. (2021). *Vietnamese Natural Language Processing: A Survey.* arXiv:2103.01331.
6. Vaswani, A. et al. (2017). *Attention is all you need.* NeurIPS 2017.

---

<div align="center">

> *"Một cây làm chả nên hồi,*
> *Ba cây chụm lại mấy đời còn xanh🌿"*


</div>
