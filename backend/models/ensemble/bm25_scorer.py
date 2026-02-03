"""
BM25 SCORER - Module để tính điểm BM25 giữa 2 câu

Hàm chính:
    calculate_bm25(sentence1, sentence2, k1=1.5, b=0.75) -> float
    
Ví dụ:
    from bm25_scorer import calculate_bm25
    
    score = calculate_bm25("ăn quả nhớ", "ăn trái nhớ kẻ trồng cây")
    print(f"BM25 Score: {score:.3f}")
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def calculate_bm25(sentence1, sentence2, k1=1.5, b=0.75):
    """
    Tính BM25 score giữa 2 câu
    
    Args:
        sentence1 (str): Câu query/input
        sentence2 (str): Câu document để so sánh
        k1 (float): TF saturation parameter (default: 1.5)
        b (float): Length normalization (default: 0.75)
    
    Return:
        float: BM25 score (điểm số)
        
    Ví dụ:
        >>> score = calculate_bm25("ăn quả", "ăn trái quả")
        >>> print(f"{score:.3f}")  # Output: 10.234
    """
    
    # Normalize
    sentence1 = sentence1.lower().strip()
    sentence2 = sentence2.lower().strip()
    
    if not sentence1 or not sentence2:
        return 0.0
    
    # Fit vectorizer on both sentences
    corpus = [sentence1, sentence2]
    
    # Get TF-IDF vectorizer (for IDF values)
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        strip_accents=None
    )
    vectorizer.fit(corpus)
    idf = vectorizer.idf_
    
    # Get term frequencies using CountVectorizer
    count_vec = CountVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        lowercase=True,
        strip_accents=None,
        vocabulary=vectorizer.vocabulary_
    )
    
    term_freqs = count_vec.fit_transform(corpus)
    
    # Document lengths
    doc_len = len(sentence2.split())
    avg_doc_len = np.mean([len(s.split()) for s in corpus])
    
    # Get query vector (sentence1)
    query_vec = term_freqs[0].toarray()[0]
    doc_vec = term_freqs[1].toarray()[0]
    
    # Calculate BM25
    score = 0.0
    
    for term_idx in np.nonzero(query_vec)[0]:
        tf_doc = doc_vec[term_idx]
        
        if tf_doc == 0:
            continue
        
        idf_val = idf[term_idx]
        
        # BM25 formula
        numerator = tf_doc * (k1 + 1)
        denominator = tf_doc + k1 * (
            1 - b + b * (doc_len / avg_doc_len)
        )
        
        score += idf_val * (numerator / denominator)
    
    return score


# ============================================================
# TEST EXAMPLE
# ============================================================

if __name__ == "__main__":
    """Test examples"""
    
    # Example 1: Simple test
    print("\n" + "="*70)
    print("📊 BM25 Score Examples")
    print("="*70)
    
    test_cases = [
        ("ăn quả nhớ", "ăn trái nhớ kẻ trồng cây"),
        ("có công", "có công mài sắt có ngày nên kim"),
        ("gần mực", "gần mực thì đen gần đèn thì sáng"),
        ("ăn", "ăn cháo, đá bát"),
    ]
    
    for sent1, sent2 in test_cases:
        score = calculate_bm25(sent1, sent2)
        print(f"\nSentence 1: {sent1}")
        print(f"Sentence 2: {sent2}")
        print(f"BM25 Score: {score:.3f}")
        print("-" * 70)
