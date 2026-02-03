"""
Ensemble Text Generation: N-gram + BM25 Scoring
1. Generate 5 candidates using BidirectionalBeamGenerator
2. Score each with BM25 similarity to input
3. Rank and select best
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
from pyvi import ViTokenizer

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, str(Path(__file__).parent.parent / "n_gram"))

from bm25_scorer import calculate_bm25
from gen_mul_layer import BidirectionalBeamGenerator


def main(input_text):
    """Main ensemble inference"""
    
    print("\n" + "="*70)
    print("🎯 Ensemble Text Generation (N-gram + BM25)")
    print("="*70)
    
    # ===== PATHS =====
    project_root = Path(__file__).parent.parent.parent  # /NLP_v01/
    model_file = str(project_root / "models" / "n_gram" / "checkpoint" / "model.bin")
    train_data_file = str(project_root / "data" / "train_data_seg.txt")
    
    # Check files
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return
    
    if not os.path.exists(train_data_file):
        print(f"❌ Training data not found: {train_data_file}")
        return
    
    try:
        # ===== STEP 1: GENERATE CANDIDATES =====
        print("\n" + "="*70)
        print("🎲 STEP 1: Generate 5 Candidates (Beam Search)")
        print("="*70)
        
        generator = BidirectionalBeamGenerator(model_file, train_data_file, n_gram_order=5)
        
        # Generate 5 candidates
        ngram_results = generator.generate_best_cases(input_text, num_results=5)
        
        if not ngram_results:
            print("❌ Failed to generate candidates")
            return
        
        candidates = [text for _, text in ngram_results]
        print(f"\n✅ Generated {len(candidates)} candidates")
        
        # ===== STEP 2: CALCULATE BM25 SCORES =====
        print("\n" + "="*70)
        print("📊 STEP 2: Calculate BM25 Scores")
        print("="*70)
        
        bm25_results = []
        for i, candidate in enumerate(candidates, 1):
            score = calculate_bm25(input_text, candidate)
            bm25_results.append((score, candidate))
            print(f"   {i}. Computing BM25 score... {score:.3f}")
        
        # Sort by BM25 score (descending)
        bm25_results.sort(key=lambda x: x[0], reverse=True)
        
        # ===== STEP 3: DISPLAY RESULTS =====
        print("\n" + "="*70)
        print("📋 RANKING RESULTS")
        print("="*70)
        
        print(f"\n{'Rank':<6} {'BM25 Score':<15} {'Output Text'}")
        print("-" * 80)
        
        for i, (score, text) in enumerate(bm25_results, 1):
            # Medal indicator
            if i == 1:
                medal = "🏆 "
            elif i == 2:
                medal = "🥈 "
            elif i == 3:
                medal = "🥉 "
            else:
                medal = f"{i:2d}. "
            
            # Shorten text for display
            display_text = text if len(text) <= 50 else text[:47] + "..."
            print(f"{medal:<6} {score:<15.3f} {display_text}")
        
        # ===== STEP 4: FINAL RESULT =====
        print("\n" + "="*70)
        print("✨ FINAL RESULT")
        print("="*70)
        
        best_score, best_text = bm25_results[0]
        
        print(f"\n📝 Input: \"{input_text}\"")
        print(f"\n🏆 Best Output (BM25 Score: {best_score:.3f}):")
        print(f"\n{best_text}")
        
        print("\n" + "="*70)
        print("✅ Complete!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test input
    input_text = "ăn quả"
    main(input_text)
