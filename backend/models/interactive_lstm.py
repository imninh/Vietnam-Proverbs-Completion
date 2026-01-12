"""
INTERACTIVE ENSEMBLE TERMINAL
Retrieval (50%) + LSTM (50%) with adaptive weighting

Usage:
    python interactive_ensemble.py
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict
import torch 

class RetrievalLSTMEnsemble:
    """
    Ensemble combining Retrieval + LSTM with adaptive weights
    """
    
    def __init__(self, retrieval_model, lstm_predictor):
        self.retrieval = retrieval_model
        self.lstm = lstm_predictor
        
        # Base weights
        self.base_weights = {
            'retrieval': 0.5,
            'lstm': 0.5
        }
        
        # Statistics
        self.stats = {
            'queries': 0,
            'retrieval_dominant': 0,
            'lstm_dominant': 0,
            'balanced': 0,
            'agreements': 0
        }
        
        print(f"\n{'─'*60}")
        print(f"🎯 RETRIEVAL + LSTM ENSEMBLE")
        print(f"{'─'*60}")
        print(f"✓ Base weights: 50% / 50%")
        print(f"✓ Strategy: Adaptive weighting")
    
    def _calculate_adaptive_weights(self, input_text, ret_cands, lstm_cands):
        """
        Calculate weights based on input characteristics
        
        Strategy:
        1. If Retrieval has high confidence (>0.9) → 70% Retrieval
        2. If input is very short (≤2 words) → 60% Retrieval
        3. If both agree → Keep balanced
        4. Default → Balanced
        """
        
        # Check Retrieval confidence
        if ret_cands and ret_cands[0]['confidence'] > 0.9:
            return {'retrieval': 0.70, 'lstm': 0.30}, 'high_retrieval_conf'
        
        # Check input length
        words = input_text.strip().split()
        if len(words) <= 2:
            return {'retrieval': 0.60, 'lstm': 0.40}, 'short_input'
        
        # Check agreement
        if ret_cands and lstm_cands:
            if ret_cands[0]['text'] == lstm_cands[0]['text']:
                self.stats['agreements'] += 1
                return {'retrieval': 0.50, 'lstm': 0.50}, 'agreement'
        
        # Default
        return self.base_weights, 'balanced'
    
    def predict_multiple(self, input_text, top_k=5, return_metadata=False):
        """
        Predict with ensemble
        
        Returns:
            List of candidates with confidence scores
        """
        
        self.stats['queries'] += 1
        
        # Get predictions from both models
        try:
            ret_cands = self.retrieval.predict_multiple(input_text, top_k=top_k*2)
        except Exception as e:
            print(f"⚠️  Retrieval error: {e}")
            ret_cands = []
        
        try:
            lstm_cands = self.lstm.predict_multiple(input_text, top_k=top_k*2)
        except Exception as e:
            print(f"⚠️  LSTM error: {e}")
            lstm_cands = []
        
        # Fallback
        if not ret_cands and not lstm_cands:
            return [{
                'text': input_text,
                'confidence': 0.0,
                'model': 'fallback'
            }]
        
        # Calculate adaptive weights
        weights, strategy = self._calculate_adaptive_weights(
            input_text, ret_cands, lstm_cands
        )
        
        # Update stats
        if strategy == 'high_retrieval_conf':
            self.stats['retrieval_dominant'] += 1
        elif strategy == 'short_input':
            self.stats['retrieval_dominant'] += 1
        elif strategy == 'balanced':
            self.stats['balanced'] += 1
        
        # Score candidates
        scores = defaultdict(lambda: {
            'score': 0.0,
            'sources': [],
            'confidences': [],
            'weights': weights,
            'strategy': strategy
        })
        
        # Score from Retrieval
        for i, cand in enumerate(ret_cands):
            text = cand['text']
            conf = cand.get('confidence', 0.5)
            
            # Rank bonus
            rank_bonus = 0.15 * (1 - i / len(ret_cands))
            
            weighted_score = (conf + rank_bonus) * weights['retrieval']
            
            scores[text]['score'] += weighted_score
            scores[text]['sources'].append('retrieval')
            scores[text]['confidences'].append(conf)
        
        # Score from LSTM
        for i, cand in enumerate(lstm_cands):
            text = cand['text']
            conf = cand.get('confidence', 0.5)
            
            rank_bonus = 0.15 * (1 - i / len(lstm_cands))
            
            weighted_score = (conf + rank_bonus) * weights['lstm']
            
            scores[text]['score'] += weighted_score
            scores[text]['sources'].append('lstm')
            scores[text]['confidences'].append(conf)
        
        # Diversity bonus
        for text, info in scores.items():
            if len(set(info['sources'])) > 1:
                scores[text]['score'] += 0.20
                scores[text]['agreement'] = True
            else:
                scores[text]['agreement'] = False
        
        # Sort
        sorted_candidates = sorted(
            scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Format output
        results = []
        for text, info in sorted_candidates[:top_k]:
            raw_score = info['score']
            
            # Calibrated confidence
            if info['agreement']:
                confidence = min(0.95, raw_score)
            else:
                confidence = min(0.85, raw_score * 0.9)
            
            result = {
                'text': text,
                'confidence': round(confidence, 3),
                'model': 'ensemble',
                'sources': list(set(info['sources'])),
                'agreement': info['agreement']
            }
            
            if return_metadata:
                result.update({
                    'weights': info['weights'],
                    'strategy': info['strategy'],
                    'raw_score': round(raw_score, 3)
                })
            
            results.append(result)
        
        return results
    
    def predict(self, input_text):
        """Get best prediction"""
        candidates = self.predict_multiple(input_text, top_k=1)
        return candidates[0]['text'] if candidates else input_text


class InteractiveEnsembleTerminal:
    """
    Interactive terminal for ensemble autocomplete
    """
    
    def __init__(self, ensemble):
        self.ensemble = ensemble
        
        self.session_stats = {
            'queries': 0,
            'successful': 0,
            'avg_confidence': [],
            'strategies': Counter()
        }
    
    def display_suggestions(self, suggestions):
        """Display suggestions with nice formatting"""
        
        if not suggestions or suggestions[0].get('text') == '':
            print("   ⚠️  Không tìm thấy câu phù hợp")
            return
        
        print(f"\n   📋 Suggestions:")
        
        for i, sugg in enumerate(suggestions, 1):
            conf = sugg['confidence']
            sources = '+'.join(sugg.get('sources', ['unknown']))
            
            # Confidence color
            if conf >= 0.8:
                icon = "🟢"
                conf_label = "HIGH"
            elif conf >= 0.6:
                icon = "🟡"
                conf_label = "MED "
            else:
                icon = "🔴"
                conf_label = "LOW "
            
            # Agreement icon
            agreement_icon = "🤝" if sugg.get('agreement') else ""
            
            print(f"   {i}. {icon} [{conf*100:.0f}% {conf_label}] {agreement_icon} {sugg['text']}")
            print(f"      └─ Sources: [{sources}]")
    
    def show_detailed_analysis(self, input_text):
        """Show detailed prediction analysis"""
        
        print(f"\n{'─'*70}")
        print(f"🔍 DETAILED ANALYSIS")
        print(f"{'─'*70}")
        
        # Get predictions with metadata
        candidates = self.ensemble.predict_multiple(
            input_text, top_k=5, return_metadata=True
        )
        
        if not candidates:
            print("   No predictions available")
            return
        
        print(f"\n📝 Input: '{input_text}'")
        print(f"   Length: {len(input_text.split())} words")
        
        # Show weighting strategy
        if candidates:
            weights = candidates[0].get('weights', {})
            strategy = candidates[0].get('strategy', 'unknown')
            
            print(f"\n⚖️  Weighting Strategy: {strategy}")
            print(f"   Retrieval: {weights.get('retrieval', 0.5)*100:.0f}%")
            print(f"   LSTM:      {weights.get('lstm', 0.5)*100:.0f}%")
        
        # Show individual model predictions
        print(f"\n🔍 Individual Model Outputs:")
        
        # Retrieval
        print(f"\n   📚 Retrieval Model:")
        try:
            ret_preds = self.ensemble.retrieval.predict_multiple(input_text, top_k=3)
            for i, pred in enumerate(ret_preds[:3], 1):
                print(f"      {i}. [{pred['confidence']:.0%}] {pred['text']}")
        except Exception as e:
            print(f"      Error: {e}")
        
        # LSTM
        print(f"\n   🧠 LSTM Model:")
        try:
            lstm_preds = self.ensemble.lstm.predict_multiple(input_text, top_k=3)
            for i, pred in enumerate(lstm_preds[:3], 1):
                print(f"      {i}. [{pred['confidence']:.0%}] {pred['text']}")
        except Exception as e:
            print(f"      Error: {e}")
        
        # Ensemble
        print(f"\n   🎯 Ensemble Output:")
        for i, cand in enumerate(candidates[:3], 1):
            agreement = "🤝" if cand.get('agreement') else ""
            print(f"      {i}. [{cand['confidence']:.0%}] {agreement} {cand['text']}")
            print(f"         Sources: {cand.get('sources', [])}")
        
        print(f"{'─'*70}")
    
    def show_stats(self):
        """Show session statistics"""
        
        print(f"\n{'─'*70}")
        print(f"📊 SESSION STATISTICS")
        print(f"{'─'*70}")
        
        stats = self.session_stats
        
        print(f"\n📈 General:")
        print(f"   Total queries: {stats['queries']}")
        
        if stats['queries'] > 0:
            success_rate = stats['successful'] / stats['queries'] * 100
            print(f"   Successful: {stats['successful']} ({success_rate:.1f}%)")
        
        if stats['avg_confidence']:
            avg_conf = sum(stats['avg_confidence']) / len(stats['avg_confidence'])
            print(f"   Avg confidence: {avg_conf:.1%}")
        
        # Ensemble stats
        ens_stats = self.ensemble.stats
        print(f"\n📊 Ensemble Behavior:")
        print(f"   Total queries: {ens_stats['queries']}")
        print(f"   Retrieval dominant: {ens_stats['retrieval_dominant']}")
        print(f"   Balanced: {ens_stats['balanced']}")
        print(f"   Agreements: {ens_stats['agreements']}")
        
        if ens_stats['queries'] > 0:
            agree_rate = ens_stats['agreements'] / ens_stats['queries'] * 100
            print(f"   Agreement rate: {agree_rate:.1f}%")
    
    def show_help(self):
        """Show help message"""
        
        print("\n📖 Available Commands:")
        print("   • <text>           : Get suggestions")
        print("   • detail:<text>    : Show detailed analysis")
        print("   • stats            : Show session statistics")
        print("   • help             : Show this help")
        print("   • q / quit / exit  : Exit program")
        
        print("\n💡 Examples:")
        print("   >>> ăn quả")
        print("   >>> detail:có công")
        print("   >>> stats")
    
    def run_interactive(self):
        """Run interactive session"""
        
        print("\n" + "="*70)
        print("🎯 INTERACTIVE ENSEMBLE AUTOCOMPLETE")
        print("   Retrieval (50%) + LSTM (50%)")
        print("="*70)
        print("\n📖 Instructions:")
        print("   • Nhập text để nhận suggestions")
        print("   • Gõ 'detail:<text>' để xem phân tích chi tiết")
        print("   • Gõ 'stats' để xem thống kê")
        print("   • Gõ 'help' để xem hướng dẫn")
        print("   • Gõ 'q' để thoát")
        print("\n" + "-"*70)
        
        while True:
            try:
                # Get input
                print("\n" + "─"*70)
                user_input = input(">>> Nhập: ").strip()
                
                # Commands
                if user_input.lower() in ['q', 'quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower().startswith('detail:'):
                    text = user_input[7:].strip()
                    if text:
                        self.show_detailed_analysis(text)
                    else:
                        print("   ⚠️  Usage: detail:<text>")
                    continue
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if not user_input:
                    print("   ⚠️  Vui lòng nhập text")
                    continue
                
                # Get suggestions
                self.session_stats['queries'] += 1
                
                suggestions = self.ensemble.predict_multiple(
                    user_input, top_k=5
                )
                
                if suggestions and suggestions[0].get('confidence', 0) > 0:
                    self.session_stats['successful'] += 1
                    self.session_stats['avg_confidence'].append(
                        suggestions[0]['confidence']
                    )
                
                # Display
                self.display_suggestions(suggestions)
                
                # Show tip after first query
                if self.session_stats['queries'] == 1:
                    print(f"\n   💡 Tip: Gõ 'detail:{user_input}' để xem phân tích")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final stats
        print("\n" + "="*70)
        self.show_stats()
        print("="*70)


# ============================================================
# MAIN
# ============================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*70)
    print("🚀 LOADING ENSEMBLE MODEL")
    print("="*70)
    
    # Setup paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "trained_models"
    
    import sys
    sys.path.append(str(BASE_DIR / "backend" / "models"))
    
    # Load Retrieval model
    print(f"\n📚 Loading Retrieval model...")
    
    try:
        from retrieval import RetrievalModel
        
        retrieval = RetrievalModel()
        retrieval.load(MODEL_DIR / "retrieval_model.pkl")
        
        print(f"✓ Retrieval loaded ({len(retrieval.database)} sentences)")
    except Exception as e:
        print(f"❌ Failed to load Retrieval: {e}")
        return
    
    # Load LSTM model
    print(f"\n🧠 Loading LSTM model...")
    
    try:
        from lstm import LSTMCaDao, LSTMPredictor
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(
            MODEL_DIR / "lstm_cadao.pt",
            map_location=device,
            weights_only=False
        )
        
        vocab = checkpoint['vocab']
        
        model = LSTMCaDao(
            vocab_size=vocab.n_words,
            embed_dim=128,
            hidden_dim=256,
            num_layers=2
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        lstm_predictor = LSTMPredictor(model, vocab, device)
        
        print(f"✓ LSTM loaded (vocab: {vocab.n_words} words)")
    except Exception as e:
        print(f"❌ Failed to load LSTM: {e}")
        print(f"\n💡 Have you trained the LSTM model?")
        print(f"   Run: python lstm_cadao.py")
        return
    
    # Create ensemble
    print(f"\n🎯 Creating ensemble...")
    
    ensemble = RetrievalLSTMEnsemble(retrieval, lstm_predictor)
    
    print(f"✓ Ensemble ready!")
    
    # Run interactive terminal
    terminal = InteractiveEnsembleTerminal(ensemble)
    terminal.run_interactive()


if __name__ == "__main__":
    main()
