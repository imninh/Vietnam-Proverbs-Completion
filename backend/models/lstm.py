"""
LSTM MODEL FOR CA DAO COMPLETION
Simple implementation for 20K dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pickle


# ============================================================
# 1. VOCABULARY
# ============================================================

class Vocabulary:
    """Build and manage vocabulary"""
    
    def __init__(self):
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,  # Start of sequence
            '<EOS>': 3   # End of sequence
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.word_count = Counter()
        self.n_words = 4
    
    def add_sentence(self, sentence):
        """Add all words in sentence to vocabulary"""
        for word in sentence.strip().split():
            self.add_word(word)
    
    def add_word(self, word):
        """Add single word"""
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def encode(self, sentence, max_len=30):
        """Convert sentence to indices"""
        words = sentence.strip().split()
        indices = [self.word2idx['<SOS>']]
        
        for word in words:
            idx = self.word2idx.get(word, self.word2idx['<UNK>'])
            indices.append(idx)
        
        indices.append(self.word2idx['<EOS>'])
        
        # Padding
        if len(indices) < max_len:
            indices += [self.word2idx['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return indices
    
    def decode(self, indices):
        """Convert indices back to sentence"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word == '<EOS>' or word == '<PAD>':
                break
            if word not in ['<SOS>', '<UNK>']:
                words.append(word)
        
        return ' '.join(words)


# ============================================================
# 2. DATASET
# ============================================================

class CaDaoDataset(Dataset):
    """Dataset for Ca Dao completion"""
    
    def __init__(self, data, vocab, max_len=30):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode input and full sentence
        input_ids = self.vocab.encode(item['input'], self.max_len)
        target_ids = self.vocab.encode(item['full'], self.max_len)
        
        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long)
        }


# ============================================================
# 3. LSTM MODEL
# ============================================================

class LSTMCaDao(nn.Module):
    """
    Simple LSTM for sequence completion
    
    Architecture:
    - Embedding (vocab_size → embed_dim)
    - LSTM (embed_dim → hidden_dim, 2 layers)
    - Dropout (0.3)
    - Linear (hidden_dim → vocab_size)
    """
    
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, 
                 num_layers=2, dropout=0.3):
        super(LSTMCaDao, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len)
        
        Returns:
            output: (batch_size, seq_len, vocab_size)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_dim)
        
        # Project to vocabulary
        output = self.fc(lstm_out)  # (batch, seq_len, vocab_size)
        
        return output
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell


# ============================================================
# 4. TRAINER
# ============================================================

class LSTMTrainer:
    """Training pipeline for LSTM"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        
        self.model.to(device)
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward
            optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Reshape for loss calculation
            outputs = outputs.view(-1, self.vocab.n_words)
            targets = targets.view(-1)
            
            # Loss
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            
            # Clip gradients (prevent exploding)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                
                outputs = outputs.view(-1, self.vocab.n_words)
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, train_loader, val_loader, epochs=20, lr=0.001):
        """Full training loop"""
        
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING LSTM MODEL")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {lr}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        best_val_loss = float('inf')
        patience = 5
        no_improve = 0
        
        for epoch in range(epochs):
            print(f"\n{'─'*60}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'─'*60}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss = self.evaluate(val_loader, criterion)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss:   {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_lstm_model.pt')
                print(f"✓ Best model saved!")
                no_improve = 0
            else:
                no_improve += 1
            
            # Early stopping
            if no_improve >= patience:
                print(f"\n⚠️  Early stopping (no improvement for {patience} epochs)")
                break
        
        print(f"\n{'='*60}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab
        }, path)
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.vocab = checkpoint['vocab']


# ============================================================
# 5. INFERENCE
# ============================================================

class LSTMPredictor:
    """Generate predictions with LSTM"""
    
    def __init__(self, model, vocab, device='cpu'):
        self.model = model
        self.vocab = vocab
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def predict_multiple(self, partial_input, top_k=5, max_len=30, 
                        temperature=1.0):
        """
        Predict completions using beam search
        
        Args:
            partial_input: Input text
            top_k: Number of candidates
            max_len: Maximum length
            temperature: Sampling temperature (1.0 = normal)
        
        Returns:
            List of candidates with confidence
        """
        
        # Encode input
        input_ids = self.vocab.encode(partial_input, max_len)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Greedy decoding (simple)
        candidates = []
        
        with torch.no_grad():
            # Get model output
            outputs = self.model(input_tensor)
            
            # Get probabilities for each position
            probs = F.softmax(outputs[0] / temperature, dim=-1)
            
            # Decode from probabilities
            generated_ids = torch.argmax(probs, dim=-1).tolist()
            
            # Convert to text
            generated_text = self.vocab.decode(generated_ids)
            
            # Calculate confidence (average probability)
            max_probs = torch.max(probs, dim=-1)[0]
            confidence = torch.mean(max_probs).item()
            
            candidates.append({
                'text': generated_text,
                'confidence': round(confidence, 3),
                'model': 'lstm',
                'method': 'greedy'
            })
            
            # Generate more with sampling
            for temp in [0.8, 1.2]:
                probs_temp = F.softmax(outputs[0] / temp, dim=-1)
                
                # Sample instead of argmax
                sampled_ids = torch.multinomial(probs_temp, 1).squeeze(-1).tolist()
                sampled_text = self.vocab.decode(sampled_ids)
                
                if sampled_text != generated_text and sampled_text not in [c['text'] for c in candidates]:
                    max_probs_temp = torch.max(probs_temp, dim=-1)[0]
                    conf_temp = torch.mean(max_probs_temp).item() * 0.9
                    
                    candidates.append({
                        'text': sampled_text,
                        'confidence': round(conf_temp, 3),
                        'model': 'lstm',
                        'method': 'sampling'
                    })
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates[:top_k]
    
    def predict(self, partial_input):
        """Get best prediction"""
        candidates = self.predict_multiple(partial_input, top_k=1)
        return candidates[0]['text'] if candidates else partial_input


# ============================================================
# 6. TRAINING SCRIPT
# ============================================================

def train_lstm_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🚀 LSTM CA DAO MODEL - TRAINING")
    print("="*70)
    
    # Setup
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"   ⚠️  No GPU detected. Training will be slow!")
        print(f"   💡 Consider using Google Colab for GPU")
    
    # Load data
    print(f"\n📂 Loading data...")
    
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(DATA_DIR / "val.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Val:   {len(val_data)} samples")
    
    # Build vocabulary
    print(f"\n📚 Building vocabulary...")
    vocab = Vocabulary()
    
    for item in train_data:
        vocab.add_sentence(item['full'])
    
    print(f"✓ Vocabulary size: {vocab.n_words:,} words")
    
    # Create datasets
    print(f"\n🔄 Creating datasets...")
    train_dataset = CaDaoDataset(train_data, vocab)
    val_dataset = CaDaoDataset(val_data, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches:   {len(val_loader)}")
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = LSTMCaDao(
        vocab_size=vocab.n_words,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created")
    print(f"✓ Parameters: {num_params:,}")
    
    # Train
    trainer = LSTMTrainer(model, vocab, device)
    trainer.train(train_loader, val_loader, epochs=20, lr=0.001)
    
    # Save final
    save_path = MODEL_DIR / "lstm_cadao.pt"
    trainer.save_checkpoint(save_path)
    
    print(f"\n✓ Model saved to: {save_path}")
    
    # Quick test
    print(f"\n{'─'*60}")
    print("🧪 QUICK TEST")
    print(f"{'─'*60}")
    
    predictor = LSTMPredictor(model, vocab, device)
    
    test_inputs = ["ăn quả", "có công", "gần mực"]
    
    for inp in test_inputs:
        print(f"\n📝 Input: '{inp}'")
        candidates = predictor.predict_multiple(inp, top_k=3)
        
        for i, cand in enumerate(candidates, 1):
            print(f"   {i}. [{cand['confidence']:.0%}] {cand['text']}")
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    train_lstm_model()
