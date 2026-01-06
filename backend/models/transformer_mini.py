"""
TRANSFORMER MINI - TRAIN TỪ ĐẦU
File: backend/models/transformer_mini.py

Transformer đơn giản hóa để train trên dataset nhỏ (1,492 câu)
Inspired by "Attention Is All You Need" paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import pickle
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# ========== POSITIONAL ENCODING ==========
class PositionalEncoding(nn.Module):
    """
    Positional Encoding như trong paper Transformer
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # Tạo ma trận PE
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


# ========== MULTI-HEAD ATTENTION ==========
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers cho Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        """
        Split thành multiple heads
        (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split heads
        Q = self.split_heads(Q)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)  # (batch, num_heads, seq_len, d_k)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear
        output = self.W_o(context)
        
        return output


# ========== FEED FORWARD ==========
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ========== ENCODER LAYER ==========
class EncoderLayer(nn.Module):
    """
    Một encoder layer gồm:
    - Multi-Head Self-Attention
    - Feed Forward Network
    - Residual connections + Layer Norm
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention với residual
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward với residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ========== TRANSFORMER ENCODER ==========
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder stack
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.1, max_len=50):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        
        # Embedding + Positional Encoding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Embedding
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Qua các encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ========== TRANSFORMER MODEL ==========
class TransformerMini(nn.Module):
    """
    Transformer Mini cho Ca Dao completion
    Chỉ dùng Encoder (không cần Decoder vì task là classification-like)
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=2, d_ff=512, dropout=0.1):
        super(TransformerMini, self).__init__()
        
        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff, dropout)
        
        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        # Encode
        encoded = self.encoder(x, mask)
        
        # Project to vocab
        logits = self.fc_out(encoded)
        
        return logits


# ========== VOCABULARY (giống LSTM) ==========
class Vocabulary:
    """Vocabulary builder"""
    
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}
        self.word_count = Counter()
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
        self.word_count[word] += 1
    
    def encode(self, sentence, max_len=30):
        words = sentence.split()
        indices = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        if len(indices) < max_len:
            indices += [self.word2idx['<PAD>']] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
        
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, '<UNK>')
            if word in ['<PAD>', '<END>']:
                break
            if word not in ['<START>', '<UNK>']:
                words.append(word)
        return ' '.join(words)


# ========== DATASET ==========
class TransformerDataset(Dataset):
    def __init__(self, data, vocab, max_len=30):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids = self.vocab.encode(item['input'], self.max_len)
        target_ids = self.vocab.encode(item['full'], self.max_len)
        
        return {
            'input': torch.tensor(input_ids, dtype=torch.long),
            'target': torch.tensor(target_ids, dtype=torch.long)
        }


# ========== TRANSFORMER CA DAO (Wrapper) ==========
class TransformerCaDao:
    """Wrapper cho Transformer model"""
    
    def __init__(self):
        self.vocab = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_len = 30
        self.database = []
    
    def train(self, train_data, val_data=None, epochs=20, batch_size=32, lr=0.0001):
        """Train Transformer từ đầu"""
        
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING TRANSFORMER MINI")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {lr}")
        
        # Build vocab
        print(f"\n📚 Building vocabulary...")
        self.vocab = Vocabulary()
        
        for item in train_data:
            self.vocab.add_sentence(item['full'])
            if item['full'] not in self.database:
                self.database.append(item['full'])
        
        print(f"✓ Vocabulary size: {self.vocab.n_words:,} words")
        
        # Create datasets
        train_dataset = TransformerDataset(train_data, self.vocab, self.max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            val_dataset = TransformerDataset(val_data, self.vocab, self.max_len)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        print(f"\n🏗️  Creating Transformer Mini...")
        self.model = TransformerMini(
            vocab_size=self.vocab.n_words,
            d_model=256,      # Nhỏ để train nhanh
            num_heads=4,      # Ít head hơn
            num_layers=2,     # 2 layers thay vì 6
            d_ff=512,
            dropout=0.1
        ).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model parameters: {num_params:,}")
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Training loop
        print(f"\n🚀 Training...")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward
                outputs = self.model(inputs)
                
                # Reshape
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                # Loss
                loss = criterion(outputs, targets)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if val_data:
                val_loss = self._evaluate(val_loader, criterion)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        print(f"\n✅ Training complete!")
    
    def _evaluate(self, dataloader, criterion):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def generate(self, partial_input, max_new_words=10):
        """Generate completion"""
        self.model.eval()
        
        input_ids = self.vocab.encode(partial_input, self.max_len)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = partial_input.split()
        
        with torch.no_grad():
            for _ in range(max_new_words):
                outputs = self.model(input_tensor)
                
                # Lấy prediction cuối cùng
                last_logits = outputs[0, len(generated)-1, :]
                
                # Greedy decoding
                next_token = torch.argmax(last_logits).item()
                
                next_word = self.vocab.idx2word.get(next_token, '<UNK>')
                
                if next_word in ['<END>', '<PAD>', '<UNK>']:
                    break
                
                generated.append(next_word)
        
        return ' '.join(generated)
    
    def predict_multiple(self, partial_input, top_k=3):
        """Generate multiple candidates"""
        # Simple: generate 1 lần, return với confidence giả lập
        try:
            generated = self.generate(partial_input)
            return [{
                'text': generated,
                'confidence': 0.75,
                'model': 'transformer'
            }]
        except:
            if self.database:
                import random
                return [{
                    'text': random.choice(self.database),
                    'confidence': 0.2,
                    'model': 'transformer',
                    'method': 'fallback'
                }]
            return []
    
    def predict(self, partial_input):
        candidates = self.predict_multiple(partial_input, top_k=1)
        return candidates[0]['text'] if candidates else partial_input
    
    def evaluate(self, test_data):
        """Evaluate"""
        print(f"\n{'─'*60}")
        print(f"📊 EVALUATING TRANSFORMER")
        print(f"{'─'*60}")
        
        exact_correct = 0
        total = min(len(test_data), 100)
        
        for item in tqdm(test_data[:total]):
            predicted = self.predict(item['input'])
            if predicted == item['full']:
                exact_correct += 1
        
        exact_acc = exact_correct / total
        
        print(f"Test samples: {total}")
        print(f"Exact matches: {exact_correct} ({exact_acc:.1%})")
        
        return {'exact_accuracy': exact_acc}
    
    def save(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'database': self.database
        }, file_path)
        print(f"✓ Model saved to {file_path}")
    
    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        
        self.vocab = checkpoint['vocab']
        self.database = checkpoint['database']
        
        self.model = TransformerMini(
            vocab_size=self.vocab.n_words,
            d_model=256,
            num_heads=4,
            num_layers=2,
            d_ff=512
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded from {file_path}")


# ========== MAIN ==========
def train_transformer_cadao():
    """Training script"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data" / "processed"
    MODEL_DIR = BASE_DIR / "trained_models"
    
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data...")
    
    with open(DATA_DIR / "train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(DATA_DIR / "val.json", 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    with open(DATA_DIR / "test.json", 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✓ Train: {len(train_data)}")
    print(f"✓ Val:   {len(val_data)}")
    print(f"✓ Test:  {len(test_data)}")
    
    # Train
    model = TransformerCaDao()
    model.train(train_data, val_data, epochs=20, batch_size=32, lr=0.0001)
    
    # Test
    print(f"\n{'─'*60}")
    print("🧪 TEST PREDICTIONS")
    print(f"{'─'*60}")
    
    for inp in ["ăn quả", "có công", "gần mực"]:
        print(f"\n📝 '{inp}' → {model.predict(inp)}")
    
    # Evaluate
    metrics = model.evaluate(test_data)
    
    # Save
    model_path = MODEL_DIR / "transformer_model.pt"
    model.save(model_path)
    
    print(f"\n✅ DONE! Accuracy: {metrics['exact_accuracy']:.1%}")


if __name__ == "__main__":
    train_transformer_cadao()