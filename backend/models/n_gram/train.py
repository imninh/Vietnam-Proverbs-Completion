#!/usr/bin/env python3
"""
KenLM Training Script
- Trains 5-gram language model on Vietnamese tokenized data
- Saves ARPA and binary format models
"""

import os
import subprocess
import sys

# Setup paths
NGRAM_DIR = os.path.dirname(os.path.abspath(__file__))
# Need to go up 2 levels: /models/n_gram/ -> /models/ -> /NLP_v01/
PROJECT_ROOT = os.path.dirname(os.path.dirname(NGRAM_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(NGRAM_DIR, "checkpoint")
KENLM_BIN_DIR = os.path.join(NGRAM_DIR, "kenlm", "build", "bin")

INPUT_FILE = os.path.join(DATA_DIR, "train_data_seg.txt")
ARPA_FILE = os.path.join(CHECKPOINT_DIR, "model.arpa")
BINARY_FILE = os.path.join(CHECKPOINT_DIR, "model.bin")

def train_model():
    """Train 5-gram KenLM model"""
    print("\n[Step 1] Training KenLM 5-gram model...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return False
    
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        print(f"[INFO] Created checkpoint directory: {CHECKPOINT_DIR}")
    
    lmplz_cmd = os.path.join(KENLM_BIN_DIR, "lmplz")
    if not os.path.exists(lmplz_cmd):
        print(f"[ERROR] lmplz not found: {lmplz_cmd}")
        return False
    
    # Run lmplz
    cmd = [
        lmplz_cmd,
        "-o", "5",
        "--text", INPUT_FILE,
        "--arpa", ARPA_FILE,
        "--discount_fallback",
        "--skip_symbols"
    ]
    
    try:
        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[SUCCESS] ARPA model created")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Training failed: {e.stderr}")
        return False
    
    # Build binary
    print("\n[Step 2] Compiling to binary format...")
    build_binary_cmd = os.path.join(KENLM_BIN_DIR, "build_binary")
    if not os.path.exists(build_binary_cmd):
        print(f"[ERROR] build_binary not found: {build_binary_cmd}")
        return False
    
    cmd = [build_binary_cmd, ARPA_FILE, BINARY_FILE]
    
    try:
        print(f"[INFO] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[SUCCESS] Binary model created")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Binary compilation failed: {e.stderr}")
        return False
    
    return True

def verify_model():
    """Verify model files were created"""
    print("\n[Step 3] Verifying model files...")
    
    if os.path.exists(ARPA_FILE):
        arpa_size = os.path.getsize(ARPA_FILE) / (1024 * 1024)
        print(f"  ARPA: {ARPA_FILE} ({arpa_size:.2f} MB)")
    else:
        print(f"  [ERROR] ARPA file not found")
        return False
    
    if os.path.exists(BINARY_FILE):
        bin_size = os.path.getsize(BINARY_FILE) / (1024 * 1024)
        print(f"  Binary: {BINARY_FILE} ({bin_size:.2f} MB)")
    else:
        print(f"  [ERROR] Binary file not found")
        return False
    
    print("[SUCCESS] Model verification complete")
    return True

def main():
    print(f"""
    {'='*60}
    KENLM TRAINING PIPELINE
    {'='*60}
    """)
    
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Input Data: {INPUT_FILE}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"KenLM Binary: {KENLM_BIN_DIR}")
    print()
    
    if not train_model():
        return False
    
    if not verify_model():
        return False
    
    print(f"\n{'='*60}")
    print("[COMPLETE] Training finished successfully!")
    print(f"{'='*60}\n")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)