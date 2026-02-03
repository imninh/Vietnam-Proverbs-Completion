#!/usr/bin/env python3
"""
NLP Project Configuration and Setup Script
- Creates virtual environment (NLP)
- Installs all required libraries
- Clones and compiles KenLM
"""

import os
import sys
import subprocess
import platform

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_NAME = "NLP"
VENV_PATH = os.path.join(PROJECT_ROOT, VENV_NAME)
NGRAM_FOLDER = os.path.join(PROJECT_ROOT, "models", "n_gram")
KENLM_PATH = os.path.join(NGRAM_FOLDER, "kenlm")

# Determine pip executable path based on OS
if platform.system() == "Windows":
    PIP_EXECUTABLE = os.path.join(VENV_PATH, "Scripts", "pip")
    PYTHON_EXECUTABLE = os.path.join(VENV_PATH, "Scripts", "python")
else:
    PIP_EXECUTABLE = os.path.join(VENV_PATH, "bin", "pip")
    PYTHON_EXECUTABLE = os.path.join(VENV_PATH, "bin", "python")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def run_command(cmd, shell=True, check=True, description=""):
    """Run shell command with error handling"""
    if description:
        print(f"\n{'='*60}")
        print(f"📌 {description}")
        print(f"{'='*60}")
    
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=shell, capture_output=False)
    
    if check and result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        sys.exit(1)
    
    return result.returncode == 0

def venv_exists():
    """Check if virtual environment already exists"""
    return os.path.exists(VENV_PATH)

def create_venv():
    """Create Python virtual environment"""
    if venv_exists():
        print(f"✅ Virtual environment '{VENV_NAME}' already exists at {VENV_PATH}")
        return True
    
    print(f"\n📌 Creating virtual environment '{VENV_NAME}'...")
    if subprocess.run([sys.executable, "-m", "venv", VENV_PATH]).returncode != 0:
        print("❌ Failed to create virtual environment")
        return False
    
    print(f"✅ Virtual environment created successfully")
    return True

def install_dependencies():
    """Install required Python packages"""
    dependencies = [
        "pandas",
        "numpy",
        "nltk",
        "pyvi",
        "tqdm",
        "matplotlib",
        "seaborn",
        "kenlm @ https://github.com/kpu/kenlm/archive/master.zip"
    ]
    
    print(f"\n{'='*60}")
    print("📌 Installing Python Dependencies")
    print(f"{'='*60}")
    
    for package in dependencies:
        print(f"\n📦 Installing: {package}")
        cmd = f'"{PIP_EXECUTABLE}" install "{package}"'
        if not run_command(cmd, check=False):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    
    print(f"\n✅ Dependency installation completed")

def clone_and_compile_kenlm():
    """Clone and compile KenLM into n_gram folder"""
    # Ensure n_gram folder exists
    os.makedirs(NGRAM_FOLDER, exist_ok=True)
    
    if os.path.exists(KENLM_PATH):
        print(f"✅ KenLM folder already exists at {KENLM_PATH}")
        return True
    
    print(f"\n{'='*60}")
    print("📌 Cloning KenLM Repository")
    print(f"{'='*60}")
    
    # Clone KenLM
    clone_cmd = f'git clone https://github.com/kpu/kenlm.git "{KENLM_PATH}"'
    if not run_command(clone_cmd, description="Cloning KenLM from GitHub"):
        print("❌ Failed to clone KenLM")
        return False
    
    # Compile KenLM
    print(f"\n{'='*60}")
    print("📌 Compiling KenLM (This may take several minutes)")
    print(f"{'='*60}")
    
    build_dir = os.path.join(KENLM_PATH, "build")
    
    # Create build directory
    os.makedirs(build_dir, exist_ok=True)
    
    # Build commands
    cd_build = f'cd "{build_dir}"'
    cmake_cmd = 'cmake -DCMAKE_POLICY_DEFAULT_CMP0167=OLD -Wno-dev ..'
    make_cmd = 'make -j 4'
    
    # Combined command for cross-platform compatibility
    if platform.system() == "Windows":
        # For Windows, use different approach
        print("⚠️  Windows detected. You may need to install KenLM manually.")
        print(f"See: https://github.com/kpu/kenlm for Windows build instructions")
        return False
    else:
        # For Linux/Mac
        full_cmd = f'{cd_build} && {cmake_cmd} && {make_cmd}'
        if not run_command(full_cmd, description="Building KenLM"):
            print("❌ KenLM compilation failed")
            return False
    
    # Verify compilation
    lmplz_path = os.path.join(build_dir, "bin", "lmplz")
    if os.path.exists(lmplz_path):
        print(f"\n✅ KenLM compiled successfully!")
        print(f"   lmplz executable: {lmplz_path}")
        return True
    else:
        print(f"❌ KenLM compilation verification failed")
        return False

# ==============================================================================
# 3. MAIN SETUP ROUTINE
# ==============================================================================
def main():
    print(f"""
    {'='*60}
    🚀 NLP PROJECT SETUP
    {'='*60}
    Project Root: {PROJECT_ROOT}
    Virtual Env:  {VENV_PATH}
    N-gram Folder: {NGRAM_FOLDER}
    KenLM Path:   {KENLM_PATH}
    {'='*60}
    """)
    
    # Step 1: Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Clone and compile KenLM
    clone_and_compile_kenlm()
    
    # Print completion message
    print(f"""
    {'='*60}
    ✅ SETUP COMPLETED SUCCESSFULLY!
    {'='*60}
    
    Virtual Environment Location:
      {VENV_PATH}
    
    KenLM Location:
      {KENLM_PATH}/build/bin/lmplz
    
    Next Step:
      Run: source setup_and_activate.sh
    
    This will automatically activate the environment.
    {'='*60}
    """)

if __name__ == "__main__":
    main()