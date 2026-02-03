#!/bin/bash
# Setup and Activate Script
# THIS SCRIPT MUST BE SOURCED: source ./setup.sh
# NOT executed: ./setup.sh

# Check if script is being sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    echo "❌ ERROR: This script must be SOURCED, not executed!"
    echo ""
    echo "Usage: source ./setup.sh"
    echo ""
    exit 1
fi

echo "=========================================="
echo "🚀 Running NLP Project Setup"
echo "=========================================="

# Run config.py (in a subshell so errors don't affect current shell)
if python3 config.py; then
    echo ""
    echo "=========================================="
    echo "✅ Setup completed! Activating environment..."
    echo "=========================================="
    
    # Activate the environment in CURRENT shell
    source ./NLP/bin/activate
    
    echo ""
    echo "✅ Environment is now ACTIVE!"
    echo "You can now use all installed packages."
    echo ""
    echo "Type 'deactivate' to exit the environment."
else
    echo "❌ Setup failed. Please check the errors above."
    return 1
fi
