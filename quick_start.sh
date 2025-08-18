#!/bin/bash

# Railway RAG Project - Quick Start Script

echo "Railway RAG Project - Quick Start"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "backend.py" ]; then
    echo "Error: Please run this script from the railway_rag_project directory"
    exit 1
fi

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    echo "Python 3 found"
else
    echo "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
if python3 -c "import langchain, chromadb, fastapi" 2>/dev/null; then
    echo "Dependencies appear to be installed"
else
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Run setup check
echo "Running setup check..."
python3 setup.py

# Run system test
echo "Running system test..."
python3 test_setup.py

echo ""
echo "Quick start complete!"
echo ""
echo "Next steps:"
echo "  1. Start the API server:"
echo "     uvicorn fastapi_app:app --reload"
echo ""
echo "  2. Or run the Jupyter notebook:"
echo "     jupyter notebook example.ipynb"
echo ""
echo "  3. Or use directly in Python:"
echo "     python3 -c \"from backend import answer_question; print(answer_question('What is this about?'))\""
echo ""
