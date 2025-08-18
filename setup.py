#!/usr/bin/env python3
"""
Setup script for Railway RAG Project
This script helps initialize the project and check dependencies.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path

def check_ollama():
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama is running and accessible")
            return True
        else:
            print("Ollama is running but not responding correctly")
            return False
    except requests.exceptions.RequestException:
        print("Ollama is not running or not accessible")
        print("   Please start Ollama with: ollama serve")
        return False

def check_models():
    """Check if required models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            required_models = [
                "qwen3",
                "deepseek-r1", 
                "gemma3",
                "nomic-embed-text"
            ]
            
            missing_models = []
            for model in required_models:
                if model in available_models:
                    print(f"{model} is available")
                else:
                    print(f"{model} is missing")
                    missing_models.append(model)
            
            if missing_models:
                print(f"\nPull missing models with:")
                for model in missing_models:
                    print(f"   ollama pull {model}")
                return False
            else:
                print("All required models are available")
                return True
        else:
            print("Could not check models - Ollama not responding")
            return False
    except Exception as e:
        print(f"Error checking models: {e}")
        return False

def check_chroma_db():
    """Check if ChromaDB exists and has data"""
    db_path = Path("railway_chroma_db")
    if db_path.exists():
        # Check if it has the expected structure
        sqlite_file = db_path / "chroma.sqlite3"
        if sqlite_file.exists():
            print("ChromaDB exists with data")
            return True
        else:
            print("ChromaDB directory exists but appears empty")
            return False
    else:
        print("ChromaDB not found - you'll need to run the notebook to create it")
        return False

def main():
    """Main setup function"""
    print("Railway RAG Project Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    else:
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
    
    # Check Ollama
    ollama_ok = check_ollama()
    
    # Check models if Ollama is running
    models_ok = False
    if ollama_ok:
        models_ok = check_models()
    
    # Check ChromaDB
    chroma_ok = check_chroma_db()
    
    print("\n" + "=" * 40)
    print("Setup Summary:")
    print(f"   Ollama: {'OK' if ollama_ok else 'FAIL'}")
    print(f"   Models: {'OK' if models_ok else 'FAIL'}")
    print(f"   ChromaDB: {'OK' if chroma_ok else 'FAIL'}")
    
    if ollama_ok and models_ok and chroma_ok:
        print("\nSetup complete! You can now:")
        print("   • Run: python -m jupyter notebook example.ipynb")
        print("   • Run: uvicorn fastapi_app:app --reload")
        print("   • Import and use: from backend import answer_question")
    else:
        print("\nSetup incomplete. Please address the issues above.")
        if not ollama_ok:
            print("   • Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
            print("   • Start Ollama: ollama serve")
        if not models_ok:
            print("   • Pull required models (see commands above)")
        if not chroma_ok:
            print("   • Run the Jupyter notebook to create the vector database")

if __name__ == "__main__":
    main()
