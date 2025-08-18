#!/usr/bin/env python3
"""
Test script for Railway RAG Project
This script tests the basic functionality of the RAG system.
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from backend import answer_question
        print("backend.py imports successfully")
        return True
    except ImportError as e:
        print(f"Failed to import backend: {e}")
        return False

def test_ollama_embeddings():
    """Test Ollama embeddings"""
    print("Testing Ollama embeddings...")
    
    try:
        from ollama_embeddings import OllamaEmbeddings
        embed_model = OllamaEmbeddings(model_name="nomic-embed-text")
        
        # Test embedding a simple text
        test_text = "This is a test document about railways."
        embedding = embed_model.embed_query(test_text)
        
        if len(embedding) > 0:
            print(f"Ollama embeddings working (vector size: {len(embedding)})")
            return True
        else:
            print("Ollama embeddings returned empty vector")
            return False
    except Exception as e:
        print(f"Ollama embeddings failed: {e}")
        return False

def test_chroma_db():
    """Test ChromaDB access"""
    print("Testing ChromaDB...")
    
    try:
        from langchain_community.vectorstores import Chroma
        from ollama_embeddings import OllamaEmbeddings
        
        embed_model = OllamaEmbeddings(model_name="nomic-embed-text")
        
        # Try to load the existing ChromaDB
        vector_store = Chroma(
            persist_directory="railway_chroma_db",
            embedding_function=embed_model,
            collection_name="railway_document_embeddings"
        )
        
        # Check if there are any documents
        collection = vector_store._collection
        count = collection.count()
        
        if count > 0:
            print(f"ChromaDB loaded successfully with {count} documents")
            return True
        else:
            print("ChromaDB loaded but contains no documents")
            return False
    except Exception as e:
        print(f"ChromaDB test failed: {e}")
        return False

def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("Testing RAG pipeline...")
    
    try:
        from backend import answer_question
        
        # Test with a simple question
        question = "What is this document about?"
        answer = answer_question(question, llm_model="qwen3")
        
        if answer and len(answer.strip()) > 0:
            print("RAG pipeline working")
            print(f"   Question: {question}")
            print(f"   Answer: {answer[:100]}...")
            return True
        else:
            print("RAG pipeline returned empty answer")
            return False
    except Exception as e:
        print(f"RAG pipeline failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Railway RAG Project - System Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Ollama Embeddings", test_ollama_embeddings),
        ("ChromaDB", test_chroma_db),
        ("RAG Pipeline", test_rag_pipeline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("Test Results:")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("   • Run: uvicorn fastapi_app:app --reload")
        print("   • Visit: http://localhost:8000/docs")
        print("   • Or use: python -m jupyter notebook example.ipynb")
    else:
        print("Some tests failed. Please check the setup and dependencies.")
        print("\nCommon issues:")
        print("   • Make sure Ollama is running: ollama serve")
        print("   • Check that all models are pulled")
        print("   • Verify ChromaDB exists and has data")

if __name__ == "__main__":
    main()
