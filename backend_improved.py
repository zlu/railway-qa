#!/usr/bin/env python3
"""
Improved Railway RAG System Backend
Better RAG implementation with explicit content usage and user level differentiation
"""

import os
from typing import Optional, Dict, List
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ollama_embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import json

# Settings (customize as needed)
PERSIST_DIRECTORY = "railway_chroma_db"
RAILWAY_COLLECTION = "railway_document_embeddings"
DOOR_CONTROL_COLLECTION = "door_control_embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"

# Choose your LLM model (can be set per request)
DEFAULT_LLM_MODEL = "gemma3"

# Load embedding model
embed_model = OllamaEmbeddings(model_name=EMBED_MODEL_NAME)

# Load vector stores
railway_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embed_model,
    collection_name=RAILWAY_COLLECTION
)

door_control_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embed_model,
    collection_name=DOOR_CONTROL_COLLECTION
)

# Set up retrievers with more documents for complete answers
railway_retriever = railway_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 8}  # Increased for better coverage
)

door_control_retriever = door_control_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 8}  # Increased for better coverage
)

# User level definitions
USER_LEVELS = {
    "beginner": {
        "name": "Beginner",
        "description": "Novice users who need comprehensive explanations and detailed steps",
        "characteristics": [
            "Need detailed background explanations",
            "Prefer step-by-step guidance",
            "Require safety reminders",
            "Prefer simple and understandable language",
            "Need more examples and analogies"
        ]
    },
    "expert": {
        "name": "Expert", 
        "description": "Experienced professionals who need concise technical information",
        "characteristics": [
            "Need technical specifications and parameters",
            "Prefer professional terminology",
            "Focus on advanced troubleshooting",
            "Need system integration information",
            "Focus on best practices and optimization"
        ]
    }
}

def create_improved_prompt(question: str, user_level: str, context: str) -> str:
    """
    Create an improved prompt that explicitly instructs the LLM to use the provided context.
    """
    level_info = USER_LEVELS[user_level]
    
    if user_level == "beginner":
        prompt_template = f"""You are a professional railway maintenance trainer helping a {level_info['name']} ({level_info['description']}).

IMPORTANT: You MUST base your answer EXCLUSIVELY on the following document content. Do not add information that is not present in the provided context.

Document content:
{context}

User question: {question}

Instructions for {level_info['name']} response:
1. Use ONLY the information provided in the document content above
2. Provide a comprehensive, detailed explanation with step-by-step instructions
3. Include all relevant details, safety precautions, and background information
4. Use simple, clear language that a beginner can understand
5. Organize the answer in a logical, easy-to-follow structure
6. If the document mentions specific tools, procedures, or technical terms, include them exactly as stated

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.

Please provide a detailed answer based on the document content:"""
    
    else:  # expert level
        prompt_template = f"""You are a senior railway maintenance expert providing technical support for a {level_info['name']} ({level_info['description']}).

IMPORTANT: You MUST base your answer EXCLUSIVELY on the following document content. Do not add information that is not present in the provided context.

Document content:
{context}

User question: {question}

Instructions for {level_info['name']} response:
1. Use ONLY the information provided in the document content above
2. Provide a concise, technical summary focusing on key points
3. Use professional terminology and technical specifications
4. Focus on actionable steps and technical details
5. Keep the response brief but comprehensive
6. If the document mentions specific tools, procedures, or technical terms, include them exactly as stated

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.

Please provide a concise technical answer based on the document content:"""
    
    return prompt_template

def answer_question_improved(question: str, llm_model: Optional[str] = None, 
                           document_type: str = "railway", user_level: Optional[str] = None) -> str:
    """
    Improved RAG pipeline with better context utilization and user level adaptation.
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Choose retriever based on document type
    if document_type.lower() in ["door_control", "door", "maintenance"]:
        retriever = door_control_retriever
        print(f"Searching door control maintenance guide...")
    else:
        retriever = railway_retriever
        print(f"Searching railway documents...")
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return "I could not find relevant information in the available documents to answer your question."
    
    # Create context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"Document {i}:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create improved prompt
    prompt = create_improved_prompt(question, user_level, context)
    
    # Create LLM with custom prompt
    llm = Ollama(model=llm_model)
    
    # Get response with improved context utilization
    response = llm.invoke(prompt)
    
    return response

def answer_question_combined_improved(question: str, llm_model: Optional[str] = None, 
                                    user_level: Optional[str] = None) -> str:
    """
    Improved combined search with better context utilization.
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Get documents from both collections
    railway_docs = railway_retriever.get_relevant_documents(question)
    door_control_docs = door_control_retriever.get_relevant_documents(question)
    
    # Combine documents
    all_docs = railway_docs + door_control_docs
    
    if not all_docs:
        return "I could not find relevant information in the available documents to answer your question."
    
    # Create context from all retrieved documents
    context_parts = []
    for i, doc in enumerate(all_docs, 1):
        context_parts.append(f"Document {i}:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create improved prompt for combined search
    prompt = create_improved_prompt(question, user_level, context)
    
    # Create LLM with custom prompt
    llm = Ollama(model=llm_model)
    
    # Get response with improved context utilization
    response = llm.invoke(prompt)
    
    return response

def get_user_level_info() -> Dict:
    """Get information about available user levels for the frontend."""
    return USER_LEVELS

def test_retrieval_quality(question: str, document_type: str = "door_control"):
    """
    Test function to check what documents are being retrieved and their relevance.
    """
    if document_type.lower() in ["door_control", "door", "maintenance"]:
        retriever = door_control_retriever
    else:
        retriever = railway_retriever
    
    docs = retriever.get_relevant_documents(question)
    
    print(f"Retrieved {len(docs)} documents for question: {question}")
    print("=" * 80)
    
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}:")
        print(f"Content: {doc.page_content[:300]}...")
        print("-" * 40)
    
    return docs
