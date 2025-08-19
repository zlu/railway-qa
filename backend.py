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
DOOR_CONTROL_COLLECTION = "door_control_embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"

# Choose your LLM model (can be set per request)
DEFAULT_LLM_MODEL = "gemma3"  # Options: "gemma3", "mistral"

# Load embedding model
embed_model = OllamaEmbeddings(model_name=EMBED_MODEL_NAME)

# Load vector store
door_control_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embed_model,
    collection_name=DOOR_CONTROL_COLLECTION
)

# Set up retriever with more documents for complete answers
door_control_retriever = door_control_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 8}  # Increased for better coverage
)

# User level definitions
USER_LEVELS = {
    "beginner": {
        "name": "Beginner",
        "description": "Novice users who need basic explanations and detailed steps",
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
        "description": "Experienced professionals who need technical details and advanced information",
        "characteristics": [
            "Need technical specifications and parameters",
            "Prefer professional terminology",
            "Focus on advanced troubleshooting",
            "Need system integration information",
            "Focus on best practices and optimization"
        ]
    }
}



def create_level_appropriate_prompt(question: str, user_level: str, context: str) -> str:
    """
    Create a prompt that produces answers with appropriate technical depth for the user level.
    """
    level_info = USER_LEVELS[user_level]
    
    if user_level == "beginner":
        prompt_template = f"""You are a railway maintenance trainer helping a {level_info['name']} ({level_info['description']}).

IMPORTANT: You MUST base your answer EXCLUSIVELY on the following document content. Do not add information that is not present in the provided context.

Document content:
{context}

User question: {question}

Instructions for {level_info['name']} response:
1. Use ONLY the information provided in the document content above
2. Explain technical concepts using simple, everyday language
3. When mentioning technical terms, ALWAYS explain what they mean in simple terms
4. Use clear, step-by-step explanations that someone new to the field can follow
5. Avoid jargon unless absolutely necessary, and when used, explain it
6. Focus on practical understanding rather than technical precision
7. Use phrases like "this means", "in other words", "essentially", "basically" to explain concepts
8. Keep the response concise but educational

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.
IMPORTANT: Do not start with phrases like "Okay, here's" or "Sure, here's". Start directly with the answer.
IMPORTANT: Every technical term must be accompanied by a simple explanation.

Please provide a beginner-friendly answer based on the document content:"""
    
    else:  # expert level
        prompt_template = f"""You are a senior railway maintenance expert providing technical support for a {level_info['name']} ({level_info['description']}).

IMPORTANT: You MUST base your answer EXCLUSIVELY on the following document content. Do not add information that is not present in the provided context.

Document content:
{context}

User question: {question}

Instructions for {level_info['name']} response:
1. Use ONLY the information provided in the document content above
2. Use extensive domain-specific technical terminology and industry jargon
3. Assume the user has deep technical knowledge of railway systems and maintenance procedures
4. Provide precise technical specifications and professional terminology
5. Focus on technical accuracy and professional standards
6. Use abbreviated forms and technical shorthand where appropriate
7. Include technical terms like DCU, fault codes, circuit breakers, TDIC, etc. without explanation
8. Keep the response concise and technically precise
9. Do NOT explain technical terms - assume the user knows them

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.
IMPORTANT: Do not start with phrases like "Based on the provided documentation" or "Sure, here's". Start directly with the answer.
IMPORTANT: Use maximum technical terminology without explanations.

Please provide a technical expert answer based on the document content:"""
    
    return prompt_template

def answer_question(question: str, llm_model: Optional[str] = None, 
                   user_level: str = "beginner") -> str:
    """
    Run the RAG pipeline with user level adaptation.
    
    Args:
        question: The question to answer
        llm_model: The LLM model to use (default: gemma3)
        user_level: "beginner" or "expert" (default: beginner)
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Use door control retriever
    retriever = door_control_retriever
    print(f"Searching door control maintenance guide...")
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    
    if not docs:
        return "I could not find relevant information in the available documents to answer your question."
    
    # Create context from retrieved documents
    context_parts = []
    for i, doc in enumerate(docs, 1):
        context_parts.append(f"Document {i}:\n{doc.page_content}")
    
    context = "\n\n".join(context_parts)
    
    # Create level-appropriate prompt
    prompt = create_level_appropriate_prompt(question, user_level, context)
    
    # Create LLM with custom prompt
    llm = Ollama(model=llm_model)
    
    # Get response with level adaptation
    response = llm.invoke(prompt)
    
    return response



def get_user_level_info() -> Dict:
    """Get information about available user levels for the frontend."""
    return USER_LEVELS
