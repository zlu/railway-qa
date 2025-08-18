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
    search_kwargs={"k": 6}
)

door_control_retriever = door_control_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 6}
)

# User level definitions and learning patterns
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

# Learning patterns for autonomous adaptation
LEARNING_PATTERNS = {
    "beginner": {
        "key_phrases": ["basic", "simple", "steps", "safety", "check", "start", "beginner", "how to"],
        "avoid_terms": ["advanced", "complex", "professional", "technical specifications", "system integration"],
        "preferred_structure": "step_by_step",
        "detail_level": "high"
    },
    "expert": {
        "key_phrases": ["technical specifications", "parameters", "troubleshooting", "system", "optimization", "integration", "advanced", "professional"],
        "avoid_terms": ["basic", "simple", "start", "beginner"],
        "preferred_structure": "technical",
        "detail_level": "comprehensive"
    }
}

def analyze_user_level_from_question(question: str) -> str:
    """
    Analyze the question to determine user level based on language patterns and content.
    This provides autonomous learning capability.
    """
    question_lower = question.lower()
    
    # Count level indicators
    beginner_indicators = sum(1 for phrase in LEARNING_PATTERNS["beginner"]["key_phrases"] 
                            if phrase in question_lower)
    expert_indicators = sum(1 for phrase in LEARNING_PATTERNS["expert"]["key_phrases"] 
                           if phrase in question_lower)
    
    # Analyze question complexity
    word_count = len(question.split())
    technical_terms = ["fault", "troubleshoot", "system", "integration", "parameters", "specifications", "optimization", "diagnostic", "advanced", "technical"]
    technical_count = sum(1 for term in technical_terms if term in question_lower)
    
    # Determine level based on analysis
    if expert_indicators > beginner_indicators or technical_count > 2:
        return "expert"
    elif beginner_indicators > expert_indicators or word_count < 10:
        return "beginner"
    else:
        # Default to beginner for safety
        return "beginner"

def create_level_appropriate_prompt(question: str, user_level: str, context: str) -> str:
    """
    Create a level-appropriate prompt that adapts to user expertise.
    This enables autonomous learning and adaptation.
    """
    level_info = USER_LEVELS[user_level]
    learning_pattern = LEARNING_PATTERNS[user_level]
    
    if user_level == "beginner":
        prompt_template = f"""You are a professional railway maintenance trainer helping a {level_info['name']} ({level_info['description']}).

Based on the following document content, please provide a detailed and easy-to-understand answer for this {level_info['name']}:

Document content:
{context}

User question: {question}

Please answer according to the following requirements:
1. Use simple and easy-to-understand language, avoid excessive technical terms
2. Provide detailed background explanations and step-by-step instructions
3. Emphasize safety precautions
4. Use concrete examples and analogies
5. Organize the answer in steps for easy understanding
6. If technical operations are involved, please explain the purpose and precautions of each step in detail

IMPORTANT: Always respond in English, regardless of the language of the question.

Please ensure the answer is accurate, comprehensive, and suitable for the {level_info['name']}'s understanding level."""
    
    else:  # expert level
        prompt_template = f"""You are a senior railway maintenance expert providing technical support for a {level_info['name']} ({level_info['description']}).

Based on the following document content, please provide a professional and comprehensive answer for this {level_info['name']}:

Document content:
{context}

User question: {question}

Please answer according to the following requirements:
1. Use professional terminology and technical specifications
2. Provide detailed technical parameters and system information
3. Include advanced troubleshooting and diagnostic methods
4. Cover system integration and optimization recommendations
5. Provide best practices and industry standards
6. Include relevant technical specifications and standards

IMPORTANT: Always respond in English, regardless of the language of the question.

Please ensure the answer is accurate, comprehensive, and suitable for the {level_info['name']}'s professional level."""
    
    return prompt_template

def answer_question(question: str, llm_model: Optional[str] = None, 
                   document_type: str = "railway", user_level: Optional[str] = None) -> str:
    """
    Run the RAG pipeline with user level adaptation and autonomous learning.
    
    Args:
        question: The question to answer
        llm_model: The LLM model to use (default: gemma3)
        document_type: "railway", "door_control" to specify which document to search
        user_level: "beginner" or "expert" - if None, will be auto-detected
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Autonomous learning: detect user level if not provided
    if user_level is None:
        user_level = analyze_user_level_from_question(question)
        print(f"Auto-detected user level: {USER_LEVELS[user_level]['name']}")
    
    # Choose retriever based on document type
    if document_type.lower() in ["door_control", "door", "maintenance"]:
        retriever = door_control_retriever
        print(f"Searching door control maintenance guide...")
    else:
        retriever = railway_retriever
        print(f"Searching railway documents...")
    
    # Get relevant documents
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create level-appropriate prompt
    prompt = create_level_appropriate_prompt(question, user_level, context)
    
    # Create LLM with custom prompt
    llm = Ollama(model=llm_model)
    
    # Get response with level adaptation
    response = llm.invoke(prompt)
    
    return response

def answer_question_combined(question: str, llm_model: Optional[str] = None, 
                           user_level: Optional[str] = None) -> str:
    """
    Search both document collections and provide a combined answer with user level adaptation.
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Autonomous learning: detect user level if not provided
    if user_level is None:
        user_level = analyze_user_level_from_question(question)
        print(f"Auto-detected user level: {USER_LEVELS[user_level]['name']}")
    
    # Get documents from both collections
    railway_docs = railway_retriever.get_relevant_documents(question)
    door_control_docs = door_control_retriever.get_relevant_documents(question)
    
    # Combine documents
    all_docs = railway_docs + door_control_docs
    
    # Create a temporary vector store with combined documents
    from langchain_core.documents import Document
    
    temp_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embed_model
    )
    
    temp_retriever = temp_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 8}  # More documents for comprehensive answers
    )
    
    # Get relevant documents
    docs = temp_retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create level-appropriate prompt for combined search
    prompt = create_level_appropriate_prompt(question, user_level, context)
    
    # Create LLM with custom prompt
    llm = Ollama(model=llm_model)
    
    # Get response with level adaptation
    response = llm.invoke(prompt)
    
    return response

def get_user_level_info() -> Dict:
    """Get information about available user levels for the frontend."""
    return USER_LEVELS
