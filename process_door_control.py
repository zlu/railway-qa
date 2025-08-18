#!/usr/bin/env python3
"""
Script to process the Door Control Units Maintenance Guideline PDF
and add it to the ChromaDB for RAG queries.
"""

import os
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama_embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import shutil

def process_door_control_pdf():
    """Process the Door Control Units Maintenance Guideline PDF"""
    
    pdf_path = "data/Door control units maintenance guideline.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return False
    
    print("Processing Door Control Units Maintenance Guideline PDF...")
    
    # 1. Load and parse the PDF
    reader = PdfReader(pdf_path)
    docs = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            docs.append(Document(
                page_content=text, 
                metadata={
                    "page": i+1,
                    "source": "Door Control Units Maintenance Guideline",
                    "document_type": "maintenance_guideline"
                }
            ))
    
    print(f"Extracted {len(docs)} pages from PDF")
    
    # 2. Split into chunks with larger size for more complete answers
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Increased from 1000 for more complete answers
        chunk_overlap=400,  # Increased overlap for better context
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    
    # 3. Load embedding model
    embed_model = OllamaEmbeddings(model_name="nomic-embed-text")
    
    # 4. Create new collection for door control document
    door_control_collection = "door_control_embeddings"
    
    # Check if collection already exists and remove it
    try:
        existing_store = Chroma(
            persist_directory="railway_chroma_db",
            embedding_function=embed_model,
            collection_name=door_control_collection
        )
        existing_store._collection.delete(where={})
        print(f"Cleared existing collection: {door_control_collection}")
    except:
        pass
    
    # 5. Create vector store with door control chunks
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory="railway_chroma_db",
        collection_name=door_control_collection
    )
    
    print(f"Successfully added {len(chunks)} chunks to ChromaDB collection: {door_control_collection}")
    
    # 6. Test the retrieval
    retriever = vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 6}  # Increased k for more complete answers
    )
    
    # Test query
    test_query = "What are the maintenance steps for door control units?"
    docs = retriever.get_relevant_documents(test_query)
    
    print(f"\nTest retrieval for: '{test_query}'")
    print(f"Retrieved {len(docs)} relevant documents")
    print(f"Total content length: {sum(len(doc.page_content) for doc in docs)} characters")
    
    return True

def update_backend_for_door_control():
    """Update backend.py to include door control collection"""
    
    backend_content = '''import os
from typing import Optional
from langchain_community.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from ollama_embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Settings (customize as needed)
PERSIST_DIRECTORY = "railway_chroma_db"
RAILWAY_COLLECTION = "railway_document_embeddings"
DOOR_CONTROL_COLLECTION = "door_control_embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"

# Choose your LLM model (can be set per request)
DEFAULT_LLM_MODEL = "qwen3"

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

def answer_question(question: str, llm_model: Optional[str] = None, document_type: str = "railway") -> str:
    """
    Run the RAG pipeline: retrieve relevant chunks and answer with the specified LLM.
    
    Args:
        question: The question to answer
        llm_model: The LLM model to use (default: qwen3)
        document_type: "railway" or "door_control" to specify which document to search
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Choose retriever based on document type
    if document_type.lower() in ["door_control", "door", "maintenance"]:
        retriever = door_control_retriever
        print(f"Searching Door Control Maintenance Guideline...")
    else:
        retriever = railway_retriever
        print(f"Searching Railway Documentation...")
    
    qa = RetrievalQA.from_chain_type(
        llm=Ollama(model=llm_model),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa({"query": question})
    return result["result"]

def answer_question_combined(question: str, llm_model: Optional[str] = None) -> str:
    """
    Search both document collections and provide a combined answer.
    """
    llm_model = llm_model or DEFAULT_LLM_MODEL
    
    # Get documents from both collections
    railway_docs = railway_retriever.get_relevant_documents(question)
    door_control_docs = door_control_retriever.get_relevant_documents(question)
    
    # Combine documents
    all_docs = railway_docs + door_control_docs
    
    # Create a temporary retriever with combined documents
    from langchain_core.documents import Document
    
    # Create a temporary vector store with combined documents
    temp_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embed_model
    )
    
    temp_retriever = temp_store.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 8}  # More documents for comprehensive answers
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=Ollama(model=llm_model),
        chain_type="stuff",
        retriever=temp_retriever,
        return_source_documents=True
    )
    
    result = qa({"query": question})
    return result["result"]
'''
    
    with open("backend.py", "w") as f:
        f.write(backend_content)
    
    print("Updated backend.py to support door control document")

if __name__ == "__main__":
    # Process the door control PDF
    if process_door_control_pdf():
        # Update the backend
        update_backend_for_door_control()
        print("\nDoor Control Units Maintenance Guideline successfully processed!")
        print("You can now ask questions about door control maintenance.")
    else:
        print("Failed to process the PDF.")
