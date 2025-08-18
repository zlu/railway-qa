from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend import answer_question, get_user_level_info

app = FastAPI(
    title="Railway RAG API",
    description="Intelligent RAG system with user-level adaptive responses - Door Control Maintenance Guidelines",
    version="3.0.0"
)

# Allow CORS for local testing and development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str
    llm_model: str = None  # Optional, defaults to backend default
    user_level: str = "beginner"  # "beginner" or "expert"

class AskResponse(BaseModel):
    answer: str
    user_level: str
    answer_length: int
    auto_detected: bool

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Intelligent Q&A interface with user-level adaptive responses
    
    - llm_model: "gemma3" (default)
    - user_level: "beginner" or "expert"
    """
    answer = answer_question(request.question, request.llm_model, request.user_level)
    
    return AskResponse(
        answer=answer,
        user_level=request.user_level,
        answer_length=len(answer),
        auto_detected=False
    )

@app.get("/")
def root():
    user_levels = get_user_level_info()
    return {
        "message": "Intelligent Railway RAG System - User Level Adaptive",
        "version": "3.0.0",
        "features": {
            "user_levels": "Supports beginner and expert levels",
            "llm_models": ["gemma3", "mistral"]
        },
        "user_levels": user_levels,
        "endpoints": {
            "/ask": "POST - Intelligent Q&A (supports user levels)",
            "/user-levels": "GET - Get user level information",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/user-levels")
def get_levels():
    """Get user level information for frontend button display"""
    return get_user_level_info()

@app.get("/ui")
def get_ui():
    """Redirect to Gradio interface"""
    return {"message": "Please use the Gradio interface at http://localhost:7860"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "message": "Intelligent Railway RAG System running normally",
        "capabilities": [
            "User level adaptation",
            "Adaptive response generation",
            "Door control maintenance support",
            "Multiple LLM models (gemma3, mistral)"
        ]
    } 