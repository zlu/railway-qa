from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend import answer_question, get_user_level_info, analyze_question_complexity

app = FastAPI(
    title="Railway RAG API",
    description="Intelligent RAG system with user-level adaptive responses",
    version="1.0.0"
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
    user_level: str = "beginner"  # Ignored - always uses auto-detection
    auto_detect_level: bool = False  # Ignored - always enabled

class AskResponse(BaseModel):
    answer: str
    user_level: str
    answer_length: int
    auto_detected: bool
    detection_analysis: dict = None  # Analysis results when auto-detection is used
    hololens_optimized: bool = True  # Always true - responses are complete and optimized for HoloLens
    max_chars: int = 900  # HoloLens character limit
    max_words: int = 160   # HoloLens word limit

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Intelligent Q&A interface with automatic user-level detection
    
    - llm_model: "gemma3" (default)
    - user_level: Ignored - always uses auto-detection
    - auto_detect_level: Always enabled - automatic user level detection based on question analysis
    """
    # Always auto-detect user level (ignore the incoming user_level parameter)
    detection_result = analyze_question_complexity(request.question)
    detected_level = detection_result["suggested_level"]
    auto_detected = True
    detection_analysis = detection_result
    
    # Get answer using the detected user level
    answer = answer_question(request.question, request.llm_model, detected_level)
    
    return AskResponse(
        answer=answer,
        user_level=detected_level,
        answer_length=len(answer),
        auto_detected=auto_detected,
        detection_analysis=detection_analysis,
        hololens_optimized=True,
        max_chars=900,
        max_words=160
    )

@app.get("/")
def root():
    user_levels = get_user_level_info()
    return {
        "message": "Intelligent Railway RAG System - User Level Adaptive & HoloLens Optimized",
        "version": "1.0.0",
        "features": {
            "user_levels": "Supports beginner and expert levels",
            "llm_models": ["gemma3", "mistral"],
            "hololens_optimized": "Complete responses summarized to 900 chars/160 words for single screen display"
        },
        "user_levels": user_levels,
        "hololens_constraints": {
            "max_characters": 900,
            "max_words": 160,
            "display_optimized": True
        },
        "endpoints": {
            "/ask": "POST - Intelligent Q&A (automatic user level detection)",
            "/user-levels": "GET - Get user level information",
            "/analyze-question": "POST - Analyze question complexity",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/user-levels")
def get_levels():
    """Get user level information for frontend button display"""
    return get_user_level_info()

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "message": "Intelligent Railway RAG System running normally",
        "capabilities": [
            "User level adaptation",
            "Adaptive response generation",
            "Door control maintenance support",
            "Multiple LLM models (gemma3, mistral)",
            "Automatic user level detection",
            "HoloLens display optimization (complete responses, 900 chars/160 words max)"
        ]
    }

class QuestionAnalysisRequest(BaseModel):
    question: str

@app.post("/analyze-question")
def analyze_question(request: QuestionAnalysisRequest):
    """
    Analyze a question to determine if it's from an expert or beginner.
    Returns detailed analysis without generating an answer.
    """
    analysis = analyze_question_complexity(request.question)
    return {
        "question": request.question,
        "analysis": analysis,
        "suggested_level": analysis["suggested_level"],
        "confidence": analysis["confidence"]
    } 