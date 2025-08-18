from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from backend import answer_question, answer_question_combined, get_user_level_info

app = FastAPI(
    title="Railway RAG API",
    description="智能RAG系统，支持用户级别自适应回答 - Railway Documentation and Door Control Maintenance Guidelines",
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
    document_type: str = "railway"  # "railway", "door_control", or "combined"
    user_level: str = None  # "beginner", "expert", or None for auto-detection

class AskResponse(BaseModel):
    answer: str
    document_type: str
    user_level: str
    answer_length: int
    auto_detected: bool

@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    智能问答接口，支持用户级别自适应回答
    
    - document_type: "railway" (default), "door_control", or "combined"
    - llm_model: "gemma3" (default), "deepseek-r1"
    - user_level: "beginner", "expert", or None (自动检测)
    """
    auto_detected = request.user_level is None
    
    if request.document_type.lower() == "combined":
        answer = answer_question_combined(request.question, request.llm_model, request.user_level)
        doc_type = "combined"
    else:
        answer = answer_question(request.question, request.llm_model, request.document_type, request.user_level)
        doc_type = request.document_type
    
    # Determine the actual user level used (either provided or auto-detected)
    from backend import analyze_user_level_from_question
    actual_user_level = request.user_level or analyze_user_level_from_question(request.question)
    
    return AskResponse(
        answer=answer,
        document_type=doc_type,
        user_level=actual_user_level,
        answer_length=len(answer),
        auto_detected=auto_detected
    )

@app.get("/")
def root():
    user_levels = get_user_level_info()
    return {
        "message": "智能铁路RAG系统 - 支持用户级别自适应",
        "version": "3.0.0",
        "features": {
            "autonomous_learning": "自动检测用户级别并调整回答",
            "user_levels": "支持新手和老手两种级别",
            "document_types": ["railway", "door_control", "combined"],
            "llm_models": ["gemma3", "deepseek-r1"]
        },
        "user_levels": user_levels,
        "endpoints": {
            "/ask": "POST - 智能问答（支持用户级别）",
            "/user-levels": "GET - 获取用户级别信息",
            "/docs": "GET - 交互式API文档"
        }
    }

@app.get("/user-levels")
def get_levels():
    """获取用户级别信息，用于前端按钮显示"""
    return get_user_level_info()

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "message": "智能铁路RAG系统运行正常",
        "capabilities": [
            "用户级别自动检测",
            "自适应回答生成",
            "多文档类型支持",
            "自主学习能力"
        ]
    } 