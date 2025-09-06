# Railway RAG Project

An intelligent Retrieval-Augmented Generation (RAG) system with user-level adaptive responses for door control maintenance documentation.

## Overview

This project implements a sophisticated RAG system that provides intelligent, context-aware answers to questions about door control maintenance guidelines. The system features autonomous user-level detection and adaptive response generation, tailoring answers to either beginner or expert users.

## Core Features

- **Intelligent RAG Pipeline**: Retrieves relevant information from door control maintenance documentation
- **User Level Adaptation**: Automatically detects user expertise level and adjusts response complexity
- **Autonomous Learning**: Analyzes question characteristics to determine appropriate response style
- **Multiple LLM Support**: Compatible with Ollama models (gemma3, mistral)
- **RESTful API**: FastAPI-based web service with comprehensive endpoints
- **Web Interface**: Simple HTML/JavaScript frontend for easy interaction

## User Level Functionality

### Beginner Mode
- **Target**: New users or those seeking comprehensive explanations
- **Response Style**: Detailed, step-by-step explanations with context
- **Characteristics**: Uses basic terminology, provides background information, includes safety warnings

### Expert Mode
- **Target**: Experienced technicians or professionals
- **Response Style**: Concise, technical summaries focusing on key points
- **Characteristics**: Uses technical terminology, assumes prior knowledge, focuses on actionable steps



## Project Structure

```
railway_rag_project/
├── backend.py              # Core RAG logic and user level adaptation
├── gradio_app.py           # Gradio web interface (recommended)
├── fastapi_app.py          # REST API endpoints
├── demo.py                 # Command-line demonstration
├── setup.py                # System setup verification
├── test_setup.py           # Basic functionality tests
├── quick_start.sh          # Quick setup script
├── requirements.txt        # Python dependencies
├── tests.txt               # Test questions and expected answers
├── level_tests.txt         # User level specific test cases
└── railway_chroma_db/      # Vector database storage
```

## Prerequisites

1. **Python 3.8+** with pip
2. **Ollama** installed and running locally
3. **Required Ollama models**:
   ```bash
   ollama pull gemma3
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

## Usage

### Web Interface Usage (which can be used to test the system instead of using Unity)

1. **Start the Gradio interface (recommended)**:
   ```bash
   python gradio_app.py
   ```
   
   Or start the API server:
   ```bash
   python fastapi_app.py
   ```

2. **Open the Gradio interface**:
   - Navigate to `http://localhost:7860` in your browser
   - Select user level (Beginner or Expert)
   - Enter your question about door control maintenance
   - Choose LLM model (gemma3 or mistral)
   - Click "Ask Question" to get an adaptive response

### API Usage

#### Example API Request
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the maintenance steps for door control units?",
    "user_level": "beginner",
    "llm_model": "gemma3"  # or "mistral"
  }'
```

#### Example API Response
```json
{
  "answer": "Based on the door control maintenance guidelines...",
  "user_level": "beginner",
  "answer_length": 450,
  "user_level_used": "beginner"
}
```

### Direct Python Usage

```python
from backend import answer_question

# Auto-detect user level
answer = answer_question("What are the safety precautions for door maintenance?")

# Specify user level
answer = answer_question(
    "How to troubleshoot door control problems?", 
    user_level="expert",
    llm_model="gemma3"
)
```

## Example Questions and Answers

### Door Control Maintenance
**Q**: "What are the maintenance steps for door control units?"
**A**: "The maintenance process involves: 1) Visual inspection of components, 2) Functional testing of door operations, 3) Lubrication of moving parts, 4) Electrical system verification..."

**Q**: "What safety precautions should be taken during maintenance?"
**A**: "Safety precautions include: wearing appropriate PPE, ensuring power isolation, following lockout/tagout procedures, maintaining clear work areas..."

**Q**: "How to troubleshoot door control unit problems?"
**A**: "Troubleshooting steps: 1) Check power supply and connections, 2) Verify sensor functionality, 3) Test control logic, 4) Inspect mechanical components..."

## Configuration

### LLM Models
- **Default**: `gemma3`
- **Available**: `gemma3`, `mistral`
- **Configuration**: Set in `backend.py` or pass via API

### Vector Database
- **Storage**: ChromaDB in `railway_chroma_db/`
- **Collection**: `door_control_embeddings`
- **Embedding Model**: `nomic-embed-text`

### User Level Configuration
```python
USER_LEVELS = {
    "beginner": {
        "name": "Beginner",
        "description": "New users seeking comprehensive explanations",
        "characteristics": ["detailed", "step-by-step", "safety-focused"]
    },
    "expert": {
        "name": "Expert", 
        "description": "Experienced technicians seeking concise technical info",
        "characteristics": ["concise", "technical", "action-oriented"]
    }
}
```

## API Endpoints

- `GET /` - API information and capabilities
- `POST /ask` - Main Q&A endpoint with user level support
- `GET /user-levels` - Get available user level definitions
- `GET /health` - System health check
- `GET /docs` - Interactive API documentation

## Testing

Run basic functionality tests:
```bash
python test_setup.py
```

Test user level adaptation:
```bash
python -c "
from backend import answer_question
print('Beginner:', answer_question('What is door maintenance?', user_level='beginner'))
print('Expert:', answer_question('What is door maintenance?', user_level='expert'))
"
```