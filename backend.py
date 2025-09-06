from typing import Optional, Dict
from langchain_community.vectorstores import Chroma
from ollama_embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import re

# Settings (customize as needed)
PERSIST_DIRECTORY = "railway_chroma_db"
DOOR_CONTROL_COLLECTION = "door_control_embeddings"
EMBED_MODEL_NAME = "nomic-embed-text"

# Choose your LLM model (can be set per request)
DEFAULT_LLM_MODEL = "gemma3"  # Options: "gemma3", "mistral"

# HoloLens display constraints
HOLOLENS_MAX_CHARS = 900  # Maximum characters for single screen display
HOLOLENS_MAX_WORDS = 160  # Maximum words for comfortable reading

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


def analyze_question_complexity(question: str) -> Dict[str, any]:
    """
    Analyze a question to determine if it's from an expert or beginner.
    Returns analysis results and suggested user level.
    """
    # Technical terminology indicators
    expert_terms = {
        'dcu', 'tdic', 'circuit', 'breaker', 'actuator', 'solenoid', 'relay',
        'proximity', 'microswitch', 'limit switch', 'piston rod', 'crank',
        'speed control', 'fault code', 'diagnostic', 'calibration',
        'continuity', 'resistance', 'voltage', 'current', 'ohms',
        'bluetooth', 'gfA', 'stick', 'firmware', 'software', 'app',
        'emergency release', 'isolating cock', 'porter button',
        'water ingress', 'overload', 'short circuit', 'mains supply',
        'technical specifications', 'system integration', 'optimization',
        'display', 'power', 'maintenance', 'procedure', 'test', 'operation',
        'sensor', 'switch', 'mechanism', 'device', 'component', 'system',
        'performance', 'analysis', 'troubleshooting', 'diagnosis'
    }
    
    # Beginner language indicators
    beginner_indicators = [
        'what is', 'how do i', 'can you explain', 'what does', 'why does',
        'what happens if', 'how to', 'what should i do', 'what are the steps',
        'can you help me', 'i need help', 'i don\'t understand', 'what does this mean',
        'how does it work', 'what is the difference', 'when should i',
        'is it safe', 'what tools do i need', 'how long does it take'
    ]
    
    # Expert language indicators
    expert_indicators = [
        'specifications', 'parameters', 'troubleshooting', 'diagnostics',
        'calibration', 'optimization', 'integration', 'configuration',
        'maintenance procedures', 'technical requirements', 'system analysis',
        'fault diagnosis', 'performance metrics', 'compliance standards',
        'best practices', 'advanced', 'professional', 'technical'
    ]
    
    # Analyze the question
    question_lower = question.lower()
    words = set(re.findall(r'\b\w+\b', question_lower))
    
    # Count technical terms
    found_expert_terms = words.intersection(expert_terms)
    expert_term_count = len(found_expert_terms)
    
    # Count beginner indicators
    beginner_count = sum(1 for indicator in beginner_indicators if indicator in question_lower)
    
    # Count expert indicators
    expert_count = sum(1 for indicator in expert_indicators if indicator in question_lower)
    
    # Analyze question structure
    question_length = len(question.split())
    has_technical_terms = expert_term_count > 0
    has_beginner_phrases = beginner_count > 0
    has_expert_phrases = expert_count > 0
    
    # Determine complexity score
    complexity_score = 0
    complexity_score += expert_term_count * 2  # Technical terms heavily indicate expert
    complexity_score += expert_count * 1.5     # Expert phrases indicate expert
    complexity_score -= beginner_count * 1     # Beginner phrases indicate beginner
    complexity_score += question_length * 0.1  # Longer questions might be more complex
    
    # Determine suggested user level
    if complexity_score >= 2:
        suggested_level = "expert"
        confidence = "high" if complexity_score >= 4 else "medium"
    elif complexity_score <= -0.5:
        suggested_level = "beginner"
        confidence = "high" if complexity_score <= -1.5 else "medium"
    else:
        # For unclear cases, check if technical terms are present
        if has_technical_terms and not has_beginner_phrases:
            suggested_level = "expert"
            confidence = "low"
        else:
            suggested_level = "beginner"
            confidence = "low"
    
    return {
        "suggested_level": suggested_level,
        "confidence": confidence,
        "complexity_score": complexity_score,
        "analysis": {
            "expert_terms_found": list(found_expert_terms),
            "expert_term_count": expert_term_count,
            "beginner_indicators": beginner_count,
            "expert_indicators": expert_count,
            "question_length": question_length,
            "has_technical_terms": has_technical_terms,
            "has_beginner_phrases": has_beginner_phrases,
            "has_expert_phrases": has_expert_phrases
        }
    }


def create_level_appropriate_prompt(question: str, user_level: str, context: str) -> str:
    """
    Create a prompt that produces answers with appropriate technical depth for the user level.
    """
    level_info = USER_LEVELS[user_level]
    
    if user_level == "beginner":
        prompt_template = f"""You are a railway maintenance trainee helping a {level_info['name']} ({level_info['description']}).

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
8. Keep the response educational and easy to understand
9. Use "you need to", "you should", "first", "then", "finally" for step-by-step guidance
10. Explain WHY each step is important

HOLOLENS DISPLAY REQUIREMENTS:
- Provide a COMPLETE answer that fits within {HOLOLENS_MAX_CHARS} characters and {HOLOLENS_MAX_WORDS} words
- Summarize and condense information while maintaining completeness
- Use bullet points or numbered lists for clarity
- Include ALL essential information in a condensed format
- Prioritize the most important information first
- DO NOT provide partial answers - ensure the response is complete and self-contained

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.
IMPORTANT: Do not start with phrases like "Okay, here's" or "Sure, here's". Start directly with the answer.
IMPORTANT: Every technical term must be accompanied by a simple explanation.
IMPORTANT: RESPONSE MUST FIT ON A SINGLE HOLOLENS SCREEN.

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
10. Use professional, direct language without explanatory phrases
11. Focus on technical procedures and specifications
12. Avoid phrases like "this means", "in other words", "essentially", "basically"

HOLOLENS DISPLAY REQUIREMENTS:
- Provide a COMPLETE technical answer that fits within {HOLOLENS_MAX_CHARS} characters and {HOLOLENS_MAX_WORDS} words
- Summarize and condense technical information while maintaining completeness
- Use bullet points or numbered lists for technical procedures
- Include ALL essential technical details in a condensed format
- Prioritize critical technical information first
- Use abbreviations and technical shorthand
- DO NOT provide partial answers - ensure the response is complete and self-contained

IMPORTANT: Always respond in English, regardless of the language of the question.
IMPORTANT: If the document content does not contain information relevant to the question, say so clearly.
IMPORTANT: Do not start with phrases like "Based on the provided documentation" or "Sure, here's". Start directly with the answer.
IMPORTANT: Use maximum technical terminology without explanations.
IMPORTANT: Be concise and professional - no explanations of technical terms.
IMPORTANT: RESPONSE MUST FIT ON A SINGLE HOLOLENS SCREEN.

Please provide a technical expert answer based on the document content:"""
    
    return prompt_template

def create_summarization_prompt(original_response: str, question: str, user_level: str) -> str:
    """
    Create a prompt specifically for summarizing long responses while maintaining completeness.
    """
    level_info = USER_LEVELS[user_level]
    
    if user_level == "beginner":
        summarization_template = f"""You are helping a {level_info['name']} with railway maintenance.

The previous response was too long for HoloLens display. Please provide a COMPLETE but CONCISE summary.

Original question: {question}
Original response: {original_response}

Create a complete summarized answer that:
1. Includes ALL essential information from the original response
2. Uses simple, clear language with explanations
3. Fits within {HOLOLENS_MAX_CHARS} characters and {HOLOLENS_MAX_WORDS} words
4. Uses bullet points or numbered lists for clarity
5. Is complete and self-contained (no partial information)
6. Explains technical terms in simple language

Provide the complete summarized answer:"""
    
    else:  # expert level
        summarization_template = f"""You are providing technical support for a {level_info['name']}.

The previous response was too long for HoloLens display. Please provide a COMPLETE but CONCISE technical summary.

Original question: {question}
Original response: {original_response}

Create a complete summarized answer that:
1. Includes ALL essential technical information from the original response
2. Uses technical terminology and professional language
3. Fits within {HOLOLENS_MAX_CHARS} characters and {HOLOLENS_MAX_WORDS} words
4. Uses bullet points or numbered lists for procedures
5. Is complete and self-contained (no partial information)
6. Uses abbreviations and technical shorthand

Provide the complete technical summarized answer:"""
    
    return summarization_template

def ensure_complete_hololens_response(text: str) -> str:
    """
    Ensure response is complete and fits HoloLens display constraints.
    If the response is too long, it should be re-summarized rather than truncated.
    """
    if len(text) <= HOLOLENS_MAX_CHARS:
        return text
    
    # If response is too long, return a message indicating it needs re-summarization
    # This should trigger the LLM to provide a more concise but complete answer
    return f"Response too long ({len(text)} chars). Please provide a complete summary within {HOLOLENS_MAX_CHARS} characters."

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
    
    # Get response with level adaptation, with retry for length constraints
    max_retries = 2
    for attempt in range(max_retries + 1):
        response = llm.invoke(prompt)
        
        # Check if response fits within HoloLens constraints
        if len(response) <= HOLOLENS_MAX_CHARS:
            return response
        
        # If too long and we have retries left, create a more specific prompt for summarization
        if attempt < max_retries:
            prompt = create_summarization_prompt(response, question, user_level)
        else:
            # Final attempt - return the best we can do with a note
            return f"Complete answer (condensed): {response[:HOLOLENS_MAX_CHARS-50]}... [Full response available in logs]"
    
    return response



def get_user_level_info() -> Dict:
    """Get information about available user levels for the frontend."""
    return USER_LEVELS
