import gradio as gr
from backend import answer_question, get_user_level_info

def ask_question(question, user_level, llm_model):
    """Handle question submission with user level adaptation"""
    if not question.strip():
        return "Please enter a question."
    
    try:
        # Get response from backend
        response = answer_question(question, llm_model, user_level)
        
        # Get user level info for display
        user_levels = get_user_level_info()
        level_info = user_levels.get(user_level, {})
        level_name = level_info.get('name', user_level.title())
        
        # Format the response
        formatted_response = f"""
## Response ({level_name} Level)

{response}

---
**Response Details:**
- User Level: {level_name}
- Model: {llm_model}
- Response Length: {len(response)} characters
        """
        
        return formatted_response
        
    except Exception as e:
        return f"Error: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Railway RAG System - Door Control Maintenance", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚂 Railway RAG System - Door Control Maintenance
    
    **Intelligent Q&A system with user-level adaptive responses**
    
    Ask questions about door control maintenance and get responses tailored to your expertise level.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Enter your question about door control maintenance...",
                lines=3
            )
            
            with gr.Row():
                user_level = gr.Dropdown(
                    choices=["beginner", "expert"],
                    value="beginner",
                    label="User Level",
                    info="Select your expertise level for tailored responses"
                )
                
                llm_model = gr.Dropdown(
                    choices=["gemma3", "mistral"],
                    value="gemma3",
                    label="AI Model",
                    info="Select the language model to use"
                )
            
            submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # User level information
            gr.Markdown("""
            ### User Levels
            
            **Beginner Mode:**
            - Comprehensive explanations
            - Step-by-step guidance
            - Safety reminders
            - Simple language
            
            **Expert Mode:**
            - Technical summaries
            - Professional terminology
            - Advanced troubleshooting
            - System integration details
            """)
    
    # Output section
    output = gr.Markdown(
        label="Response",
        value="Enter a question above to get started..."
    )
    
    # Example questions
    gr.Markdown("""
    ### 💡 Example Questions
    
    **Beginner Examples:**
    - What are the basic maintenance steps for door control units?
    - How to safely perform door control maintenance?
    - What safety precautions should I take?
    
    **Expert Examples:**
    - What are the technical parameters and advanced troubleshooting methods?
    - How to optimize door control system performance?
    - What are the system integration requirements?
    """)
    
    # Event handlers
    submit_btn.click(
        fn=ask_question,
        inputs=[question_input, user_level, llm_model],
        outputs=output
    )
    
    # Allow Enter key to submit
    question_input.submit(
        fn=ask_question,
        inputs=[question_input, user_level, llm_model],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
