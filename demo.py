#!/usr/bin/env python3
"""
Demo script for Railway RAG Project
This script demonstrates the capabilities of the RAG system.
"""

import time
from backend import answer_question

def demo_questions():
    """Demo with various questions about railway documentation"""
    
    questions = [
        "What are the maintenance steps for door control units?",
        "What are the 4 steps of the TAR?",
        "What are the safety precautions for door control maintenance?",
        "How to troubleshoot door control unit problems?",
        "What is the scope of this assessment?"
    ]
    
    models = ["gemma3", "deepseek-r1"]
    
    print("Railway RAG System Demo")
    print("=" * 50)
    print()
    
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 40)
        
        for model in models:
            print(f"\nUsing {model}:")
            try:
                start_time = time.time()
                # Determine document type based on question content
                if any(keyword in question.lower() for keyword in ['door', 'control', 'maintenance', 'safety', 'troubleshoot']):
                    answer = answer_question(question, llm_model=model, document_type="door_control")
                else:
                    answer = answer_question(question, llm_model=model, document_type="railway")
                end_time = time.time()
                
                # Truncate long answers for display
                display_answer = answer[:200] + "..." if len(answer) > 200 else answer
                print(f"   {display_answer}")
                print(f"   Response time: {end_time - start_time:.2f}s")
                
            except Exception as e:
                print(f"   Error with {model}: {e}")
        
        print("\n" + "=" * 50)
        print()

def interactive_demo():
    """Interactive demo where user can ask questions"""
    
    print("Railway RAG System - Interactive Demo")
    print("=" * 50)
    print("Ask questions about the railway documentation.")
    print("Type 'quit' to exit, 'models' to see available models.")
    print()
    
    models = ["gemma3", "deepseek-r1"]
    
    while True:
        try:
            question = input("Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif question.lower() == 'models':
                print("Available models:")
                for model in models:
                    print(f"   • {model}")
                print()
                continue
            elif not question:
                continue
            
            # Ask which model to use
            print("Choose a model:")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model}")
            print("   Enter number or press Enter for default (gemma3):")
            
            model_choice = input("   Choice: ").strip()
            
            if model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
                selected_model = models[int(model_choice) - 1]
            else:
                selected_model = "gemma3"
            
            print(f"\nSearching with {selected_model}...")
            start_time = time.time()
            
            # Determine document type based on question content
            if any(keyword in question.lower() for keyword in ['door', 'control', 'maintenance', 'safety', 'troubleshoot']):
                answer = answer_question(question, llm_model=selected_model, document_type="door_control")
            else:
                answer = answer_question(question, llm_model=selected_model, document_type="railway")
            
            end_time = time.time()
            print(f"\nAnswer ({end_time - start_time:.2f}s):")
            print(f"   {answer}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print()

def main():
    """Main demo function"""
    print("Choose demo mode:")
    print("1. Predefined questions demo")
    print("2. Interactive demo")
    print("3. Exit")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        demo_questions()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        print("Goodbye!")
    else:
        print("Invalid choice. Running predefined demo...")
        demo_questions()

if __name__ == "__main__":
    main()
