#!/usr/bin/env python3
"""
Test script for automatic user level detection feature
"""

from backend import analyze_question_complexity

def test_question_analysis():
    """Test various questions to see how the auto-detection works"""
    
    test_questions = [
        # Beginner questions
        "What is a DCU?",
        "How do I check if the door is working?",
        "What should I do if the door won't close?",
        "Can you explain what a fault code means?",
        "I need help understanding the maintenance manual",
        
        # Expert questions
        "What are the technical specifications for DCU fault code diagnostics?",
        "How to calibrate proximity sensors for optimal performance?",
        "Troubleshooting TDIC circuit breaker overload issues",
        "DCU firmware update procedures and system integration requirements",
        "Advanced maintenance procedures for actuator optimization",
        
        # Mixed complexity questions
        "What happens if the DCU display shows no power?",
        "How to test the emergency release mechanism?",
        "What tools are needed for DCU maintenance?"
    ]
    
    print("Testing Automatic User Level Detection")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = analyze_question_complexity(question)
        
        print(f"   Detected Level: {result['suggested_level'].upper()}")
        print(f"   Confidence: {result['confidence']}")
        print(f"   Complexity Score: {result['complexity_score']:.1f}")
        
        analysis = result['analysis']
        print(f"   Technical Terms: {analysis['expert_terms_found']}")
        print(f"   Expert Indicators: {analysis['expert_indicators']}")
        print(f"   Beginner Indicators: {analysis['beginner_indicators']}")
        print("-" * 30)

if __name__ == "__main__":
    test_question_analysis()
