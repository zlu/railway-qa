#!/usr/bin/env python3
"""
Test script to compare original vs improved RAG implementation
"""

from backend import answer_question as original_answer
from backend_improved import answer_question_improved as improved_answer, test_retrieval_quality
import time

def compare_responses(question, user_level="beginner"):
    """Compare original vs improved responses"""
    print(f"\n{'='*80}")
    print(f"Comparing responses for: {question}")
    print(f"User Level: {user_level}")
    print(f"{'='*80}")
    
    # Test retrieval quality first
    print("\n--- Document Retrieval Analysis ---")
    test_retrieval_quality(question, "door_control")
    
    # Test original implementation
    print(f"\n--- Original Implementation ---")
    start_time = time.time()
    original_response = original_answer(question, user_level=user_level, document_type="door_control")
    original_time = time.time() - start_time
    
    print(f"Response Length: {len(original_response)} characters")
    print(f"Generation Time: {original_time:.2f} seconds")
    print("Response Preview:")
    print(original_response[:500] + "...")
    
    # Test improved implementation
    print(f"\n--- Improved Implementation ---")
    start_time = time.time()
    improved_response = improved_answer(question, user_level=user_level, document_type="door_control")
    improved_time = time.time() - start_time
    
    print(f"Response Length: {len(improved_response)} characters")
    print(f"Generation Time: {improved_time:.2f} seconds")
    print("Response Preview:")
    print(improved_response[:500] + "...")
    
    # Compare key phrases
    print(f"\n--- Key Phrase Analysis ---")
    
    # Expected phrases for this question
    expected_phrases = [
        "disconnect electrical loads", "measure current consumption", "water damage",
        "water-resistant DCU", "input voltage", "mains supply", "replace DCU"
    ]
    
    original_found = sum(1 for phrase in expected_phrases if phrase.lower() in original_response.lower())
    improved_found = sum(1 for phrase in expected_phrases if phrase.lower() in improved_response.lower())
    
    print(f"Original - Phrases found: {original_found}/{len(expected_phrases)} ({original_found/len(expected_phrases):.1%})")
    print(f"Improved - Phrases found: {improved_found}/{len(expected_phrases)} ({improved_found/len(expected_phrases):.1%})")
    
    return {
        "original_length": len(original_response),
        "improved_length": len(improved_response),
        "original_time": original_time,
        "improved_time": improved_time,
        "original_phrases": original_found,
        "improved_phrases": improved_found,
        "total_phrases": len(expected_phrases)
    }

def test_expert_vs_beginner():
    """Test the difference between expert and beginner responses"""
    question = "What should be done if the DCU display is dark or shows no power?"
    
    print(f"\n{'='*80}")
    print("Testing Expert vs Beginner Response Differences")
    print(f"{'='*80}")
    
    # Test beginner
    print(f"\n--- Beginner Response ---")
    start_time = time.time()
    beginner_response = improved_answer(question, user_level="beginner", document_type="door_control")
    beginner_time = time.time() - start_time
    
    print(f"Length: {len(beginner_response)} characters")
    print(f"Time: {beginner_time:.2f} seconds")
    print("Preview:")
    print(beginner_response[:300] + "...")
    
    # Test expert
    print(f"\n--- Expert Response ---")
    start_time = time.time()
    expert_response = improved_answer(question, user_level="expert", document_type="door_control")
    expert_time = time.time() - start_time
    
    print(f"Length: {len(expert_response)} characters")
    print(f"Time: {expert_time:.2f} seconds")
    print("Preview:")
    print(expert_response[:300] + "...")
    
    # Compare lengths
    length_ratio = len(expert_response) / len(beginner_response) if len(beginner_response) > 0 else 1.0
    print(f"\n--- Comparison ---")
    print(f"Expert/Beginner length ratio: {length_ratio:.2f}")
    print(f"Expert is {'shorter' if length_ratio < 0.8 else 'longer' if length_ratio > 1.2 else 'similar length'} than beginner")
    
    return {
        "beginner_length": len(beginner_response),
        "expert_length": len(expert_response),
        "length_ratio": length_ratio
    }

def main():
    """Run comprehensive comparison tests"""
    print("🚂 Railway RAG System - Improvement Comparison")
    print("=" * 80)
    
    # Test questions from level_tests.txt
    test_questions = [
        "What should be done if the DCU display is dark or shows no power?",
        "How should relays and solenoids be maintained?",
        "What are the maintenance steps for sensors?",
        "What are the maintenance steps for the emergency release mechanism?",
        "What are the maintenance steps for limit, proximity, and micro switches?"
    ]
    
    results = []
    
    for question in test_questions:
        result = compare_responses(question, "beginner")
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 IMPROVEMENT SUMMARY")
    print(f"{'='*80}")
    
    avg_original_phrases = sum(r["original_phrases"] for r in results) / len(results)
    avg_improved_phrases = sum(r["improved_phrases"] for r in results) / len(results)
    total_phrases = results[0]["total_phrases"]
    
    print(f"Average Original Phrase Accuracy: {avg_original_phrases}/{total_phrases} ({avg_original_phrases/total_phrases:.1%})")
    print(f"Average Improved Phrase Accuracy: {avg_improved_phrases}/{total_phrases} ({avg_improved_phrases/total_phrases:.1%})")
    print(f"Improvement: +{avg_improved_phrases - avg_original_phrases:.1f} phrases (+{(avg_improved_phrases - avg_original_phrases)/total_phrases:.1%})")
    
    # Test expert vs beginner
    expert_beginner_result = test_expert_vs_beginner()
    
    print(f"\n{'='*80}")
    print("🎯 KEY IMPROVEMENTS")
    print(f"{'='*80}")
    
    if avg_improved_phrases > avg_original_phrases:
        print("✅ Better content utilization from retrieved documents")
    else:
        print("⚠️ Content utilization needs further improvement")
    
    if expert_beginner_result["length_ratio"] < 0.8:
        print("✅ Expert responses are appropriately concise")
    else:
        print("⚠️ Expert responses could be more concise")
    
    print("✅ Explicit instructions to use document content")
    print("✅ Better prompt engineering for user levels")
    print("✅ Increased document retrieval (k=8 instead of k=6)")

if __name__ == "__main__":
    main()
