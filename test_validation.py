#!/usr/bin/env python3
"""
Test Validation Script for Railway RAG System
Validates system responses against expected results from tests.txt
"""

from backend import answer_question, analyze_user_level_from_question
import re
import time

# Expected results from tests.txt
EXPECTED_RESULTS = {
    "What steps are included in the general repair framework for DCU maintenance?": {
        "key_phrases": [
            "fault identification", "fault code display", "root cause analysis", 
            "corrective actions", "step-by-step instructions", "repair", "replace"
        ],
        "expected_level": "expert"
    },
    "What are the typical tools needed for DCU maintenance?": {
        "key_phrases": [
            "spanners", "allen keys", "screwdriver", "circlip pliers", 
            "knives", "scissors", "hammers", "tools"
        ],
        "expected_level": "beginner"
    },
    "How should a train operator respond when all doors fail to close?": {
        "key_phrases": [
            "single door", "all doors", "porter's button", "circuit breakers",
            "door isolating cock", "TDIC", "TMS2", "driving cab"
        ],
        "expected_level": "expert"
    },
    "What are the common causes for slamming doors?": {
        "key_phrases": [
            "piston-rod length", "crank position", "speed control", "causes"
        ],
        "expected_level": "expert"
    },
    "What are the maintenance steps for the emergency release mechanism?": {
        "key_phrases": [
            "manual operation", "lubricate", "adjust", "replace", "emergency release"
        ],
        "expected_level": "expert"
    }
}

def check_english_response(response):
    """Check if response is in English"""
    # Simple check for common Chinese characters
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', response)
    return len(chinese_chars) == 0

def check_key_phrases(response, expected_phrases):
    """Check if response contains expected key phrases"""
    response_lower = response.lower()
    found_phrases = []
    missing_phrases = []
    
    for phrase in expected_phrases:
        if phrase.lower() in response_lower:
            found_phrases.append(phrase)
        else:
            missing_phrases.append(phrase)
    
    return found_phrases, missing_phrases

def test_question(question, expected_data):
    """Test a single question against expected results"""
    print(f"\n{'='*80}")
    print(f"Testing: {question}")
    print(f"{'='*80}")
    
    # Test auto-detection
    detected_level = analyze_user_level_from_question(question)
    expected_level = expected_data["expected_level"]
    
    print(f"Expected Level: {expected_level}")
    print(f"Detected Level: {detected_level}")
    print(f"Level Match: {'✅' if detected_level == expected_level else '❌'}")
    
    # Test beginner mode
    print(f"\n--- Beginner Mode Test ---")
    start_time = time.time()
    beginner_response = answer_question(question, user_level="beginner", document_type="door_control")
    beginner_time = time.time() - start_time
    
    # Test expert mode
    print(f"\n--- Expert Mode Test ---")
    start_time = time.time()
    expert_response = answer_question(question, user_level="expert", document_type="door_control")
    expert_time = time.time() - start_time
    
    # Test auto-detection mode
    print(f"\n--- Auto-Detection Mode Test ---")
    start_time = time.time()
    auto_response = answer_question(question, document_type="door_control")
    auto_time = time.time() - start_time
    
    # Validate responses
    print(f"\n--- Response Validation ---")
    
    # Check English responses
    beginner_english = check_english_response(beginner_response)
    expert_english = check_english_response(expert_response)
    auto_english = check_english_response(auto_response)
    
    print(f"Beginner Response (English): {'✅' if beginner_english else '❌'}")
    print(f"Expert Response (English): {'✅' if expert_english else '❌'}")
    print(f"Auto Response (English): {'✅' if auto_english else '❌'}")
    
    # Check key phrases for expert response (most comprehensive)
    found_phrases, missing_phrases = check_key_phrases(expert_response, expected_data["key_phrases"])
    
    print(f"\nKey Phrases Found ({len(found_phrases)}/{len(expected_data['key_phrases'])}):")
    for phrase in found_phrases:
        print(f"  ✅ {phrase}")
    
    if missing_phrases:
        print(f"\nMissing Key Phrases ({len(missing_phrases)}):")
        for phrase in missing_phrases:
            print(f"  ❌ {phrase}")
    
    # Response statistics
    print(f"\n--- Response Statistics ---")
    print(f"Beginner Response Length: {len(beginner_response)} characters")
    print(f"Expert Response Length: {len(expert_response)} characters")
    print(f"Auto Response Length: {len(auto_response)} characters")
    print(f"Beginner Generation Time: {beginner_time:.2f} seconds")
    print(f"Expert Generation Time: {expert_time:.2f} seconds")
    print(f"Auto Generation Time: {auto_time:.2f} seconds")
    
    # Calculate accuracy score
    phrase_accuracy = len(found_phrases) / len(expected_data["key_phrases"])
    level_accuracy = 1.0 if detected_level == expected_level else 0.0
    english_accuracy = (beginner_english + expert_english + auto_english) / 3.0
    
    overall_score = (phrase_accuracy + level_accuracy + english_accuracy) / 3.0
    
    print(f"\n--- Accuracy Scores ---")
    print(f"Key Phrase Accuracy: {phrase_accuracy:.2%}")
    print(f"Level Detection Accuracy: {level_accuracy:.2%}")
    print(f"English Response Accuracy: {english_accuracy:.2%}")
    print(f"Overall Score: {overall_score:.2%}")
    
    return {
        "level_match": detected_level == expected_level,
        "english_responses": [beginner_english, expert_english, auto_english],
        "phrase_accuracy": phrase_accuracy,
        "response_lengths": [len(beginner_response), len(expert_response), len(auto_response)],
        "generation_times": [beginner_time, expert_time, auto_time],
        "overall_score": overall_score
    }

def main():
    """Run comprehensive tests"""
    print("🚂 Railway RAG System - Comprehensive Test Validation")
    print("=" * 80)
    print("Testing system against expected results from tests.txt")
    print("=" * 80)
    
    results = {}
    total_score = 0
    
    for question, expected_data in EXPECTED_RESULTS.items():
        try:
            result = test_question(question, expected_data)
            results[question] = result
            total_score += result["overall_score"]
        except Exception as e:
            print(f"Error testing question: {e}")
            results[question] = {"error": str(e), "overall_score": 0}
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = len([r for r in results.values() if "error" not in r])
    total_tests = len(results)
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful Tests: {successful_tests}")
    print(f"Failed Tests: {total_tests - successful_tests}")
    
    if successful_tests > 0:
        avg_score = total_score / successful_tests
        print(f"Average Overall Score: {avg_score:.2%}")
        
        # Level detection accuracy
        level_matches = sum(1 for r in results.values() if "error" not in r and r["level_match"])
        level_accuracy = level_matches / successful_tests
        print(f"Level Detection Accuracy: {level_accuracy:.2%}")
        
        # English response accuracy
        english_scores = []
        for r in results.values():
            if "error" not in r:
                english_scores.extend(r["english_responses"])
        
        if english_scores:
            english_accuracy = sum(english_scores) / len(english_scores)
            print(f"English Response Accuracy: {english_accuracy:.2%}")
        
        # Phrase accuracy
        phrase_scores = [r["phrase_accuracy"] for r in results.values() if "error" not in r]
        avg_phrase_accuracy = sum(phrase_scores) / len(phrase_scores)
        print(f"Average Phrase Accuracy: {avg_phrase_accuracy:.2%}")
    
    print(f"\n{'='*80}")
    print("🎯 KEY FINDINGS")
    print(f"{'='*80}")
    
    if successful_tests > 0:
        if avg_score >= 0.8:
            print("✅ System performing excellently")
        elif avg_score >= 0.6:
            print("✅ System performing well")
        else:
            print("⚠️ System needs improvement")
        
        if level_accuracy >= 0.8:
            print("✅ User level detection working well")
        else:
            print("⚠️ User level detection needs improvement")
        
        if english_accuracy >= 0.9:
            print("✅ English response requirement met")
        else:
            print("❌ English response requirement not fully met")
        
        if avg_phrase_accuracy >= 0.7:
            print("✅ Content accuracy is good")
        else:
            print("⚠️ Content accuracy needs improvement")
    
    print(f"\n{'='*80}")
    print("🏁 Test validation complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
