#!/usr/bin/env python3
"""
Level Validation Test Script for Railway RAG System
Validates beginner and expert responses against expected results from level_tests.txt
"""

from backend import answer_question
import re
import time

# Expected results from level_tests.txt
LEVEL_TEST_RESULTS = {
    "What should be done if the DCU display is dark or shows no power?": {
        "expert_key_phrases": [
            "disconnect electrical loads", "measure current consumption", "replace DCU"
        ],
        "beginner_key_phrases": [
            "disconnect electrical loads", "measure current consumption", "water damage", 
            "water-resistant DCU", "input voltage", "mains supply", "replace DCU"
        ]
    },
    "How should relays and solenoids be maintained?": {
        "expert_key_phrases": [
            "replace faulty relays", "replace solenoids"
        ],
        "beginner_key_phrases": [
            "test operation", "relays", "solenoids", "replace faulty"
        ]
    },
    "What are the maintenance steps for sensors?": {
        "expert_key_phrases": [
            "test sensor detection", "signal transmission"
        ],
        "beginner_key_phrases": [
            "test sensor detection", "signal transmission", "calibrate sensor positions", 
            "replace faulty sensors"
        ]
    },
    "What are the maintenance steps for the emergency release mechanism?": {
        "expert_key_phrases": [
            "test manual operation", "emergency release mechanism"
        ],
        "beginner_key_phrases": [
            "test manual operation", "emergency release mechanism", "lubricate components", 
            "adjust release device", "replace mechanism"
        ]
    },
    "What are the maintenance steps for limit, proximity, and micro switches?": {
        "expert_key_phrases": [
            "test switch triggering", "signal output"
        ],
        "beginner_key_phrases": [
            "test switch triggering", "signal output", "adjust switch positions", 
            "replace damaged switches"
        ]
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

def test_question_levels(question, expected_data):
    """Test a single question for both beginner and expert levels"""
    print(f"\n{'='*80}")
    print(f"Testing: {question}")
    print(f"{'='*80}")
    
    # Test beginner level
    print(f"\n--- Beginner Level Test ---")
    start_time = time.time()
    beginner_response = answer_question(question, user_level="beginner", document_type="door_control")
    beginner_time = time.time() - start_time
    
    # Test expert level
    print(f"\n--- Expert Level Test ---")
    start_time = time.time()
    expert_response = answer_question(question, user_level="expert", document_type="door_control")
    expert_time = time.time() - start_time
    
    # Validate responses
    print(f"\n--- Response Validation ---")
    
    # Check English responses
    beginner_english = check_english_response(beginner_response)
    expert_english = check_english_response(expert_response)
    
    print(f"Beginner Response (English): {'✅' if beginner_english else '❌'}")
    print(f"Expert Response (English): {'✅' if expert_english else '❌'}")
    
    # Check key phrases for both levels
    beginner_found, beginner_missing = check_key_phrases(beginner_response, expected_data["beginner_key_phrases"])
    expert_found, expert_missing = check_key_phrases(expert_response, expected_data["expert_key_phrases"])
    
    print(f"\nBeginner Key Phrases Found ({len(beginner_found)}/{len(expected_data['beginner_key_phrases'])}):")
    for phrase in beginner_found:
        print(f"  ✅ {phrase}")
    
    if beginner_missing:
        print(f"\nBeginner Missing Key Phrases ({len(beginner_missing)}):")
        for phrase in beginner_missing:
            print(f"  ❌ {phrase}")
    
    print(f"\nExpert Key Phrases Found ({len(expert_found)}/{len(expected_data['expert_key_phrases'])}):")
    for phrase in expert_found:
        print(f"  ✅ {phrase}")
    
    if expert_missing:
        print(f"\nExpert Missing Key Phrases ({len(expert_missing)}):")
        for phrase in expert_missing:
            print(f"  ❌ {phrase}")
    
    # Response statistics
    print(f"\n--- Response Statistics ---")
    print(f"Beginner Response Length: {len(beginner_response)} characters")
    print(f"Expert Response Length: {len(expert_response)} characters")
    print(f"Beginner Generation Time: {beginner_time:.2f} seconds")
    print(f"Expert Generation Time: {expert_time:.2f} seconds")
    
    # Calculate accuracy scores
    beginner_phrase_accuracy = len(beginner_found) / len(expected_data["beginner_key_phrases"])
    expert_phrase_accuracy = len(expert_found) / len(expected_data["expert_key_phrases"])
    english_accuracy = (beginner_english + expert_english) / 2.0
    
    overall_score = (beginner_phrase_accuracy + expert_phrase_accuracy + english_accuracy) / 3.0
    
    print(f"\n--- Accuracy Scores ---")
    print(f"Beginner Phrase Accuracy: {beginner_phrase_accuracy:.2%}")
    print(f"Expert Phrase Accuracy: {expert_phrase_accuracy:.2%}")
    print(f"English Response Accuracy: {english_accuracy:.2%}")
    print(f"Overall Score: {overall_score:.2%}")
    
    # Check if expert response is more concise than beginner
    length_ratio = len(expert_response) / len(beginner_response) if len(beginner_response) > 0 else 1.0
    conciseness_score = 1.0 if length_ratio < 0.8 else (0.5 if length_ratio < 1.0 else 0.0)
    
    print(f"Expert Conciseness (shorter than beginner): {'✅' if length_ratio < 0.8 else '⚠️' if length_ratio < 1.0 else '❌'}")
    print(f"Length Ratio (Expert/Beginner): {length_ratio:.2f}")
    
    return {
        "beginner_english": beginner_english,
        "expert_english": expert_english,
        "beginner_phrase_accuracy": beginner_phrase_accuracy,
        "expert_phrase_accuracy": expert_phrase_accuracy,
        "english_accuracy": english_accuracy,
        "response_lengths": [len(beginner_response), len(expert_response)],
        "generation_times": [beginner_time, expert_time],
        "expert_conciseness": conciseness_score,
        "overall_score": overall_score
    }

def main():
    """Run comprehensive level-based tests"""
    print("🚂 Railway RAG System - Level-Based Test Validation")
    print("=" * 80)
    print("Testing beginner and expert responses against level_tests.txt")
    print("=" * 80)
    
    results = {}
    total_score = 0
    
    for question, expected_data in LEVEL_TEST_RESULTS.items():
        try:
            result = test_question_levels(question, expected_data)
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
        
        # English response accuracy
        english_scores = []
        for r in results.values():
            if "error" not in r:
                english_scores.extend([r["beginner_english"], r["expert_english"]])
        
        if english_scores:
            english_accuracy = sum(english_scores) / len(english_scores)
            print(f"English Response Accuracy: {english_accuracy:.2%}")
        
        # Phrase accuracy by level
        beginner_phrase_scores = [r["beginner_phrase_accuracy"] for r in results.values() if "error" not in r]
        expert_phrase_scores = [r["expert_phrase_accuracy"] for r in results.values() if "error" not in r]
        
        avg_beginner_accuracy = sum(beginner_phrase_scores) / len(beginner_phrase_scores)
        avg_expert_accuracy = sum(expert_phrase_scores) / len(expert_phrase_scores)
        
        print(f"Average Beginner Phrase Accuracy: {avg_beginner_accuracy:.2%}")
        print(f"Average Expert Phrase Accuracy: {avg_expert_accuracy:.2%}")
        
        # Conciseness analysis
        conciseness_scores = [r["expert_conciseness"] for r in results.values() if "error" not in r]
        avg_conciseness = sum(conciseness_scores) / len(conciseness_scores)
        print(f"Expert Conciseness Score: {avg_conciseness:.2%}")
        
        # Response length analysis
        all_lengths = []
        for r in results.values():
            if "error" not in r:
                all_lengths.extend(r["response_lengths"])
        
        if all_lengths:
            avg_length = sum(all_lengths) / len(all_lengths)
            print(f"Average Response Length: {avg_length:.0f} characters")
    
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
        
        if english_accuracy >= 0.9:
            print("✅ English response requirement fully met")
        else:
            print("❌ English response requirement not fully met")
        
        if avg_beginner_accuracy >= 0.7:
            print("✅ Beginner level responses are accurate")
        else:
            print("⚠️ Beginner level responses need improvement")
        
        if avg_expert_accuracy >= 0.7:
            print("✅ Expert level responses are accurate")
        else:
            print("⚠️ Expert level responses need improvement")
        
        if avg_conciseness >= 0.7:
            print("✅ Expert responses are appropriately concise")
        else:
            print("⚠️ Expert responses could be more concise")
    
    print(f"\n{'='*80}")
    print("🏁 Level-based test validation complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
