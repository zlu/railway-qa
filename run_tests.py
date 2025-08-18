#!/usr/bin/env python3
"""
Comprehensive test script for Railway RAG System
Tests both gemma3 and mistral models against tests.txt and level_tests.txt
"""

import time
import sys
from datetime import datetime
from backend import answer_question

def load_test_file(filename):
    """Load test questions and expected answers from file"""
    tests = []
    current_test = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith(('1.', '2.', '3.', '4.', '5.')):
            # New test question
            if current_test:
                tests.append(current_test)
            
            current_test = {
                'question': line.split('.', 1)[1].strip(),
                'expected_answer': ''
            }
            
            # Look for answer
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                if lines[i].strip().startswith('Answer'):
                    # Skip the "Answer:" line
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        current_test['expected_answer'] += lines[i].strip() + ' '
                        i += 1
                    break
                i += 1
        else:
            i += 1
    
    if current_test:
        tests.append(current_test)
    
    return tests

def load_level_test_file(filename):
    """Load level-specific test questions and expected answers"""
    tests = []
    current_test = {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith(('1.', '2.', '3.', '4.', '5.')):
            # New test question
            if current_test:
                tests.append(current_test)
            
            current_test = {
                'question': line.split('.', 1)[1].strip(),
                'expert_answer': '',
                'novice_answer': ''
            }
            
            # Look for expert and novice answers
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                line = lines[i].strip()
                if line.startswith('Expert Answer'):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith(('Novice Answer', '1.', '2.', '3.', '4.', '5.')):
                        current_test['expert_answer'] += lines[i].strip() + ' '
                        i += 1
                elif line.startswith('Novice Answer'):
                    i += 1
                    while i < len(lines) and not lines[i].strip().startswith(('1.', '2.', '3.', '4.', '5.')):
                        current_test['novice_answer'] += lines[i].strip() + ' '
                        i += 1
                else:
                    i += 1
        else:
            i += 1
    
    if current_test:
        tests.append(current_test)
    
    return tests

def simple_similarity(text1, text2):
    """Improved text similarity based on key technical terms and concepts"""
    import re
    
    # Clean and normalize text
    def clean_text(text):
        # Remove common filler words and phrases
        text = re.sub(r'\b(okay|here\'s|let\'s|sure|based|solely|provided|document|content|information)\b', '', text.lower())
        # Remove punctuation but keep important separators
        text = re.sub(r'[^\w\s\-]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Get key technical terms (words that appear in both texts)
    clean1 = clean_text(text1)
    clean2 = clean_text(text2)
    
    words1 = set(clean1.split())
    words2 = set(clean2.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate multiple similarity metrics
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Overlap similarity (how much of the shorter text is covered)
    min_len = min(len(words1), len(words2))
    overlap = len(intersection) / min_len if min_len > 0 else 0.0
    
    # Key terms bonus (technical terms that match)
    technical_terms = {
        'dcu', 'door', 'control', 'maintenance', 'repair', 'fault', 'identification',
        'tools', 'spanners', 'allen', 'keys', 'screwdriver', 'pliers', 'knives',
        'scissors', 'hammers', 'relays', 'solenoids', 'sensors', 'emergency',
        'release', 'mechanism', 'switches', 'limit', 'proximity', 'micro',
        'test', 'operation', 'replace', 'faulty', 'components', 'lubricate',
        'adjust', 'position', 'calibrate', 'damaged', 'electrical', 'loads',
        'current', 'consumption', 'voltage', 'power', 'display', 'dark'
    }
    
    key_matches = sum(1 for term in technical_terms if term in intersection)
    key_bonus = min(key_matches * 0.1, 0.3)  # Max 0.3 bonus for key terms
    
    # Combined similarity score
    combined_score = (jaccard * 0.4 + overlap * 0.4 + key_bonus)
    
    return min(combined_score, 1.0)  # Cap at 1.0

def evaluate_response(actual, expected, question_num):
    """Evaluate response against expected answer"""
    similarity = simple_similarity(actual, expected)
    
    print(f"Question {question_num}:")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual}")
    print(f"  Similarity: {similarity:.2f}")
    print(f"  Length: {len(actual)} chars")
    
    # Show key technical terms found in both
    import re
    def extract_technical_terms(text):
        technical_terms = {
            'dcu', 'door', 'control', 'maintenance', 'repair', 'fault', 'identification',
            'tools', 'spanners', 'allen', 'keys', 'screwdriver', 'pliers', 'knives',
            'scissors', 'hammers', 'relays', 'solenoids', 'sensors', 'emergency',
            'release', 'mechanism', 'switches', 'limit', 'proximity', 'micro',
            'test', 'operation', 'replace', 'faulty', 'components', 'lubricate',
            'adjust', 'position', 'calibrate', 'damaged', 'electrical', 'loads',
            'current', 'consumption', 'voltage', 'power', 'display', 'dark'
        }
        words = set(re.sub(r'[^\w\s]', ' ', text.lower()).split())
        return words.intersection(technical_terms)
    
    expected_terms = extract_technical_terms(expected)
    actual_terms = extract_technical_terms(actual)
    common_terms = expected_terms.intersection(actual_terms)
    
    if common_terms:
        print(f"  Key terms matched: {', '.join(sorted(common_terms))}")
    else:
        print(f"  No key technical terms matched")
    
    print()
    
    return similarity

def run_general_tests(model):
    """Run tests from tests.txt"""
    print(f"\n{'='*60}")
    print(f"GENERAL TESTS - {model.upper()}")
    print(f"{'='*60}")
    
    tests = load_test_file('tests.txt')
    total_similarity = 0
    
    for i, test in enumerate(tests, 1):
        print(f"Testing Question {i}: {test['question']}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            response = answer_question(test['question'], llm_model=model, user_level='beginner')
            end_time = time.time()
            
            similarity = evaluate_response(response, test['expected_answer'], i)
            total_similarity += similarity
            
            print(f"  Response time: {end_time - start_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    avg_similarity = total_similarity / len(tests) if tests else 0
    print(f"Average similarity: {avg_similarity:.2f}")
    return avg_similarity

def run_level_tests(model):
    """Run level-specific tests from level_tests.txt"""
    print(f"\n{'='*60}")
    print(f"LEVEL TESTS - {model.upper()}")
    print(f"{'='*60}")
    
    tests = load_level_test_file('level_tests.txt')
    expert_similarities = []
    novice_similarities = []
    
    for i, test in enumerate(tests, 1):
        print(f"Testing Question {i}: {test['question']}")
        print("-" * 50)
        
        # Test expert level
        try:
            print("Expert Level:")
            start_time = time.time()
            expert_response = answer_question(test['question'], llm_model=model, user_level='expert')
            end_time = time.time()
            
            expert_sim = evaluate_response(expert_response, test['expert_answer'], f"{i} (Expert)")
            expert_similarities.append(expert_sim)
            
            print(f"  Response time: {end_time - start_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
        
        # Test novice level
        try:
            print("Novice Level:")
            start_time = time.time()
            novice_response = answer_question(test['question'], llm_model=model, user_level='beginner')
            end_time = time.time()
            
            novice_sim = evaluate_response(novice_response, test['novice_answer'], f"{i} (Novice)")
            novice_similarities.append(novice_sim)
            
            print(f"  Response time: {end_time - start_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
    
    avg_expert = sum(expert_similarities) / len(expert_similarities) if expert_similarities else 0
    avg_novice = sum(novice_similarities) / len(novice_similarities) if novice_similarities else 0
    
    print(f"Average Expert similarity: {avg_expert:.2f}")
    print(f"Average Novice similarity: {avg_novice:.2f}")
    
    return avg_expert, avg_novice

def main():
    """Run comprehensive tests for both models"""
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"test_results_{timestamp}.log"
    
    # Redirect output to both console and file
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    # Redirect stdout to logger
    original_stdout = sys.stdout
    logger = Logger(log_filename)
    sys.stdout = logger
    
    try:
        print(f"Railway RAG System - Comprehensive Testing")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        models = ['gemma3', 'mistral']
        results = {}
        
        for model in models:
            print(f"\n{'='*60}")
            print(f"TESTING MODEL: {model.upper()}")
            print(f"{'='*60}")
            
            # Run general tests
            general_sim = run_general_tests(model)
            
            # Run level tests
            expert_sim, novice_sim = run_level_tests(model)
            
            results[model] = {
                'general': general_sim,
                'expert': expert_sim,
                'novice': novice_sim
            }
        
        # Summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*60}")
        
        for model, scores in results.items():
            print(f"\n{model.upper()}:")
            print(f"  General Tests: {scores['general']:.2f}")
            print(f"  Expert Level:  {scores['expert']:.2f}")
            print(f"  Novice Level:  {scores['novice']:.2f}")
            print(f"  Overall Avg:   {(scores['general'] + scores['expert'] + scores['novice']) / 3:.2f}")
        
        print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {log_filename}")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        logger.log.close()

if __name__ == "__main__":
    main()
