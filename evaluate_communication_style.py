#!/usr/bin/env python3
"""
Evaluation script for communication style differences between expert and novice responses
Focuses on technical jargon vs simple explanations rather than just similarity scores
"""

import time
import sys
from datetime import datetime
from backend import answer_question

def analyze_technical_terms(text):
    """Analyze the use of technical terms in the response"""
    technical_terms = {
        'dcu', 'door', 'control', 'maintenance', 'repair', 'fault', 'identification',
        'tools', 'spanners', 'allen', 'keys', 'screwdriver', 'pliers', 'knives',
        'scissors', 'hammers', 'relays', 'solenoids', 'sensors', 'emergency',
        'release', 'mechanism', 'switches', 'limit', 'proximity', 'micro',
        'test', 'operation', 'replace', 'faulty', 'components', 'lubricate',
        'adjust', 'position', 'calibrate', 'damaged', 'electrical', 'loads',
        'current', 'consumption', 'voltage', 'power', 'display', 'dark',
        'piston', 'rod', 'crank', 'speed', 'control', 'settings', 'obstructions',
        'circuit', 'breakers', 'isolating', 'cock', 'tdic', 'interlock',
        'actuator', 'valve', 'filter', 'regulator', 'bluetooth', 'gfA', 'stick',
        'app', 'software', 'firmware', 'continuity', 'resistance', 'ohms',
        'calibration', 'detection', 'range', 'signal', 'transmission', 'triggering',
        'sensitivity', 'edges', 'profiles', 'gaskets', 'seals', 'water', 'damage',
        'deformed', 'rubber', 'profiles', 'safety', 'edges', 'dry', 'seal'
    }
    
    words = set(text.lower().split())
    found_terms = words.intersection(technical_terms)
    return found_terms

def analyze_explanatory_phrases(text):
    """Analyze the use of explanatory phrases that indicate beginner-friendly language"""
    explanatory_indicators = [
        'like', 'essentially', 'basically', 'in other words', 'that means',
        'this means', 'in simple terms', 'to put it simply', 'what this means is',
        'first', 'second', 'third', 'then', 'next', 'finally', 'when', 'if',
        'because', 'so that', 'in order to', 'you need to', 'you should',
        'check if', 'make sure', 'ensure that', 'verify that', 'is like',
        'works as', 'functions as', 'acts as', 'serves as', 'is a device that',
        'is a part that', 'is a tool that', 'is a system that', 'is a mechanism that',
        'is used to', 'is designed to', 'is meant to', 'is supposed to',
        'in simple terms', 'simply put', 'put simply', 'to explain', 'to clarify'
    ]
    
    text_lower = text.lower()
    found_explanations = [phrase for phrase in explanatory_indicators if phrase in text_lower]
    return found_explanations

def evaluate_communication_style(expert_response, novice_response, question):
    """Evaluate the communication style differences"""
    print(f"\nQuestion: {question}")
    print("=" * 80)
    
    # Analyze technical terms
    expert_terms = analyze_technical_terms(expert_response)
    novice_terms = analyze_technical_terms(novice_response)
    
    # Analyze explanatory phrases
    expert_explanations = analyze_explanatory_phrases(expert_response)
    novice_explanations = analyze_explanatory_phrases(novice_response)
    
    print("EXPERT RESPONSE:")
    print(f"  Length: {len(expert_response)} chars")
    print(f"  Technical terms: {', '.join(sorted(expert_terms))}")
    print(f"  Explanatory phrases: {len(expert_explanations)}")
    print(f"  Response: {expert_response}")
    
    print("\nNOVICE RESPONSE:")
    print(f"  Length: {len(novice_response)} chars")
    print(f"  Technical terms: {', '.join(sorted(novice_terms))}")
    print(f"  Explanatory phrases: {len(novice_explanations)}")
    print(f"  Response: {novice_response}")
    
    # Evaluate the difference
    print("\nANALYSIS:")
    
    # Length difference
    length_diff = len(novice_response) - len(expert_response)
    print(f"  Length difference: {length_diff} chars (novice - expert)")
    
    # Technical terms difference
    technical_diff = len(expert_terms) - len(novice_terms)
    print(f"  Technical terms: Expert has {len(expert_terms)}, Novice has {len(novice_terms)}")
    
    # Explanatory phrases difference
    explanation_diff = len(novice_explanations) - len(expert_explanations)
    print(f"  Explanatory phrases: Expert has {len(expert_explanations)}, Novice has {len(novice_explanations)}")
    
    # Overall assessment
    print("\nASSESSMENT:")
    if len(expert_terms) > len(novice_terms) and len(novice_explanations) > len(expert_explanations):
        print("  GOOD: Expert uses more technical terms, Novice uses more explanations")
    elif len(expert_terms) > len(novice_terms):
        print("  PARTIAL: Expert uses more technical terms, but Novice doesn't explain enough")
    elif len(novice_explanations) > len(expert_explanations):
        print("  PARTIAL: Novice uses more explanations, but Expert doesn't use enough technical terms")
    else:
        print("  POOR: No clear differentiation in communication style")
    
    return {
        'expert_length': len(expert_response),
        'novice_length': len(novice_response),
        'expert_technical_terms': len(expert_terms),
        'novice_technical_terms': len(novice_terms),
        'expert_explanations': len(expert_explanations),
        'novice_explanations': len(novice_explanations)
    }

def run_communication_style_evaluation():
    """Run evaluation on both models"""
    # Create log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"communication_style_evaluation_{timestamp}.log"
    
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
        questions = [
            "What steps are included in the general repair framework for DCU maintenance?",
            "What are the typical tools needed for DCU maintenance?",
            "How should a train operator respond when all doors fail to close?",
            "What are the common causes for slamming doors?",
            "What are the maintenance steps for the emergency release mechanism?"
        ]
        
        models = ['gemma3']
        results = {}
        
        for model in models:
            print(f"\n{'='*80}")
            print(f"EVALUATING MODEL: {model.upper()}")
            print(f"{'='*80}")
            
            model_results = []
            
            for question in questions:
                try:
                    print(f"\nTesting: {question}")
                    
                    # Get expert response
                    expert_response = answer_question(question, model, 'expert')
                    
                    # Get novice response
                    novice_response = answer_question(question, model, 'beginner')
                    
                    # Evaluate communication style
                    result = evaluate_communication_style(expert_response, novice_response, question)
                    model_results.append(result)
                    
                    time.sleep(1)  # Brief pause between questions
                    
                except Exception as e:
                    print(f"Error testing question: {e}")
            
            # Calculate averages
            if model_results:
                avg_expert_length = sum(r['expert_length'] for r in model_results) / len(model_results)
                avg_novice_length = sum(r['novice_length'] for r in model_results) / len(model_results)
                avg_expert_technical = sum(r['expert_technical_terms'] for r in model_results) / len(model_results)
                avg_novice_technical = sum(r['novice_technical_terms'] for r in model_results) / len(model_results)
                avg_expert_explanations = sum(r['expert_explanations'] for r in model_results) / len(model_results)
                avg_novice_explanations = sum(r['novice_explanations'] for r in model_results) / len(model_results)
                
                results[model] = {
                    'avg_expert_length': avg_expert_length,
                    'avg_novice_length': avg_novice_length,
                    'avg_expert_technical': avg_expert_technical,
                    'avg_novice_technical': avg_novice_technical,
                    'avg_expert_explanations': avg_expert_explanations,
                    'avg_novice_explanations': avg_novice_explanations
                }
        
        # Summary
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        for model, metrics in results.items():
            print(f"\n{model.upper()}:")
            print(f"  Average Expert Length: {metrics['avg_expert_length']:.0f} chars")
            print(f"  Average Novice Length: {metrics['avg_novice_length']:.0f} chars")
            print(f"  Average Expert Technical Terms: {metrics['avg_expert_technical']:.1f}")
            print(f"  Average Novice Technical Terms: {metrics['avg_novice_technical']:.1f}")
            print(f"  Average Expert Explanations: {metrics['avg_expert_explanations']:.1f}")
            print(f"  Average Novice Explanations: {metrics['avg_novice_explanations']:.1f}")
            
            # Overall assessment
            if (metrics['avg_expert_technical'] > metrics['avg_novice_technical'] and 
                metrics['avg_novice_explanations'] > metrics['avg_expert_explanations']):
                print("  EXCELLENT: Clear communication style differentiation")
            elif metrics['avg_expert_technical'] > metrics['avg_novice_technical']:
                print("  GOOD: Expert uses more technical terms")
            elif metrics['avg_novice_explanations'] > metrics['avg_expert_explanations']:
                print("  GOOD: Novice uses more explanations")
            else:
                print("  POOR: No clear differentiation")
        
        print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {log_filename}")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        logger.log.close()

if __name__ == "__main__":
    run_communication_style_evaluation()
