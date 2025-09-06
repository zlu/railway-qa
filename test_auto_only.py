#!/usr/bin/env python3
"""
Test script to demonstrate that the system always uses auto-detection
regardless of the user_level parameter sent in the request.
"""

import requests

def test_auto_detection_only():
    """Test that auto-detection is always used regardless of parameters"""
    
    base_url = "http://localhost:8000"
    
    test_cases = [
        {
            "name": "Beginner question with expert parameter",
            "question": "What is a DCU?",
            "user_level": "expert",
            "auto_detect_level": False,
            "expected_detection": "beginner"
        },
        {
            "name": "Expert question with beginner parameter", 
            "question": "Troubleshooting TDIC circuit breaker overload issues",
            "user_level": "beginner",
            "auto_detect_level": False,
            "expected_detection": "expert"
        },
        {
            "name": "Beginner question with minimal parameters",
            "question": "How do I check if the door is working?",
            "user_level": "expert",
            "auto_detect_level": False,
            "expected_detection": "beginner"
        }
    ]
    
    print("Testing Auto-Detection Only Mode")
    print("=" * 50)
    print("The system should always use auto-detection regardless of parameters\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['name']}")
        print(f"   Question: {test_case['question']}")
        print(f"   Sent user_level: {test_case['user_level']}")
        print(f"   Sent auto_detect_level: {test_case['auto_detect_level']}")
        
        # Make the request
        payload = {
            "question": test_case['question'],
            "user_level": test_case['user_level'],
            "auto_detect_level": test_case['auto_detect_level']
        }
        
        try:
            response = requests.post(f"{base_url}/ask", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                detected_level = result['user_level']
                auto_detected = result['auto_detected']
                confidence = result['detection_analysis']['confidence']
                
                print(f"   ✅ Detected Level: {detected_level}")
                print(f"   ✅ Auto-detected: {auto_detected}")
                print(f"   ✅ Confidence: {confidence}")
                
                if detected_level == test_case['expected_detection']:
                    print(f"   ✅ CORRECT: Expected {test_case['expected_detection']}, got {detected_level}")
                else:
                    print(f"   ❌ INCORRECT: Expected {test_case['expected_detection']}, got {detected_level}")
                    
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            
        print("-" * 50)

if __name__ == "__main__":
    test_auto_detection_only()
