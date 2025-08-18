#!/usr/bin/env python3
"""
Intelligent Railway RAG System - User Level Function Demo
Demonstrating the differences between beginner/expert modes and auto-detection functionality
"""

from backend import answer_question, analyze_user_level_from_question, get_user_level_info
import time

def demo_user_level_detection():
    """Demonstrate user level auto-detection functionality"""
    
    print("=" * 80)
    print("🤖 User Level Auto-Detection Demo")
    print("=" * 80)
    
    test_questions = [
        "What are the basic maintenance steps for door control units?",
        "What are the technical parameters and advanced troubleshooting methods for door control systems?",
        "How to safely perform door control maintenance?",
        "What are the integration optimization and performance tuning strategies for DCU systems?",
        "What should be paid attention to in door control maintenance?",
        "What are the CAN bus communication protocols and diagnostic methods for door control systems?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        detected_level = analyze_user_level_from_question(question)
        level_info = get_user_level_info()[detected_level]
        
        print(f"\n{i}. Question: {question}")
        print(f"   Detected Level: {level_info['name']} ({detected_level})")
        print(f"   Level Description: {level_info['description']}")
        print("-" * 60)

def demo_level_comparison():
    """Demonstrate comparison of different level responses"""
    
    print("\n" + "=" * 80)
    print("🎯 User Level Response Comparison Demo")
    print("=" * 80)
    
    test_question = "What are the basic maintenance steps for door control units?"
    
    print(f"Test Question: {test_question}")
    print("=" * 60)
    
    # Beginner mode response
    print("\n🆕 Beginner Mode Response:")
    print("-" * 40)
    start_time = time.time()
    beginner_answer = answer_question(test_question, user_level="beginner", document_type="door_control")
    end_time = time.time()
    print(f"Response Length: {len(beginner_answer)} characters")
    print(f"Generation Time: {end_time - start_time:.2f} seconds")
    print("Response Preview:")
    print(beginner_answer[:500] + "...")
    
    # Expert mode response
    print("\n🔧 Expert Mode Response:")
    print("-" * 40)
    start_time = time.time()
    expert_answer = answer_question(test_question, user_level="expert", document_type="door_control")
    end_time = time.time()
    print(f"Response Length: {len(expert_answer)} characters")
    print(f"Generation Time: {end_time - start_time:.2f} seconds")
    print("Response Preview:")
    print(expert_answer[:500] + "...")
    
    # Auto-detection mode
    print("\n🤖 Auto-Detection Mode Response:")
    print("-" * 40)
    start_time = time.time()
    auto_answer = answer_question(test_question, document_type="door_control")
    end_time = time.time()
    print(f"Response Length: {len(auto_answer)} characters")
    print(f"Generation Time: {end_time - start_time:.2f} seconds")
    print("Response Preview:")
    print(auto_answer[:500] + "...")

def demo_advanced_questions():
    """Demonstrate auto-detection of advanced questions"""
    
    print("\n" + "=" * 80)
    print("🔬 Advanced Question Auto-Detection Demo")
    print("=" * 80)
    
    advanced_questions = [
        "What are the CAN bus communication protocols and diagnostic methods for door control systems?",
        "What are the real-time operating system scheduling algorithms and memory management strategies for DCU?",
        "How to implement fault prediction and preventive maintenance algorithms for door control systems?",
        "How to build machine learning-based anomaly detection models for door control systems?"
    ]
    
    for i, question in enumerate(advanced_questions, 1):
        print(f"\n{i}. Advanced Question: {question}")
        
        # Auto-detection
        detected_level = analyze_user_level_from_question(question)
        level_info = get_user_level_info()[detected_level]
        print(f"   Auto-detected Level: {level_info['name']}")
        
        # Generate response
        start_time = time.time()
        answer = answer_question(question, document_type="door_control")
        end_time = time.time()
        
        print(f"   Response Length: {len(answer)} characters")
        print(f"   Generation Time: {end_time - start_time:.2f} seconds")
        print(f"   Response Preview: {answer[:200]}...")
        print("-" * 60)

def demo_beginner_questions():
    """Demonstrate auto-detection of beginner questions"""
    
    print("\n" + "=" * 80)
    print("🆕 Beginner Question Auto-Detection Demo")
    print("=" * 80)
    
    beginner_questions = [
        "What should be paid attention to in door control maintenance?",
        "How to safely perform door control maintenance?",
        "What are the basic components of door control systems?",
        "What tools are needed for door control maintenance?"
    ]
    
    for i, question in enumerate(beginner_questions, 1):
        print(f"\n{i}. Beginner Question: {question}")
        
        # Auto-detection
        detected_level = analyze_user_level_from_question(question)
        level_info = get_user_level_info()[detected_level]
        print(f"   Auto-detected Level: {level_info['name']}")
        
        # Generate response
        start_time = time.time()
        answer = answer_question(question, document_type="door_control")
        end_time = time.time()
        
        print(f"   Response Length: {len(answer)} characters")
        print(f"   Generation Time: {end_time - start_time:.2f} seconds")
        print(f"   Response Preview: {answer[:200]}...")
        print("-" * 60)

def demo_system_capabilities():
    """Demonstrate overall system capabilities"""
    
    print("\n" + "=" * 80)
    print("🚀 System Capabilities Summary")
    print("=" * 80)
    
    user_levels = get_user_level_info()
    
    print("Supported User Levels:")
    for level, info in user_levels.items():
        print(f"\n{info['name']} ({level}):")
        print(f"  Description: {info['description']}")
        print(f"  Characteristics:")
        for char in info['characteristics']:
            print(f"    - {char}")
    
    print(f"\nSystem Features:")
    print("  ✅ Intelligent User Level Detection")
    print("  ✅ Adaptive Response Generation")
    print("  ✅ Multi-Document Type Support")
    print("  ✅ Autonomous Learning Capability")
    print("  ✅ Real-time Level Adjustment")
    print("  ✅ Professional Terminology Recognition")
    print("  ✅ Safety Reminder Integration")
    print("  ✅ Technical Parameter Provision")

def main():
    """Main demonstration function"""
    
    print("🚂 Intelligent Railway RAG System - User Level Function Demo")
    print("=" * 80)
    print("This demo will showcase the system's autonomous learning capability and user level adaptive functionality")
    print("=" * 80)
    
    try:
        # 1. User level detection demo
        demo_user_level_detection()
        
        # 2. Level comparison demo
        demo_level_comparison()
        
        # 3. Advanced questions demo
        demo_advanced_questions()
        
        # 4. Beginner questions demo
        demo_beginner_questions()
        
        # 5. System capabilities summary
        demo_system_capabilities()
        
        print("\n" + "=" * 80)
        print("🎉 Demo Completed!")
        print("=" * 80)
        print("The system has successfully demonstrated the following features:")
        print("  • Intelligent user level auto-detection")
        print("  • Beginner/Expert mode response comparison")
        print("  • Autonomous learning capability")
        print("  • Adaptive response generation")
        print("\nYou can now:")
        print("  • Use simple_ui.html for web interface testing")
        print("  • Make programmatic calls via API")
        print("  • Customize user level configurations")
        
    except Exception as e:
        print(f"Error occurred during demo: {e}")
        print("Please ensure the system is properly configured and running")

if __name__ == "__main__":
    main()
