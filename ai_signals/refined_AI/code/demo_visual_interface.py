#!/usr/bin/env python3
"""
Demo Visual Accuracy Interface
Shows how the visual interface displays results
"""

import os
import json
import sys
from datetime import datetime
from visual_accuracy_interface import VisualAccuracyInterface

def demo_visual_interface():
    print("🎯 DEMO: Visual Accuracy Results Interface")
    print("=" * 60)
    
    interface = VisualAccuracyInterface()
    
    if not interface.available_tests:
        print("❌ No test results found in test_results/ folder")
        print("💡 Let's create a sample test result first...")
        return
    
    print(f"📊 Found {len(interface.available_tests)} test result(s)")
    print("\n🔄 Loading the most recent test...")
    
    # Get the most recent test
    latest_test = interface.available_tests[0]
    test_data = interface.load_test_data(latest_test)
    
    if not test_data:
        print("❌ Failed to load test data")
        return
    
    print("✅ Test data loaded successfully!")
    print("\n🎨 Displaying visual results...")
    print("=" * 60)
    
    # Display the visual results
    interface.display_visual_results(test_data)
    
    print("\n🎯 This is how the visual interface displays accuracy results!")
    print("💡 Run 'python visual_accuracy_interface.py' for the full interactive experience")

if __name__ == "__main__":
    demo_visual_interface()
