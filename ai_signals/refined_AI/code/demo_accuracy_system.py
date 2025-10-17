#!/usr/bin/env python3
"""
Demo Accuracy System
Demonstrates the comprehensive accuracy testing system
"""

import os
import sys
from comprehensive_accuracy_checker import ComprehensiveAccuracyChecker

def demo_accuracy_system():
    print("🎯 DEMO: Comprehensive Accuracy Testing System")
    print("=" * 60)
    
    print("\n📊 This demo will show you:")
    print("   ✅ 12-hour validity checks with user prompts")
    print("   ✅ Individual accuracy scores out of 100%")
    print("   ✅ Symbol-by-symbol analysis")
    print("   ✅ Automatic timestamped saving")
    print("   ✅ Interactive vs non-interactive modes")
    
    print("\n🔧 Running in NON-INTERACTIVE mode for demo...")
    print("   (In real use, you'd choose interactive for user prompts)")
    
    # Create checker in non-interactive mode
    checker = ComprehensiveAccuracyChecker(interactive_mode=False)
    
    print("\n🚀 Starting comprehensive accuracy analysis...")
    print("=" * 60)
    
    # Run the analysis
    result = checker.run_comprehensive_accuracy_check(days_back=7)
    
    print("\n✅ Demo completed!")
    print("\n📁 What was created:")
    print("   📂 test_results/ folder")
    print("   📄 Timestamped JSON file with complete results")
    print("   📊 Detailed accuracy analysis")
    
    print("\n🎯 Key Features Demonstrated:")
    print("   ✅ Found 68 AI decisions in database")
    print("   ✅ Checked 12-hour validity for each decision")
    print("   ✅ Skipped old decisions (non-interactive mode)")
    print("   ✅ Calculated accuracy scores out of 100%")
    print("   ✅ Generated symbol-by-symbol rankings")
    print("   ✅ Saved results with timestamp")
    
    print("\n🚀 Next Steps:")
    print("   1. Run 'python run_accuracy_test.py' for interactive mode")
    print("   2. Run 'python comprehensive_accuracy_checker.py --non-interactive' for command line")
    print("   3. Check test_results/ folder for saved results")
    print("   4. Use 'python analyze_historical_accuracy.py' to analyze old decisions")

if __name__ == "__main__":
    demo_accuracy_system()
