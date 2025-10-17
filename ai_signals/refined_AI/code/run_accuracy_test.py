#!/usr/bin/env python3
"""
Accuracy Test Launcher
Easy-to-use launcher for comprehensive accuracy testing
"""

import sys
import os
from comprehensive_accuracy_checker import ComprehensiveAccuracyChecker

def main():
    print("🎯 REFINED AI ACCURACY TEST LAUNCHER")
    print("=" * 50)
    
    # Get user preferences
    print("\n📊 Test Configuration:")
    
    # Symbol selection
    print("\n1. Symbol Selection:")
    print("   a) All symbols (default)")
    print("   b) Specific symbol")
    symbol_choice = input("   Choose (a/b): ").lower().strip()
    
    symbol = None
    if symbol_choice == 'b':
        symbol = input("   Enter symbol (e.g., /ES:XCME{=h}): ").strip()
        if not symbol:
            print("   ⚠️  No symbol entered, using all symbols")
            symbol = None
    
    # Days back
    print("\n2. Analysis Period:")
    days_input = input("   Days back to analyze (default: 7): ").strip()
    try:
        days_back = int(days_input) if days_input else 7
    except ValueError:
        print("   ⚠️  Invalid input, using default: 7 days")
        days_back = 7
    
    # Interactive mode
    print("\n3. Interactive Mode:")
    print("   a) Interactive (prompts for old decisions) - Recommended")
    print("   b) Non-interactive (skips old decisions automatically)")
    mode_choice = input("   Choose (a/b): ").lower().strip()
    interactive_mode = mode_choice != 'b'
    
    # Confirmation
    print(f"\n📋 Test Configuration Summary:")
    print(f"   Symbol: {symbol or 'All symbols'}")
    print(f"   Period: Last {days_back} days")
    print(f"   Mode: {'Interactive' if interactive_mode else 'Non-interactive'}")
    
    confirm = input("\n🤔 Proceed with this configuration? (y/n): ").lower().strip()
    if confirm not in ['y', 'yes']:
        print("❌ Test cancelled by user")
        return
    
    # Run the test
    print(f"\n🚀 Starting comprehensive accuracy test...")
    print("=" * 50)
    
    checker = ComprehensiveAccuracyChecker(interactive_mode=interactive_mode)
    result = checker.run_comprehensive_accuracy_check(symbol=symbol, days_back=days_back)
    
    print(f"\n✅ Test completed successfully!")
    print(f"📁 Results saved to test_results/ folder")

if __name__ == "__main__":
    main()
