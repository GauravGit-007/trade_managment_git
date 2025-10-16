#!/usr/bin/env python3
# Test script to demonstrate interactive mode with simulated user input

import os
import sys
from unittest.mock import patch

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def test_interactive_mode():
    """Test interactive mode with simulated user input"""
    print("🧪 Testing Interactive Mode with Simulated User Input")
    print("=" * 60)
    
    from main_processor import RefinedAIProcessor
    
    # Test with 'y' response (continue with outdated data)
    print("\n1. Testing with 'y' response (continue with outdated data):")
    print("-" * 50)
    
    with patch('builtins.input', return_value='y'):
        processor = RefinedAIProcessor(interactive_mode=True)
        # Test just one symbol to avoid too much output
        symbol = "/ES:XCME{=h}"
        freshness_report = processor.check_data_freshness(symbol)
        
        if not freshness_report['all_fresh']:
            print(f"Testing user prompt for {symbol}...")
            result = processor.prompt_user_continuation(symbol, freshness_report)
            print(f"User chose to continue: {result}")
    
    # Test with 'n' response (skip outdated data)
    print("\n2. Testing with 'n' response (skip outdated data):")
    print("-" * 50)
    
    with patch('builtins.input', return_value='n'):
        processor = RefinedAIProcessor(interactive_mode=True)
        symbol = "/NQ:XCME{=h}"
        freshness_report = processor.check_data_freshness(symbol)
        
        if not freshness_report['all_fresh']:
            print(f"Testing user prompt for {symbol}...")
            result = processor.prompt_user_continuation(symbol, freshness_report)
            print(f"User chose to continue: {result}")
    
    print("\n✅ Interactive mode testing completed!")

def test_non_interactive_mode():
    """Test non-interactive mode"""
    print("\n🤖 Testing Non-Interactive Mode")
    print("=" * 40)
    
    from main_processor import RefinedAIProcessor
    
    processor = RefinedAIProcessor(interactive_mode=False)
    symbol = "/ES:XCME{=h}"
    freshness_report = processor.check_data_freshness(symbol)
    
    if not freshness_report['all_fresh']:
        print(f"Testing non-interactive prompt for {symbol}...")
        result = processor.prompt_user_continuation(symbol, freshness_report)
        print(f"Non-interactive mode result: {result}")
    
    print("✅ Non-interactive mode testing completed!")

def main():
    """Main test function"""
    try:
        test_interactive_mode()
        test_non_interactive_mode()
        print("\n🎉 All tests completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
