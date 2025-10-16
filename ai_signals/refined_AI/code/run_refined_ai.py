#!/usr/bin/env python3
# Refined AI Launcher
# Main entry point for the refined AI trading system

import os
import sys
import argparse
from datetime import datetime

def run_main_processor(interactive=True):
    """Run the main AI processor to generate signals"""
    print("🚀 Starting Refined AI Signal Generation...")
    print("=" * 60)
    
    try:
        from main_processor import RefinedAIProcessor
        processor = RefinedAIProcessor(interactive_mode=interactive)
        results = processor.process_all_symbols()
        
        print("\n✅ Signal generation completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in signal generation: {e}")
        return False

def run_accuracy_checker(symbol=None):
    """Run the accuracy checker"""
    print("🔍 Starting Refined AI Accuracy Check...")
    print("=" * 60)
    
    try:
        from accuracy_checker import RefinedAIAccuracyChecker
        checker = RefinedAIAccuracyChecker()
        results, metrics = checker.run_accuracy_check(symbol=symbol)
        
        print("\n✅ Accuracy check completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in accuracy check: {e}")
        return False

def run_system_test():
    """Run the system test suite"""
    print("🧪 Running Refined AI System Tests...")
    print("=" * 60)
    
    try:
        from test_system import main as test_main
        success = test_main()
        
        if success:
            print("\n✅ All system tests passed!")
        else:
            print("\n❌ Some system tests failed!")
        
        return success
    except Exception as e:
        print(f"❌ Error running system tests: {e}")
        return False

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description='Refined AI Trading System Launcher')
    parser.add_argument('command', choices=['generate', 'check', 'test', 'all'], 
                       help='Command to run: generate (signals), check (accuracy), test (system), all')
    parser.add_argument('--symbol', type=str, help='Specific symbol to check (for accuracy check)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run in non-interactive mode (skip outdated data without prompting)')
    
    args = parser.parse_args()
    
    print("🤖 Refined AI Trading System")
    print(f"Command: {args.command}")
    print(f"Mode: {'Non-interactive' if args.non_interactive else 'Interactive'}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    success = True
    
    if args.command == 'generate':
        success = run_main_processor(interactive=not args.non_interactive)
    elif args.command == 'check':
        success = run_accuracy_checker(symbol=args.symbol)
    elif args.command == 'test':
        success = run_system_test()
    elif args.command == 'all':
        print("🔄 Running complete workflow...")
        print("\n1. Testing system...")
        if not run_system_test():
            print("❌ System tests failed, stopping workflow")
            return False
        
        print("\n2. Generating signals...")
        if not run_main_processor(interactive=not args.non_interactive):
            print("❌ Signal generation failed, stopping workflow")
            return False
        
        print("\n3. Checking accuracy...")
        if not run_accuracy_checker():
            print("❌ Accuracy check failed")
            success = False
    
    if success:
        print("\n🎉 Refined AI workflow completed successfully!")
    else:
        print("\n💥 Refined AI workflow encountered errors!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
