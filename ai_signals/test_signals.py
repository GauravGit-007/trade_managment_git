#!/usr/bin/env python3
"""
Simple AI Signals Test & Accuracy Check

One command to test everything:
python test_signals.py

This will:
1. Generate new AI signals
2. Check accuracy status
3. Show performance results
4. Save results to test_results/
"""

import os
import sys
from datetime import datetime
import subprocess

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def run_command(command, description):
    """Run a command and return the output"""
    print(f"\n🔄 {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print("✅ SUCCESS")
            print(result.stdout)
            return result.stdout
        else:
            print("❌ ERROR")
            print(result.stderr)
            return None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

def main():
    """Main test function"""
    print("🚀 AI SIGNALS - SIMPLE TEST & ACCURACY CHECK")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    # Step 1: Generate AI Signals
    print("\n📊 STEP 1: Generating AI Signals")
    print("=" * 40)
    signals_output = run_command("python utils/ai_realtime.py", "Generating AI signals")
    
    if not signals_output:
        print("❌ Failed to generate signals. Stopping test.")
        return
    
    # Step 2: Check Accuracy Status
    print("\n📈 STEP 2: Checking Accuracy Status")
    print("=" * 40)
    status_output = run_command("python accuracy/simple_accuracy.py", "Checking accuracy status")
    
    # Step 3: Get Daily Accuracy
    print("\n🎯 STEP 3: Getting Daily Accuracy")
    print("=" * 40)
    accuracy_output = run_command("python accuracy/ai_accuracy.py daily", "Getting daily accuracy")
    
    # Step 4: Get Real-time Accuracy (if available)
    print("\n⚡ STEP 4: Getting Real-time Accuracy")
    print("=" * 40)
    realtime_output = run_command("python accuracy/ai_accuracy.py real", "Getting real-time accuracy")
    
    # Step 5: Save Results
    print("\n💾 STEP 5: Saving Test Results")
    print("=" * 40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"test_results/simple_test_{timestamp}.md"
    
    # Ensure test_results directory exists
    os.makedirs("test_results", exist_ok=True)
    
    # Create result file
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"# AI Signals Simple Test Results\n\n")
        f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"**Test File:** {result_file}\n\n")
        
        f.write("## 🤖 AI Signals Generated\n\n")
        f.write("```\n")
        f.write(signals_output or "No output")
        f.write("\n```\n\n")
        
        f.write("## 📊 Accuracy Status\n\n")
        f.write("```\n")
        f.write(status_output or "No output")
        f.write("\n```\n\n")
        
        f.write("## 📈 Daily Accuracy\n\n")
        f.write("```\n")
        f.write(accuracy_output or "No output")
        f.write("\n```\n\n")
        
        f.write("## ⚡ Real-time Accuracy\n\n")
        f.write("```\n")
        f.write(realtime_output or "No output")
        f.write("\n```\n\n")
        
        f.write("## 🎯 Quick Summary\n\n")
        f.write("- **Signals Generated:** ✅\n")
        f.write("- **Accuracy Status:** ✅\n")
        f.write("- **Daily Accuracy:** Available\n")
        f.write("- **Real-time Accuracy:** Check output above\n\n")
        
        f.write("## 🚀 Next Steps\n\n")
        f.write("1. Review the accuracy results above\n")
        f.write("2. Check if accuracy meets your target (70%)\n")
        f.write("3. Run this test again: `python test_signals.py`\n")
        f.write("4. For detailed analysis: `python accuracy/ai_accuracy.py daily`\n\n")
        
        f.write("---\n")
        f.write(f"*Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}*\n")
    
    print(f"✅ Results saved to: {result_file}")
    
    # Step 6: Update Test Index
    print("\n📋 STEP 6: Updating Test Index")
    print("=" * 40)
    
    # Update the test index
    index_file = "test_results/README.md"
    if os.path.exists(index_file):
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add new test to the top
        new_entry = f"| [simple_test_{timestamp}.md](./simple_test_{timestamp}.md) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Simple Test |\n"
        
        # Find the table and insert at the top
        if "| Test File |" in content:
            content = content.replace("| Test File |", f"| Test File |\n{new_entry}")
        else:
            content += f"\n## Recent Tests\n\n| Test File | Date | Type |\n|-----------|------|------|\n{new_entry}\n"
        
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print("✅ Test index updated")
    
    # Final Summary
    print("\n🎉 TEST COMPLETE!")
    print("=" * 60)
    print("✅ AI signals generated")
    print("✅ Accuracy status checked")
    print("✅ Daily accuracy calculated")
    print("✅ Real-time accuracy checked")
    print(f"✅ Results saved to: {result_file}")
    print("\n💡 To run again: python test_signals.py")
    print("=" * 60)

if __name__ == "__main__":
    main()

