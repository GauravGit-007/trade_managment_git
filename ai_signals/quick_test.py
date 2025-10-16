#!/usr/bin/env python3
"""
Ultra Simple AI Signals Test

Just run: python quick_test.py

This will:
1. Generate signals
2. Check accuracy
3. Show results
"""

import os
import sys
import subprocess
from datetime import datetime

def main():
    print("🚀 AI SIGNALS - QUICK TEST")
    print("=" * 40)
    
    # Generate signals
    print("\n1️⃣ Generating signals...")
    os.system("python utils/ai_realtime.py")
    
    # Check accuracy
    print("\n2️⃣ Checking accuracy...")
    os.system("python accuracy/ai_accuracy.py daily")
    
    print("\n✅ DONE! Check results above.")
    print("💡 Run again: python quick_test.py")

if __name__ == "__main__":
    main()
