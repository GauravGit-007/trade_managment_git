#!/usr/bin/env python3
"""
AI Signals Real-Time Generator

Generates AI signals immediately when executed - no waiting, no intervals.
Perfect for testing and immediate signal generation.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_decision_engine import AIDecisionEngine

def generate_realtime_signals():
    """Generate AI signals immediately"""
    print("🚀 AI SIGNALS - REAL-TIME GENERATION")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 50)
    
    try:
        # Initialize decision engine
        decision_engine = AIDecisionEngine()
        
        print("\n🤖 Generating AI signals for all symbols...")
        print("-" * 40)
        
        # Generate signals for all symbols
        decisions = decision_engine.generate_signals_for_all_symbols()
        
        if not decisions:
            print("❌ No signals generated!")
            return False
        
        # Display results
        print(f"\n✅ Generated {len(decisions)} AI signals")
        print("\n📊 SIGNAL RESULTS:")
        print("=" * 50)
        
        for symbol, decision in decisions.items():
            action = decision['action']
            confidence = decision['confidence']
            current_price = decision.get('current_price', 'N/A')
            reasoning = decision.get('reasoning', 'No reasoning provided')
            
            # Color coding for actions
            if action in ['STRONG_BUY', 'BUY']:
                action_emoji = "🟢"
            elif action in ['STRONG_SELL', 'SELL']:
                action_emoji = "🔴"
            else:
                action_emoji = "🟡"
            
            print(f"\n{symbol}:")
            print(f"  {action_emoji} Action: {action}")
            print(f"  💪 Confidence: {confidence:.2f}")
            print(f"  💰 Current Price: ${current_price}")
            print(f"  🧠 Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
        
        print("\n" + "=" * 50)
        print("🎯 REAL-TIME SIGNALS COMPLETE!")
        print("💡 Run 'python accuracy/ai_accuracy.py real' to check accuracy")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating real-time signals: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Signals Real-Time Generator')
    parser.add_argument('--symbol', type=str, help='Generate signal for specific symbol only')
    
    args = parser.parse_args()
    
    if args.symbol:
        print(f"🎯 Generating signal for: {args.symbol}")
        # TODO: Implement single symbol generation
        print("Single symbol generation not yet implemented")
        return
    
    success = generate_realtime_signals()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()


