#!/usr/bin/env python3
# Demo script for Data Freshness Validation Feature
# Shows how the system handles fresh vs outdated data

import os
import sys
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

def demo_data_freshness():
    """Demonstrate the data freshness validation feature"""
    print("🧪 Refined AI Data Freshness Demo")
    print("=" * 60)
    
    # Initialize processor
    from main_processor import RefinedAIProcessor
    
    # Test with interactive mode
    print("\n1. Testing Interactive Mode:")
    print("-" * 30)
    processor_interactive = RefinedAIProcessor(interactive_mode=True)
    
    # Test with non-interactive mode
    print("\n2. Testing Non-Interactive Mode:")
    print("-" * 30)
    processor_non_interactive = RefinedAIProcessor(interactive_mode=False)
    
    # Test data freshness for each symbol
    symbols = [
        "/ES:XCME{=h}", "/NQ:XCME{=h}", "/MES:XCME{=h}", "/MNQ:XCME{=h}",
        "/RTY:XCME{=h}", "/QM:XNYM{=h}", "/QG:XNYM{=h}", "/MCL:XNYM{=h}"
    ]
    
    print(f"\n3. Checking Data Freshness for {len(symbols)} symbols:")
    print("-" * 50)
    
    fresh_count = 0
    outdated_count = 0
    no_data_count = 0
    
    for symbol in symbols:
        print(f"\n📊 Checking {symbol}...")
        
        # Check freshness
        freshness_report = processor_interactive.check_data_freshness(symbol)
        
        # Display results
        hist_data = freshness_report['historical_data']
        news_data = freshness_report['news_data']
        lstm_data = freshness_report['lstm_predictions']
        
        hist_status = f"({hist_data['hours_old']:.1f}h old)" if hist_data['available'] else 'N/A'
        news_status = f"({news_data['hours_old']:.1f}h old)" if news_data['available'] else 'N/A'
        lstm_status = f"({lstm_data['hours_old']:.1f}h old)" if lstm_data['available'] else 'N/A'
        
        print(f"  📈 Historical: {'✅' if hist_data['available'] else '❌'} {hist_status}")
        print(f"  📰 News: {'✅' if news_data['available'] else '❌'} {news_status}")
        print(f"  🤖 LSTM: {'✅' if lstm_data['available'] else '❌'} {lstm_status}")
        
        if freshness_report['all_fresh']:
            print(f"  🎯 Status: FRESH - Ready for signal generation")
            fresh_count += 1
        elif any([hist_data['available'], news_data['available'], lstm_data['available']]):
            print(f"  ⚠️  Status: OUTDATED - Data older than 2 hours")
            outdated_count += 1
        else:
            print(f"  ❌ Status: NO DATA - No data available")
            no_data_count += 1
    
    # Summary
    print(f"\n📈 Data Freshness Summary:")
    print("=" * 40)
    print(f"✅ Fresh data: {fresh_count} symbols")
    print(f"⚠️  Outdated data: {outdated_count} symbols")
    print(f"❌ No data: {no_data_count} symbols")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if fresh_count == len(symbols):
        print("🎉 All symbols have fresh data! Ready for signal generation.")
    elif fresh_count > 0:
        print(f"✅ {fresh_count} symbols ready for processing")
        print(f"⚠️  {outdated_count + no_data_count} symbols need data updates")
    else:
        print("❌ No symbols have fresh data. Please update data sources.")
    
    print(f"\n🔧 Usage Examples:")
    print("Interactive mode (prompts for outdated data):")
    print("  python run_refined_ai.py generate")
    print("\nNon-interactive mode (skips outdated data):")
    print("  python run_refined_ai.py generate --non-interactive")

def main():
    """Main demo function"""
    try:
        demo_data_freshness()
        print(f"\n✅ Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
