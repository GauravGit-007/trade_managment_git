# AI Signals Demo Script
# This script demonstrates how to use the AI signals system

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def demo_signal_generation():
    """Demonstrate AI signal generation"""
    print("🤖 AI SIGNAL GENERATION DEMO")
    print("=" * 40)
    
    try:
        from ai_decision_engine import AIDecisionEngine
        
        # Initialize the AI decision engine
        print("Initializing AI Decision Engine...")
        engine = AIDecisionEngine()
        
        # Generate signals for a few symbols (demo with limited symbols)
        demo_symbols = ["/ES:XCME{=h}", "/NQ:XCME{=h}", "BTC/USD:CXTALP{=h}"]
        
        print(f"Generating signals for {len(demo_symbols)} symbols...")
        print("Note: This will use real Azure OpenAI API calls")
        
        decisions = {}
        for symbol in demo_symbols:
            print(f"\n📊 Analyzing {symbol}...")
            
            # Get market context
            market_context = engine.get_market_context(symbol)
            if not market_context:
                print(f"❌ No data available for {symbol}")
                continue
            
            print(f"   Current Price: ${market_context['current_price']:.2f}")
            print(f"   News Articles: {market_context['news_sentiment']['article_count']}")
            print(f"   Avg Sentiment: {market_context['news_sentiment']['avg_sentiment']:.3f}")
            
            # Generate AI signal
            decision = engine.generate_ai_signal(market_context)
            
            # Log decision
            engine.log_decision(decision)
            
            # Store for display
            decisions[symbol] = decision
            
            print(f"   ✅ Signal: {decision['action']} (confidence: {decision['confidence']:.2f})")
            print(f"   💭 Reasoning: {decision['reasoning'][:100]}...")
        
        # Save demo results
        output_path = os.path.join(os.path.dirname(__file__), "demo_signals.json")
        with open(output_path, "w") as f:
            json.dump(decisions, f, indent=2)
        
        print(f"\n📁 Demo signals saved to {output_path}")
        
        return decisions
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return None

def demo_evaluation():
    """Demonstrate evaluation system"""
    print("\n📊 AI EVALUATION DEMO")
    print("=" * 40)
    
    try:
        from ai_evaluation import AIEvaluation
        
        # Initialize evaluator
        evaluator = AIEvaluation()
        
        # Run evaluation
        print("Running evaluation on recent decisions...")
        results = evaluator.evaluate_all_decisions(hours_back=24)
        
        if results and results['overall_statistics']['evaluated_decisions'] > 0:
            evaluator.print_evaluation_summary()
        else:
            print("ℹ️ No decisions found for evaluation (this is normal for first run)")
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation demo failed: {e}")
        return None

def demo_monitoring():
    """Demonstrate monitoring system"""
    print("\n🔍 AI MONITORING DEMO")
    print("=" * 40)
    
    try:
        from ai_monitoring import AIMonitoring
        
        # Initialize monitor
        monitor = AIMonitoring()
        
        # Generate monitoring report
        print("Generating monitoring report...")
        report = monitor.generate_monitoring_report(hours_back=6)
        
        return report
        
    except Exception as e:
        print(f"❌ Monitoring demo failed: {e}")
        return None

def demo_workflow():
    """Demonstrate complete workflow"""
    print("\n🔄 AI WORKFLOW DEMO")
    print("=" * 40)
    
    try:
        from ai_workflow import AIWorkflow
        
        # Initialize workflow
        workflow = AIWorkflow()
        
        # Run signal generation only (safer for demo)
        print("Running signal generation workflow...")
        success = workflow.run_signal_generation_only()
        
        if success:
            print("✅ Workflow completed successfully!")
        else:
            print("❌ Workflow failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Workflow demo failed: {e}")
        return False

def main():
    """Run all demos"""
    print("🚀 AI SIGNALS SYSTEM DEMO")
    print("=" * 50)
    print(f"⏰ Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 50)
    
    # Check if user wants to proceed with API calls
    print("\n⚠️  WARNING: This demo will make real API calls to Azure OpenAI")
    print("This will consume API credits and may take a few minutes.")
    
    response = input("\nDo you want to proceed? (y/N): ").strip().lower()
    if response != 'y':
        print("Demo cancelled. No API calls made.")
        return
    
    # Run demos
    demos = [
        ("Signal Generation", demo_signal_generation),
        ("Evaluation System", demo_evaluation),
        ("Monitoring System", demo_monitoring),
        ("Complete Workflow", demo_workflow)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\n🎯 Running {demo_name} Demo...")
        try:
            result = demo_func()
            results[demo_name] = "SUCCESS" if result else "FAILED"
        except Exception as e:
            print(f"❌ {demo_name} failed with error: {e}")
            results[demo_name] = "ERROR"
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 DEMO SUMMARY")
    print("=" * 50)
    
    for demo_name, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "❌"
        print(f"{status_icon} {demo_name}: {status}")
    
    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Check the generated files in the ai_signals directory")
    print("2. Run 'python ai_workflow.py --mode full' for complete workflow")
    print("3. Use 'python ai_workflow.py --mode continuous' for live monitoring")

if __name__ == "__main__":
    main()
