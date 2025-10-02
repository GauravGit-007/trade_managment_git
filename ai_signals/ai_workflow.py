# AI Signals Workflow
# This module orchestrates the complete AI signals workflow

import os
import sys
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_decision_engine import AIDecisionEngine
from ai_evaluation import AIEvaluation
from ai_monitoring import AIMonitoring

class AIWorkflow:
    def __init__(self):
        """Initialize the AI workflow system"""
        self.decision_engine = AIDecisionEngine()
        self.evaluator = AIEvaluation()
        self.monitor = AIMonitoring()
        
    def run_full_workflow(self):
        """Run the complete AI signals workflow"""
        print("🚀 Starting AI Signals Full Workflow")
        print("=" * 50)
        
        start_time = datetime.utcnow()
        
        # Step 1: Generate AI signals
        print("\n1️⃣ Generating AI Signals...")
        try:
            decisions = self.decision_engine.generate_signals_for_all_symbols()
            print(f"✅ Generated {len(decisions)} AI signals")
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return False
        
        # Step 2: Evaluate recent decisions
        print("\n2️⃣ Evaluating Recent Decisions...")
        try:
            evaluation_results = self.evaluator.evaluate_all_decisions(hours_back=24)
            print("✅ Evaluation completed")
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
        
        # Step 3: Generate monitoring report
        print("\n3️⃣ Generating Monitoring Report...")
        try:
            monitoring_report = self.monitor.generate_monitoring_report(hours_back=6)
            print("✅ Monitoring report generated")
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
        
        # Step 4: Print summary
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 Workflow completed in {duration:.1f} seconds")
        print("=" * 50)
        
        return True
    
    def run_signal_generation_only(self):
        """Run only the signal generation step"""
        print("🤖 Running AI Signal Generation Only")
        print("=" * 40)
        
        try:
            decisions = self.decision_engine.generate_signals_for_all_symbols()
            print(f"✅ Generated {len(decisions)} AI signals")
            
            # Print signal summary
            print("\n📋 SIGNAL SUMMARY:")
            for symbol, decision in decisions.items():
                print(f"   {symbol}: {decision['action']} (confidence: {decision['confidence']:.2f})")
            
            return True
        except Exception as e:
            print(f"❌ Error generating signals: {e}")
            return False
    
    def run_evaluation_only(self, hours_back=24):
        """Run only the evaluation step"""
        print(f"📊 Running AI Evaluation (last {hours_back} hours)")
        print("=" * 40)
        
        try:
            results = self.evaluator.evaluate_all_decisions(hours_back)
            self.evaluator.print_evaluation_summary()
            return True
        except Exception as e:
            print(f"❌ Error in evaluation: {e}")
            return False
    
    def run_monitoring_only(self, hours_back=6):
        """Run only the monitoring step"""
        print(f"🔍 Running AI Monitoring (last {hours_back} hours)")
        print("=" * 40)
        
        try:
            report = self.monitor.generate_monitoring_report(hours_back)
            return True
        except Exception as e:
            print(f"❌ Error in monitoring: {e}")
            return False
    
    def run_continuous_monitoring(self, check_interval_minutes=30):
        """Run continuous monitoring"""
        print(f"🔄 Starting Continuous Monitoring (every {check_interval_minutes} minutes)")
        print("=" * 50)
        
        try:
            self.monitor.start_continuous_monitoring(check_interval_minutes)
            return True
        except Exception as e:
            print(f"❌ Error in continuous monitoring: {e}")
            return False

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='AI Signals Workflow')
    parser.add_argument('--mode', choices=['full', 'signals', 'evaluate', 'monitor', 'continuous'], 
                       default='full', help='Workflow mode to run')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Hours to look back for evaluation/monitoring')
    parser.add_argument('--interval', type=int, default=30,
                       help='Check interval in minutes for continuous monitoring')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = AIWorkflow()
    
    print(f"🎯 AI Signals Workflow - Mode: {args.mode.upper()}")
    print(f"⏰ Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)
    
    success = False
    
    if args.mode == 'full':
        success = workflow.run_full_workflow()
    elif args.mode == 'signals':
        success = workflow.run_signal_generation_only()
    elif args.mode == 'evaluate':
        success = workflow.run_evaluation_only(args.hours)
    elif args.mode == 'monitor':
        success = workflow.run_monitoring_only(args.hours)
    elif args.mode == 'continuous':
        success = workflow.run_continuous_monitoring(args.interval)
    
    if success:
        print("\n✅ Workflow completed successfully!")
    else:
        print("\n❌ Workflow failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
