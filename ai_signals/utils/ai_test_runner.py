#!/usr/bin/env python3
"""
AI Signals Test Runner

Creates timestamped test result files in test_results/ folder.
Each test run generates a new file with complete results and status.
Files are sorted by timestamp (newest first) for easy navigation.
"""

import os
import sys
import json
import time
from datetime import datetime
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_decision_engine import AIDecisionEngine
from accuracy.ai_accuracy import AIAccuracyCalculator

class AITestRunner:
    def __init__(self):
        """Initialize the test runner"""
        self.test_results_dir = "test_results"
        self.ensure_test_results_dir()
        
    def ensure_test_results_dir(self):
        """Ensure test_results directory exists"""
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)
    
    def generate_test_filename(self):
        """Generate timestamped filename for test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ai_test_{timestamp}.md"
    
    def run_ai_signals_test(self):
        """Run AI signals test and capture all results"""
        print("🧪 AI SIGNALS TEST RUNNER")
        print("=" * 50)
        
        test_start_time = datetime.now()
        test_filename = self.generate_test_filename()
        test_filepath = os.path.join(self.test_results_dir, test_filename)
        
        print(f"📁 Test results will be saved to: {test_filepath}")
        print(f"⏰ Test started at: {test_start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 50)
        
        # Initialize test results
        test_results = {
            'test_info': {
                'start_time': test_start_time.isoformat(),
                'filename': test_filename,
                'test_type': 'AI Signals Real-Time Test'
            },
            'ai_signals': {},
            'accuracy_status': {},
            'data_status': {},
            'errors': [],
            'summary': {}
        }
        
        try:
            # Step 1: Test AI Signal Generation
            print("\n🤖 STEP 1: Testing AI Signal Generation...")
            print("-" * 40)
            
            decision_engine = AIDecisionEngine()
            decisions = decision_engine.generate_signals_for_all_symbols()
            
            test_results['ai_signals'] = {
                'total_signals': len(decisions),
                'signals': decisions,
                'generation_success': True
            }
            
            print(f"✅ Generated {len(decisions)} AI signals")
            
            # Step 2: Test Accuracy Status
            print("\n📊 STEP 2: Testing Accuracy Status...")
            print("-" * 40)
            
            accuracy_calc = AIAccuracyCalculator()
            if accuracy_calc.connect_db():
                try:
                    # Get AI decisions from database
                    db_decisions = accuracy_calc.get_ai_decisions(24)
                    
                    # Get data status
                    accuracy_calc.cursor.execute("SELECT COUNT(*) FROM historical_data_1h")
                    hist_count = accuracy_calc.cursor.fetchone()[0]
                    
                    accuracy_calc.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM historical_data_1h")
                    min_ts, max_ts = accuracy_calc.cursor.fetchone()
                    
                    test_results['accuracy_status'] = {
                        'ai_decisions_in_db': len(db_decisions),
                        'historical_data_points': hist_count,
                        'data_range': {
                            'start': min_ts,
                            'end': max_ts
                        },
                        'accuracy_ready': len(db_decisions) > 0
                    }
                    
                    test_results['data_status'] = {
                        'historical_data_available': hist_count > 0,
                        'data_points': hist_count,
                        'data_range_start': min_ts,
                        'data_range_end': max_ts,
                        'ai_decisions_available': len(db_decisions) > 0
                    }
                    
                    print(f"✅ AI decisions in DB: {len(db_decisions)}")
                    print(f"✅ Historical data points: {hist_count}")
                    print(f"✅ Data range: {min_ts} to {max_ts}")
                    
                finally:
                    accuracy_calc.close_db()
            else:
                test_results['errors'].append("Failed to connect to database for accuracy check")
                print("❌ Failed to connect to database")
            
            # Step 3: Test Real-Time Accuracy (if possible)
            print("\n🎯 STEP 3: Testing Real-Time Accuracy...")
            print("-" * 40)
            
            try:
                if accuracy_calc.connect_db():
                    try:
                        decisions = accuracy_calc.get_ai_decisions(6)
                        results = accuracy_calc.calculate_accuracy(decisions, 4)
                        
                        test_results['accuracy_status']['real_time_accuracy'] = {
                            'total_decisions': results['total_decisions'],
                            'evaluated_decisions': results['evaluated_decisions'],
                            'accuracy_percentage': results['accuracy'],
                            'can_evaluate': results['evaluated_decisions'] > 0
                        }
                        
                        if results['evaluated_decisions'] > 0:
                            print(f"✅ Real-time accuracy: {results['accuracy']:.1f}%")
                        else:
                            print("⏳ No decisions can be evaluated yet (need 4+ hours future data)")
                            
                    finally:
                        accuracy_calc.close_db()
            except Exception as e:
                test_results['errors'].append(f"Accuracy calculation error: {str(e)}")
                print(f"❌ Accuracy calculation error: {e}")
            
            # Step 4: Generate Summary
            test_end_time = datetime.now()
            duration = (test_end_time - test_start_time).total_seconds()
            
            test_results['summary'] = {
                'end_time': test_end_time.isoformat(),
                'duration_seconds': duration,
                'test_success': len(test_results['errors']) == 0,
                'signals_generated': len(decisions),
                'accuracy_available': test_results['accuracy_status'].get('real_time_accuracy', {}).get('can_evaluate', False)
            }
            
            print(f"\n🎉 Test completed in {duration:.1f} seconds")
            
        except Exception as e:
            test_results['errors'].append(f"Test execution error: {str(e)}")
            test_results['summary'] = {
                'end_time': datetime.now().isoformat(),
                'test_success': False,
                'error': str(e)
            }
            print(f"❌ Test execution error: {e}")
        
        # Step 5: Save Results to File
        self.save_test_results(test_filepath, test_results)
        
        # Step 6: Update Test Index
        self.update_test_index()
        
        return test_results
    
    def save_test_results(self, filepath: str, results: dict):
        """Save test results to markdown file"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.generate_markdown_report(results))
            print(f"✅ Test results saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving test results: {e}")
    
    def generate_markdown_report(self, results: dict) -> str:
        """Generate markdown report from test results"""
        test_info = results['test_info']
        ai_signals = results['ai_signals']
        accuracy_status = results['accuracy_status']
        data_status = results['data_status']
        summary = results['summary']
        errors = results['errors']
        
        report = f"""# AI Signals Test Results

## 📅 Test Information
- **Test Date & Time:** {datetime.fromisoformat(test_info['start_time']).strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Test Type:** {test_info['test_type']}
- **Test File:** {test_info['filename']}
- **Duration:** {summary.get('duration_seconds', 0):.1f} seconds
- **Status:** {'✅ SUCCESS' if summary.get('test_success', False) else '❌ FAILED'}

---

## 🤖 AI Signal Generation Results

### Signal Summary
- **Total Signals Generated:** {ai_signals.get('total_signals', 0)}
- **Generation Status:** {'✅ SUCCESS' if ai_signals.get('generation_success', False) else '❌ FAILED'}

### Individual Signals
"""
        
        if ai_signals.get('signals'):
            report += "\n| Symbol | Action | Confidence | Current Price | Status |\n"
            report += "|--------|--------|------------|---------------|--------|\n"
            
            for symbol, decision in ai_signals['signals'].items():
                action = decision['action']
                confidence = decision['confidence']
                current_price = decision.get('current_price', 'N/A')
                
                # Status emoji
                if action in ['STRONG_BUY', 'BUY']:
                    status_emoji = "🟢"
                elif action in ['STRONG_SELL', 'SELL']:
                    status_emoji = "🔴"
                else:
                    status_emoji = "🟡"
                
                report += f"| {symbol} | {action} | {confidence:.2f} | ${current_price} | {status_emoji} |\n"
        
        report += f"""

---

## 📊 Accuracy Status

### Database Status
- **AI Decisions in DB:** {accuracy_status.get('ai_decisions_in_db', 0)}
- **Historical Data Points:** {accuracy_status.get('historical_data_points', 0)}
- **Data Range:** {accuracy_status.get('data_range', {}).get('start', 'N/A')} to {accuracy_status.get('data_range', {}).get('end', 'N/A')}
- **Accuracy Ready:** {'✅ YES' if accuracy_status.get('accuracy_ready', False) else '❌ NO'}

### Real-Time Accuracy
"""
        
        if 'real_time_accuracy' in accuracy_status:
            rt_acc = accuracy_status['real_time_accuracy']
            report += f"- **Total Decisions:** {rt_acc.get('total_decisions', 0)}\n"
            report += f"- **Evaluated Decisions:** {rt_acc.get('evaluated_decisions', 0)}\n"
            report += f"- **Accuracy Percentage:** {rt_acc.get('accuracy_percentage', 0):.1f}%\n"
            report += f"- **Can Evaluate:** {'✅ YES' if rt_acc.get('can_evaluate', False) else '❌ NO'}\n"
        else:
            report += "- **Status:** Not available\n"
        
        report += f"""

---

## 📈 Data Status

### Historical Data
- **Available:** {'✅ YES' if data_status.get('historical_data_available', False) else '❌ NO'}
- **Data Points:** {data_status.get('data_points', 0)}
- **Range Start:** {data_status.get('data_range_start', 'N/A')}
- **Range End:** {data_status.get('data_range_end', 'N/A')}

### AI Decisions
- **Available:** {'✅ YES' if data_status.get('ai_decisions_available', False) else '❌ NO'}

---

## 🚨 Errors & Issues

"""
        
        if errors:
            for i, error in enumerate(errors, 1):
                report += f"{i}. ❌ {error}\n"
        else:
            report += "✅ No errors encountered\n"
        
        report += f"""

---

## 📋 Test Summary

### Overall Results
- **Test Success:** {'✅ YES' if summary.get('test_success', False) else '❌ NO'}
- **Signals Generated:** {summary.get('signals_generated', 0)}
- **Accuracy Available:** {'✅ YES' if summary.get('accuracy_available', False) else '❌ NO'}

### Next Steps
"""
        
        if summary.get('test_success', False):
            if summary.get('accuracy_available', False):
                report += "- ✅ System is fully operational\n"
                report += "- ✅ Accuracy can be calculated\n"
                report += "- 💡 Run `python accuracy/ai_accuracy.py real` for detailed accuracy\n"
            else:
                report += "- ✅ AI signals generation working\n"
                report += "- ⏳ Wait 4+ hours for accuracy evaluation\n"
                report += "- 💡 Run `python accuracy/ai_accuracy.py real` after 4+ hours\n"
        else:
            report += "- ❌ System has issues that need to be resolved\n"
            report += "- 🔧 Check error messages above\n"
            report += "- 🔄 Re-run test after fixing issues\n"
        
        report += f"""

---

## 🔗 Quick Commands

```bash
# Generate new signals
python utils/ai_realtime.py

# Check accuracy status
python accuracy/ai_accuracy.py status

# Real-time accuracy
python accuracy/ai_accuracy.py real

# Run another test
python utils/ai_test_runner.py
```

---

*Test completed at: {datetime.fromisoformat(summary.get('end_time', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S UTC')}*
"""
        
        return report
    
    def update_test_index(self):
        """Update the test index file for easy navigation"""
        try:
            # Get all test files
            test_files = []
            for filename in os.listdir(self.test_results_dir):
                if filename.startswith('ai_test_') and filename.endswith('.md'):
                    filepath = os.path.join(self.test_results_dir, filename)
                    stat = os.stat(filepath)
                    test_files.append({
                        'filename': filename,
                        'created_time': stat.st_ctime,
                        'size': stat.st_size
                    })
            
            # Sort by creation time (newest first)
            test_files.sort(key=lambda x: x['created_time'], reverse=True)
            
            # Generate index file
            index_filepath = os.path.join(self.test_results_dir, 'README.md')
            with open(index_filepath, 'w', encoding='utf-8') as f:
                f.write("# AI Signals Test Results Index\n\n")
                f.write("## 📋 Test History (Newest First)\n\n")
                f.write("| Test File | Date & Time | Size |\n")
                f.write("|-----------|-------------|------|\n")
                
                for test_file in test_files:
                    timestamp = datetime.fromtimestamp(test_file['created_time'])
                    size_kb = test_file['size'] / 1024
                    f.write(f"| [{test_file['filename']}](./{test_file['filename']}) | {timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {size_kb:.1f} KB |\n")
                
                f.write(f"\n\n## 📊 Total Tests: {len(test_files)}\n")
                f.write(f"## 📁 Latest Test: {test_files[0]['filename'] if test_files else 'None'}\n")
                f.write(f"\n## 🚀 Run New Test\n\n```bash\npython utils/ai_test_runner.py\n```\n")
            
            print(f"✅ Test index updated: {index_filepath}")
            
        except Exception as e:
            print(f"❌ Error updating test index: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI Signals Test Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick test (signals only)')
    
    args = parser.parse_args()
    
    runner = AITestRunner()
    
    if args.quick:
        print("🚀 Running quick test...")
        # TODO: Implement quick test mode
    else:
        print("🧪 Running full AI signals test...")
    
    results = runner.run_ai_signals_test()
    
    if results['summary'].get('test_success', False):
        print("\n🎉 Test completed successfully!")
        print(f"📁 Results saved to: test_results/{results['test_info']['filename']}")
    else:
        print("\n❌ Test completed with errors!")
        print("Check the test results file for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()


