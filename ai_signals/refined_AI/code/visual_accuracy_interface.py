#!/usr/bin/env python3
"""
Visual Accuracy Results Interface
Creates a beautiful visual display of accuracy test results
"""

import os
import json
import sys
from datetime import datetime
from typing import Dict, List, Any

class VisualAccuracyInterface:
    def __init__(self, test_results_dir="test_results"):
        self.test_results_dir = test_results_dir
        self.available_tests = self.get_available_tests()
    
    def get_available_tests(self) -> List[Dict]:
        """Get list of available test result files"""
        if not os.path.exists(self.test_results_dir):
            return []
        
        tests = []
        for filename in os.listdir(self.test_results_dir):
            if filename.startswith("accuracy_test_") and filename.endswith(".json"):
                filepath = os.path.join(self.test_results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    tests.append({
                        'filename': filename,
                        'filepath': filepath,
                        'timestamp': data.get('analysis_metadata', {}).get('timestamp', 'Unknown'),
                        'total_decisions': data.get('analysis_metadata', {}).get('total_decisions_found', 0),
                        'valid_decisions': data.get('analysis_metadata', {}).get('valid_decisions_analyzed', 0),
                        'skipped_decisions': data.get('analysis_metadata', {}).get('skipped_decisions', 0)
                    })
                except Exception as e:
                    print(f"⚠️  Error reading {filename}: {e}")
        
        # Sort by timestamp (newest first)
        tests.sort(key=lambda x: x['timestamp'], reverse=True)
        return tests
    
    def display_test_selection_menu(self):
        """Display menu to select which test to view"""
        if not self.available_tests:
            print("❌ No test results found in test_results/ folder")
            return None
        
        print("🎯 VISUAL ACCURACY RESULTS INTERFACE")
        print("=" * 60)
        print("📊 Available Test Results:")
        print("-" * 60)
        
        for i, test in enumerate(self.available_tests, 1):
            timestamp = test['timestamp']
            if timestamp != 'Unknown':
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = "Unknown"
            
            print(f"{i:2d}. {formatted_time}")
            print(f"    📊 Decisions: {test['valid_decisions']}/{test['total_decisions']} analyzed")
            print(f"    ⏭️  Skipped: {test['skipped_decisions']}")
            print(f"    📁 File: {test['filename']}")
            print()
        
        while True:
            try:
                choice = input(f"Select test to view (1-{len(self.available_tests)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_tests):
                    return self.available_tests[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 1 and {len(self.available_tests)}")
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit")
    
    def load_test_data(self, test_info: Dict) -> Dict:
        """Load test data from JSON file"""
        try:
            with open(test_info['filepath'], 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return {}
    
    def display_visual_results(self, test_data: Dict):
        """Display beautiful visual results"""
        metadata = test_data.get('analysis_metadata', {})
        overall_stats = test_data.get('overall_statistics', {})
        symbol_summary = test_data.get('symbol_summary', {})
        detailed_results = test_data.get('detailed_results', [])
        
        # Header
        print("\n" + "=" * 80)
        print("🎯 VISUAL ACCURACY ANALYSIS RESULTS")
        print("=" * 80)
        
        # Analysis Info
        print(f"📅 Analysis Time: {metadata.get('timestamp', 'Unknown')}")
        print(f"📈 Period: Last {metadata.get('days_back', 'Unknown')} days")
        print(f"🎯 Symbol Filter: {metadata.get('symbol_filter', 'All symbols')}")
        print(f"🤖 Mode: {'Interactive' if metadata.get('interactive_mode', False) else 'Non-interactive'}")
        print()
        
        # Summary Stats
        print("📊 ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"📊 Total Decisions Found: {metadata.get('total_decisions_found', 0)}")
        print(f"✅ Valid Decisions Analyzed: {metadata.get('valid_decisions_analyzed', 0)}")
        print(f"⏭️  Skipped Decisions: {metadata.get('skipped_decisions', 0)}")
        print()
        
        # Overall Performance
        if overall_stats.get('overall_accuracy_score', 0) > 0:
            print("🎯 OVERALL PERFORMANCE")
            print("-" * 40)
            print(f"🎯 Accuracy Score: {overall_stats.get('overall_accuracy_score', 0):.1f}%")
            print(f"📈 Directional Score: {overall_stats.get('overall_directional_score', 0):.1f}%")
            print(f"⭐ Final Accuracy: {overall_stats.get('overall_final_accuracy', 0):.1f}%")
            print(f"🏆 Best Single Accuracy: {overall_stats.get('best_accuracy', 0):.1f}%")
            print(f"📉 Worst Single Accuracy: {overall_stats.get('worst_accuracy', 0):.1f}%")
            if overall_stats.get('validity_penalty_applied', 0) > 0:
                print(f"⚠️  Average Validity Penalty: {overall_stats.get('validity_penalty_applied', 0):.1f}%")
            print()
        
        # Symbol Performance Table
        if symbol_summary:
            print("📋 SYMBOL PERFORMANCE RANKINGS")
            print("-" * 80)
            print(f"{'Rank':<4} {'Symbol':<15} {'Accuracy':<8} {'Directional':<10} {'Final':<8} {'Decisions':<10} {'Rating':<15}")
            print("-" * 80)
            
            # Sort symbols by final score
            symbol_scores = []
            for symbol, stats in symbol_summary.items():
                if stats.get('analyzed_decisions', 0) > 0:
                    symbol_scores.append((symbol, stats.get('avg_final_score', 0)))
            symbol_scores.sort(key=lambda x: x[1], reverse=True)
            
            for i, (symbol, final_score) in enumerate(symbol_scores, 1):
                stats = symbol_summary[symbol]
                accuracy = stats.get('avg_accuracy_score', 0)
                directional = stats.get('avg_directional_score', 0)
                final = stats.get('avg_final_score', 0)
                decisions = f"{stats.get('analyzed_decisions', 0)}/{stats.get('total_decisions', 0)}"
                
                # Performance rating
                if final >= 90:
                    rating = "🏆 EXCELLENT"
                elif final >= 80:
                    rating = "✅ VERY GOOD"
                elif final >= 70:
                    rating = "👍 GOOD"
                elif final >= 60:
                    rating = "⚠️ FAIR"
                else:
                    rating = "❌ NEEDS IMPROVEMENT"
                
                print(f"{i:<4} {symbol:<15} {accuracy:<7.1f}% {directional:<9.1f}% {final:<7.1f}% {decisions:<10} {rating:<15}")
            
            print()
        
        # Recent High-Performance Decisions
        if detailed_results:
            print("📈 RECENT HIGH-PERFORMANCE DECISIONS")
            print("-" * 60)
            
            # Get analyzed decisions and sort by accuracy
            analyzed_decisions = [r for r in detailed_results if r.get('status') == 'ANALYZED']
            if analyzed_decisions:
                top_decisions = sorted(analyzed_decisions, key=lambda x: x.get('accuracy_score', 0), reverse=True)[:5]
                
                for i, decision in enumerate(top_decisions, 1):
                    symbol = decision.get('symbol', 'Unknown')
                    decision_time = decision.get('decision_time', 'Unknown')
                    recommendation = decision.get('recommendation', 'Unknown')
                    accuracy = decision.get('accuracy_score', 0)
                    predicted_price = decision.get('predicted_price', 0)
                    actual_price = decision.get('actual_price', 0)
                    
                    # Format timestamp
                    try:
                        dt = datetime.fromisoformat(decision_time.replace('Z', '+00:00'))
                        formatted_time = dt.strftime('%m-%d %H:%M')
                    except:
                        formatted_time = decision_time
                    
                    print(f"{i}. {symbol} ({formatted_time}): {recommendation}")
                    print(f"   🎯 Accuracy: {accuracy:.1f}% | Price: {predicted_price:.2f} → {actual_price:.2f}")
                    print()
            else:
                print("No analyzed decisions available (all were skipped)")
                print()
        
        # Performance Distribution
        if symbol_summary:
            print("📊 PERFORMANCE DISTRIBUTION")
            print("-" * 40)
            
            excellent = sum(1 for stats in symbol_summary.values() if stats.get('avg_final_score', 0) >= 90)
            very_good = sum(1 for stats in symbol_summary.values() if 80 <= stats.get('avg_final_score', 0) < 90)
            good = sum(1 for stats in symbol_summary.values() if 70 <= stats.get('avg_final_score', 0) < 80)
            fair = sum(1 for stats in symbol_summary.values() if 60 <= stats.get('avg_final_score', 0) < 70)
            needs_improvement = sum(1 for stats in symbol_summary.values() if stats.get('avg_final_score', 0) < 60)
            
            total_symbols = len(symbol_summary)
            
            print(f"🏆 EXCELLENT (90%+): {excellent}/{total_symbols} symbols")
            print(f"✅ VERY GOOD (80-89%): {very_good}/{total_symbols} symbols")
            print(f"👍 GOOD (70-79%): {good}/{total_symbols} symbols")
            print(f"⚠️ FAIR (60-69%): {fair}/{total_symbols} symbols")
            print(f"❌ NEEDS IMPROVEMENT (<60%): {needs_improvement}/{total_symbols} symbols")
            print()
        
        # Footer
        print("=" * 80)
        print("✅ Visual accuracy analysis complete!")
        print("=" * 80)
    
    def run_visual_interface(self):
        """Run the visual interface"""
        print("🎯 VISUAL ACCURACY RESULTS INTERFACE")
        print("=" * 60)
        
        if not self.available_tests:
            print("❌ No test results found in test_results/ folder")
            print("💡 Run an accuracy test first:")
            print("   python comprehensive_accuracy_checker.py --non-interactive")
            return
        
        while True:
            test_info = self.display_test_selection_menu()
            if test_info is None:
                break
            
            print(f"\n🔄 Loading test results...")
            test_data = self.load_test_data(test_info)
            
            if not test_data:
                print("❌ Failed to load test data")
                continue
            
            self.display_visual_results(test_data)
            
            # Ask if user wants to view another test
            while True:
                choice = input("\n🤔 View another test? (y/n): ").lower().strip()
                if choice in ['y', 'yes']:
                    break
                elif choice in ['n', 'no']:
                    print("👋 Goodbye!")
                    return
                else:
                    print("❌ Please enter 'y' or 'n'")

def main():
    interface = VisualAccuracyInterface()
    interface.run_visual_interface()

if __name__ == "__main__":
    main()
