#!/usr/bin/env python3
"""
Comprehensive Accuracy Checker for Refined AI Signals
This script performs a complete accuracy analysis including:
- 12-hour validity checks with user prompts
- Individual accuracy scores out of 100%
- Symbol-by-symbol analysis
- Detailed reporting with timestamps
- Automatic saving to test_results folder
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from db.database import TradeDatabase

class ComprehensiveAccuracyChecker:
    def __init__(self, interactive_mode=True):
        self.db = TradeDatabase()
        self.interactive_mode = interactive_mode
        self.test_results_dir = "test_results"
        
        # Ensure test results directory exists
        if not os.path.exists(self.test_results_dir):
            os.makedirs(self.test_results_dir)
    
    def run_comprehensive_accuracy_check(self, symbol: str = None, days_back: int = 7) -> Dict:
        """
        Run comprehensive accuracy check with all features
        """
        print("🔍 COMPREHENSIVE ACCURACY CHECKER")
        print("=" * 60)
        print(f"📅 Analysis Period: Last {days_back} days")
        print(f"🤖 Interactive Mode: {'ON' if self.interactive_mode else 'OFF'}")
        print(f"📊 Symbol Filter: {symbol if symbol else 'All symbols'}")
        print("=" * 60)
        
        # Get user preference for handling old decisions
        global_choice = None
        if self.interactive_mode:
            global_choice = self.get_global_continuation_choice()
        
        # Get AI decisions
        decisions = self.get_ai_decisions(symbol, days_back)
        if not decisions:
            print("❌ No AI decisions found in the specified time range")
            return self.create_empty_result("No AI decisions found")
        
        print(f"📊 Found {len(decisions)} AI decisions to analyze")
        
        # Analyze each decision
        analysis_results = []
        valid_decisions = 0
        skipped_decisions = 0
        
        for i, decision in enumerate(decisions, 1):
            symbol = decision['symbol']
            decision_time = datetime.fromisoformat(decision['timestamp'].replace('Z', '+00:00'))
            predicted_price = decision['predicted_price']
            recommendation = decision['recommendation']
            
            print(f"\n📊 [{i}/{len(decisions)}] Analyzing {symbol} at {decision_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Check if decision is within 12-hour validity window
            hours_old = (datetime.now() - decision_time).total_seconds() / 3600
            
            if hours_old > 12:
                print(f"   ⚠️  Decision is {hours_old:.1f} hours old (>12 hours)")
                
                if self.interactive_mode:
                    # Use global choice if available
                    if global_choice == 'analyze_all':
                        print(f"   ✅ Analyzing {symbol} - global choice: analyze all")
                        # Continue with analysis
                    elif global_choice == 'skip_all':
                        print(f"   ⏭️  Skipping {symbol} - global choice: skip all")
                        analysis_results.append({
                            'symbol': symbol,
                            'decision_time': decision_time.isoformat(),
                            'status': 'SKIPPED',
                            'reason': 'Outside 12-hour criteria - global choice: skip all',
                            'hours_old': hours_old
                        })
                        skipped_decisions += 1
                        continue
                    elif global_choice == 'individual':
                        continue_analysis = self.prompt_user_continuation(symbol, hours_old)
                        if not continue_analysis:
                            print(f"   ⏭️  Skipping {symbol} - user chose to skip")
                            analysis_results.append({
                                'symbol': symbol,
                                'decision_time': decision_time.isoformat(),
                                'status': 'SKIPPED',
                                'reason': 'Outside 12-hour criteria - user chose to skip',
                                'hours_old': hours_old
                            })
                            skipped_decisions += 1
                            continue
                    else:
                        # Fallback to individual prompt
                        continue_analysis = self.prompt_user_continuation(symbol, hours_old)
                        if not continue_analysis:
                            print(f"   ⏭️  Skipping {symbol} - user chose to skip")
                            analysis_results.append({
                                'symbol': symbol,
                                'decision_time': decision_time.isoformat(),
                                'status': 'SKIPPED',
                                'reason': 'Outside 12-hour criteria - user chose to skip',
                                'hours_old': hours_old
                            })
                            skipped_decisions += 1
                            continue
                else:
                    print(f"   ⏭️  Non-interactive mode: Skipping {symbol}")
                    analysis_results.append({
                        'symbol': symbol,
                        'decision_time': decision_time.isoformat(),
                        'status': 'SKIPPED',
                        'reason': 'Outside 12-hour criteria - non-interactive mode',
                        'hours_old': hours_old
                    })
                    skipped_decisions += 1
                    continue
            
            # Get actual prices after the decision time
            actual_prices = self.get_actual_prices_after(symbol, decision_time)
            
            if not actual_prices:
                print(f"   ⚠️  No actual data available after decision time")
                analysis_results.append({
                    'symbol': symbol,
                    'decision_time': decision_time.isoformat(),
                    'status': 'NO_DATA',
                    'reason': 'No actual data available after decision time',
                    'hours_old': hours_old
                })
                continue
            
            # Calculate comprehensive accuracy metrics
            accuracy_metrics = self.calculate_comprehensive_metrics(
                predicted_price, actual_prices, recommendation, decision_time, hours_old
            )
            
            if accuracy_metrics:
                analysis_results.append({
                    'symbol': symbol,
                    'decision_time': decision_time.isoformat(),
                    'predicted_price': predicted_price,
                    'recommendation': recommendation,
                    'confidence': decision.get('confidence', 0.0),
                    'hours_old': hours_old,
                    'status': 'ANALYZED',
                    **accuracy_metrics
                })
                valid_decisions += 1
                print(f"   ✅ Accuracy Score: {accuracy_metrics['accuracy_score']:.1f}%")
            else:
                print(f"   ❌ Could not calculate accuracy metrics")
                analysis_results.append({
                    'symbol': symbol,
                    'decision_time': decision_time.isoformat(),
                    'status': 'ERROR',
                    'reason': 'Could not calculate accuracy metrics',
                    'hours_old': hours_old
                })
        
        # Calculate overall statistics
        overall_stats = self.calculate_overall_comprehensive_stats(analysis_results)
        overall_stats.update({
            'total_decisions': len(decisions),
            'valid_decisions': valid_decisions,
            'skipped_decisions': skipped_decisions,
            'analysis_timestamp': datetime.now().isoformat()
        })
        
        # Create comprehensive result
        result = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'days_back': days_back,
                'symbol_filter': symbol,
                'interactive_mode': self.interactive_mode,
                'total_decisions_found': len(decisions),
                'valid_decisions_analyzed': valid_decisions,
                'skipped_decisions': skipped_decisions
            },
            'overall_statistics': overall_stats,
            'detailed_results': analysis_results,
            'symbol_summary': self.calculate_symbol_summary(analysis_results)
        }
        
        # Save results to test_results folder
        self.save_results_to_file(result)
        
        # Print comprehensive report
        self.print_comprehensive_report(result)
        
        return result
    
    def get_global_continuation_choice(self) -> str:
        """Get user's global choice for handling old decisions"""
        print("\n⚠️  OLD DECISIONS DETECTED")
        print("Some decisions are older than 12 hours (outside validity window)")
        print("\n🤔 How would you like to handle old decisions?")
        print("   a) Analyze ALL old decisions (skip individual prompts)")
        print("   b) Skip ALL old decisions (don't analyze any)")
        print("   c) Ask for EACH decision individually")
        
        while True:
            choice = input("\nChoose (a/b/c): ").lower().strip()
            if choice in ['a', 'analyze all']:
                print("✅ Will analyze ALL old decisions")
                return 'analyze_all'
            elif choice in ['b', 'skip all']:
                print("⏭️  Will skip ALL old decisions")
                return 'skip_all'
            elif choice in ['c', 'individual']:
                print("🤔 Will ask for each decision individually")
                return 'individual'
            else:
                print("❌ Please enter 'a', 'b', or 'c'")
    
    def prompt_user_continuation(self, symbol: str, hours_old: float) -> bool:
        """Prompt user whether to continue with analysis of old decisions"""
        print(f"\n⚠️  DECISION VALIDITY WARNING")
        print(f"   Symbol: {symbol}")
        print(f"   Age: {hours_old:.1f} hours (exceeds 12-hour validity window)")
        print(f"   Recommendation: Generate new AI signal for accurate analysis")
        
        while True:
            response = input(f"\n🤔 Do you want to continue analyzing this decision anyway? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print(f"   ✅ Continuing with analysis of {symbol}")
                return True
            elif response in ['n', 'no']:
                print(f"   ⏭️  Skipping {symbol}")
                return False
            else:
                print(f"   ❌ Please enter 'y' or 'n'")
    
    def get_ai_decisions(self, symbol: str = None, days_back: int = 7) -> List[Dict]:
        """Get AI decisions from the database"""
        try:
            conn, cursor = self.db.sql_connect()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            if symbol:
                query = """
                    SELECT symbol, decision_timestamp, predicted_price, signal, confidence_score, reasoning
                    FROM Smart_AI_decisions 
                    WHERE symbol = ? AND decision_timestamp >= ? AND decision_timestamp <= ?
                    ORDER BY decision_timestamp DESC
                """
                cursor.execute(query, (symbol, start_time.isoformat(), end_time.isoformat()))
            else:
                query = """
                    SELECT symbol, decision_timestamp, predicted_price, signal, confidence_score, reasoning
                    FROM Smart_AI_decisions 
                    WHERE decision_timestamp >= ? AND decision_timestamp <= ?
                    ORDER BY decision_timestamp DESC
                """
                cursor.execute(query, (start_time.isoformat(), end_time.isoformat()))
            
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            decisions = []
            for row in rows:
                decisions.append({
                    'symbol': row[0],
                    'timestamp': row[1],
                    'predicted_price': row[2],
                    'recommendation': row[3],
                    'confidence': row[4],
                    'reason': row[5]
                })
            
            return decisions
            
        except Exception as e:
            print(f"❌ Error fetching AI decisions: {e}")
            return []
    
    def get_actual_prices_after(self, symbol: str, decision_time: datetime) -> List[Dict]:
        """Get actual prices after the decision time"""
        try:
            conn, cursor = self.db.sql_connect()
            
            query = """
                SELECT timestamp, close, high, low, open, volume
                FROM historical_data_1h 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp ASC
                LIMIT 24
            """
            cursor.execute(query, (symbol, decision_time.isoformat()))
            rows = cursor.fetchall()
            self.db.close_connection(conn)
            
            prices = []
            for row in rows:
                prices.append({
                    'timestamp': row[0],
                    'close': row[1],
                    'high': row[2],
                    'low': row[3],
                    'open': row[4],
                    'volume': row[5]
                })
            
            return prices
            
        except Exception as e:
            print(f"❌ Error fetching actual prices for {symbol}: {e}")
            return []
    
    def calculate_comprehensive_metrics(self, predicted_price: float, actual_prices: List[Dict], 
                                      recommendation: str, decision_time: datetime, hours_old: float) -> Dict:
        """Calculate comprehensive accuracy metrics for a single decision"""
        if not actual_prices:
            return None
        
        try:
            # Get the first few actual prices for comparison
            first_actual = actual_prices[0]['close']
            hours_available = len(actual_prices)
            
            # Calculate basic accuracy metrics
            price_error = abs(predicted_price - first_actual)
            mae = price_error
            mape = (price_error / first_actual) * 100 if first_actual != 0 else 0
            
            # Calculate percentage accuracy score (out of 100%)
            accuracy_score = max(0, 100 - mape)
            
            # Calculate directional accuracy
            price_change = first_actual - predicted_price
            price_change_pct = (price_change / predicted_price) * 100 if predicted_price != 0 else 0
            
            # Determine if directional prediction was correct
            if recommendation == "BUY" and price_change > 0:
                directional_correct = True
                directional_score = 100
            elif recommendation == "SELL" and price_change < 0:
                directional_correct = True
                directional_score = 100
            elif recommendation == "HOLD":
                if abs(price_change_pct) <= 2.0:
                    directional_correct = True
                    directional_score = 100
                else:
                    directional_correct = False
                    directional_score = max(0, 100 - abs(price_change_pct) * 10)
            else:
                directional_correct = False
                directional_score = 0
            
            # Calculate overall accuracy score (weighted average)
            overall_accuracy = (accuracy_score * 0.7) + (directional_score * 0.3)
            
            # Calculate validity penalty (decisions older than 12 hours get penalty)
            validity_penalty = max(0, (hours_old - 12) * 2)  # 2% penalty per hour over 12
            final_accuracy = max(0, overall_accuracy - validity_penalty)
            
            return {
                'actual_price': first_actual,
                'price_error': price_error,
                'mae': mae,
                'mape': mape,
                'accuracy_score': accuracy_score,
                'directional_correct': directional_correct,
                'directional_score': directional_score,
                'overall_accuracy': overall_accuracy,
                'validity_penalty': validity_penalty,
                'final_accuracy': final_accuracy,
                'hours_available': hours_available,
                'price_change_pct': price_change_pct
            }
            
        except Exception as e:
            print(f"❌ Error calculating comprehensive metrics: {e}")
            return None
    
    def calculate_overall_comprehensive_stats(self, analysis_results: List[Dict]) -> Dict:
        """Calculate overall comprehensive statistics"""
        valid_results = [r for r in analysis_results if r.get('status') == 'ANALYZED']
        
        if not valid_results:
            return {
                'overall_accuracy_score': 0,
                'overall_directional_score': 0,
                'overall_final_accuracy': 0,
                'best_accuracy': 0,
                'worst_accuracy': 0,
                'validity_penalty_applied': 0
            }
        
        accuracy_scores = [r['accuracy_score'] for r in valid_results]
        directional_scores = [r['directional_score'] for r in valid_results]
        final_scores = [r['final_accuracy'] for r in valid_results]
        validity_penalties = [r.get('validity_penalty', 0) for r in valid_results]
        
        return {
            'overall_accuracy_score': sum(accuracy_scores) / len(accuracy_scores),
            'overall_directional_score': sum(directional_scores) / len(directional_scores),
            'overall_final_accuracy': sum(final_scores) / len(final_scores),
            'best_accuracy': max(accuracy_scores),
            'worst_accuracy': min(accuracy_scores),
            'validity_penalty_applied': sum(validity_penalties) / len(validity_penalties)
        }
    
    def calculate_symbol_summary(self, analysis_results: List[Dict]) -> Dict:
        """Calculate symbol-by-symbol summary"""
        symbol_stats = {}
        
        for result in analysis_results:
            symbol = result['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'total_decisions': 0,
                    'analyzed_decisions': 0,
                    'skipped_decisions': 0,
                    'accuracy_scores': [],
                    'directional_scores': [],
                    'final_scores': []
                }
            
            symbol_stats[symbol]['total_decisions'] += 1
            
            if result.get('status') == 'ANALYZED':
                symbol_stats[symbol]['analyzed_decisions'] += 1
                symbol_stats[symbol]['accuracy_scores'].append(result['accuracy_score'])
                symbol_stats[symbol]['directional_scores'].append(result['directional_score'])
                symbol_stats[symbol]['final_scores'].append(result['final_accuracy'])
            elif result.get('status') == 'SKIPPED':
                symbol_stats[symbol]['skipped_decisions'] += 1
        
        # Calculate averages for each symbol
        for symbol, stats in symbol_stats.items():
            if stats['analyzed_decisions'] > 0:
                stats['avg_accuracy_score'] = sum(stats['accuracy_scores']) / len(stats['accuracy_scores'])
                stats['avg_directional_score'] = sum(stats['directional_scores']) / len(stats['directional_scores'])
                stats['avg_final_score'] = sum(stats['final_scores']) / len(stats['final_scores'])
                stats['best_accuracy'] = max(stats['accuracy_scores'])
                stats['worst_accuracy'] = min(stats['accuracy_scores'])
            else:
                stats['avg_accuracy_score'] = 0
                stats['avg_directional_score'] = 0
                stats['avg_final_score'] = 0
                stats['best_accuracy'] = 0
                stats['worst_accuracy'] = 0
        
        return symbol_stats
    
    def save_results_to_file(self, result: Dict):
        """Save results to timestamped file in test_results folder"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"accuracy_test_{timestamp}.json"
        filepath = os.path.join(self.test_results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filepath}")
    
    def print_comprehensive_report(self, result: Dict):
        """Print comprehensive accuracy report"""
        metadata = result['analysis_metadata']
        overall_stats = result['overall_statistics']
        symbol_summary = result['symbol_summary']
        
        print("\n" + "=" * 70)
        print("🎯 COMPREHENSIVE ACCURACY ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"📊 Analysis Summary:")
        print(f"   📅 Timestamp: {metadata['timestamp']}")
        print(f"   📈 Period: Last {metadata['days_back']} days")
        print(f"   🎯 Symbol Filter: {metadata['symbol_filter'] or 'All symbols'}")
        print(f"   🤖 Interactive Mode: {'ON' if metadata['interactive_mode'] else 'OFF'}")
        print(f"   📊 Total Decisions Found: {metadata['total_decisions_found']}")
        print(f"   ✅ Valid Decisions Analyzed: {metadata['valid_decisions_analyzed']}")
        print(f"   ⏭️  Skipped Decisions: {metadata['skipped_decisions']}")
        
        if overall_stats['overall_accuracy_score'] > 0:
            print(f"\n🎯 Overall Performance:")
            print(f"   🎯 Accuracy Score: {overall_stats['overall_accuracy_score']:.1f}%")
            print(f"   📈 Directional Score: {overall_stats['overall_directional_score']:.1f}%")
            print(f"   ⭐ Final Accuracy: {overall_stats['overall_final_accuracy']:.1f}%")
            print(f"   🏆 Best Single Accuracy: {overall_stats['best_accuracy']:.1f}%")
            print(f"   📉 Worst Single Accuracy: {overall_stats['worst_accuracy']:.1f}%")
            print(f"   ⚠️  Average Validity Penalty: {overall_stats['validity_penalty_applied']:.1f}%")
        
        print(f"\n📋 Symbol-by-Symbol Analysis:")
        print("-" * 70)
        
        # Sort symbols by final score (best first)
        symbol_scores = []
        for symbol, stats in symbol_summary.items():
            if stats['analyzed_decisions'] > 0:
                symbol_scores.append((symbol, stats['avg_final_score']))
        symbol_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (symbol, final_score) in enumerate(symbol_scores, 1):
            stats = symbol_summary[symbol]
            print(f"\n{i}. {symbol}:")
            print(f"   🎯 Accuracy Score: {stats['avg_accuracy_score']:.1f}%")
            print(f"   📈 Directional Score: {stats['avg_directional_score']:.1f}%")
            print(f"   ⭐ Final Score: {stats['avg_final_score']:.1f}%")
            print(f"   📊 Decisions: {stats['analyzed_decisions']}/{stats['total_decisions']}")
            print(f"   🏆 Best: {stats['best_accuracy']:.1f}% | 📉 Worst: {stats['worst_accuracy']:.1f}%")
            
            if stats['skipped_decisions'] > 0:
                print(f"   ⏭️  Skipped: {stats['skipped_decisions']}")
            
            # Performance rating
            if stats['avg_final_score'] >= 90:
                rating = "🏆 EXCELLENT"
            elif stats['avg_final_score'] >= 80:
                rating = "✅ VERY GOOD"
            elif stats['avg_final_score'] >= 70:
                rating = "👍 GOOD"
            elif stats['avg_final_score'] >= 60:
                rating = "⚠️ FAIR"
            else:
                rating = "❌ NEEDS IMPROVEMENT"
            
            print(f"   📊 Rating: {rating}")
    
    def create_empty_result(self, message: str) -> Dict:
        """Create empty result for error cases"""
        return {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'error': message
            },
            'overall_statistics': {},
            'detailed_results': [],
            'symbol_summary': {}
        }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Accuracy Checker for Refined AI Signals')
    parser.add_argument('--symbol', type=str, help='Specific symbol to analyze (optional)')
    parser.add_argument('--days', type=int, default=7, help='Number of days back to analyze (default: 7)')
    parser.add_argument('--non-interactive', action='store_true', help='Run in non-interactive mode')
    
    args = parser.parse_args()
    
    # Create checker instance
    checker = ComprehensiveAccuracyChecker(interactive_mode=not args.non_interactive)
    
    # Run comprehensive analysis
    result = checker.run_comprehensive_accuracy_check(
        symbol=args.symbol,
        days_back=args.days
    )
    
    print(f"\n✅ Comprehensive accuracy analysis completed!")
    print(f"📁 Results saved to test_results/ folder with timestamp")

if __name__ == "__main__":
    main()
