# Sample evaluation to show what results will look like
from ai_evaluation import AIEvaluation

def show_sample_results():
    evaluator = AIEvaluation()
    
    # Mock some test results to show you what the output looks like
    test_results = [
        {
            'symbol': '/ES:XCME{=h}',
            'action': 'STRONG_BUY',
            'confidence': 0.85,
            'status': 'Evaluated',
            'directional_accuracy': {'is_correct': True, 'price_change': 0.025},
            'profit_loss': {'pnl': 125.50, 'pnl_percentage': 2.5}
        },
        {
            'symbol': '/NQ:XCME{=h}',
            'action': 'BUY',
            'confidence': 0.80,
            'status': 'Evaluated',
            'directional_accuracy': {'is_correct': True, 'price_change': 0.018},
            'profit_loss': {'pnl': 89.30, 'pnl_percentage': 1.8}
        },
        {
            'symbol': '/RTY:XCME{=h}',
            'action': 'BUY',
            'confidence': 0.75,
            'status': 'Evaluated',
            'directional_accuracy': {'is_correct': False, 'price_change': -0.012},
            'profit_loss': {'pnl': -45.20, 'pnl_percentage': -1.2}
        }
    ]
    
    # Calculate stats
    stats = evaluator.calculate_overall_statistics(test_results)
    
    print('📊 SAMPLE EVALUATION RESULTS (What you\'ll see in 4+ hours):')
    print('=' * 60)
    print(f'🎯 Directional Accuracy: {stats["directional_accuracy"]}%')
    print(f'🏆 Win Rate: {stats["win_rate"]}%')
    print(f'💰 Total P&L: ${stats["total_pnl"]}')
    print(f'💪 Average Confidence: {stats["average_confidence"]}')
    print(f'✅ Profitable Trades: {stats["profitable_trades"]}')
    print(f'❌ Losing Trades: {stats["losing_trades"]}')
    print('=' * 60)
    
    if stats['directional_accuracy'] >= 70:
        print('🎉 EXCELLENT! Accuracy exceeds 70% target!')
    elif stats['directional_accuracy'] >= 60:
        print('👍 GOOD! Accuracy is above 60%')
    else:
        print('⚠️ Needs improvement - below 60%')

if __name__ == "__main__":
    show_sample_results()
