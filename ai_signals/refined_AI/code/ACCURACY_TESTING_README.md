# 🎯 Comprehensive Accuracy Testing System

## Overview

This system provides comprehensive accuracy analysis for Refined AI trading signals with all the features you requested:

- ✅ **12-hour validity checks** with user prompts
- ✅ **Individual accuracy scores** out of 100%
- ✅ **Symbol-by-symbol analysis**
- ✅ **Automatic timestamped saving** to test_results folder
- ✅ **Interactive and non-interactive modes**

## 📁 Files Created

### Core Testing Files

- `comprehensive_accuracy_checker.py` - Main accuracy checker with all features
- `run_accuracy_test.py` - Easy-to-use launcher with interactive prompts
- `percentage_accuracy_analyzer.py` - Percentage-based accuracy analysis
- `analyze_historical_accuracy.py` - Historical accuracy analysis (ignores 12-hour rule)

### Test Results

- `test_results/` - Folder containing timestamped test results
- Files named: `accuracy_test_YYYYMMDD_HHMMSS.json`

## 🚀 How to Use

### Method 1: Easy Launcher (Recommended)

```bash
python run_accuracy_test.py
```

This will guide you through:

- Symbol selection (all or specific)
- Analysis period (days back)
- Interactive vs non-interactive mode

### Method 2: Direct Command Line

```bash
# Interactive mode (prompts for old decisions)
python comprehensive_accuracy_checker.py

# Non-interactive mode (skips old decisions automatically)
python comprehensive_accuracy_checker.py --non-interactive

# Analyze specific symbol
python comprehensive_accuracy_checker.py --symbol "/ES:XCME{=h}"

# Analyze last 3 days
python comprehensive_accuracy_checker.py --days 3
```

### Method 3: Historical Analysis (Ignores 12-hour rule)

```bash
# Analyze all decisions regardless of age
python analyze_historical_accuracy.py

# Percentage-based analysis
python percentage_accuracy_analyzer.py
```

## 📊 Features Explained

### 1. 12-Hour Validity Check

- **Interactive Mode**: Asks "Decision is X hours old (>12 hours). Continue anyway? (y/n)"
- **Non-Interactive Mode**: Automatically skips decisions older than 12 hours
- **Validity Penalty**: Decisions older than 12 hours get accuracy penalties

### 2. Individual Accuracy Scores (Out of 100%)

- **Accuracy Score**: Based on price prediction accuracy (0-100%)
- **Directional Score**: Based on BUY/SELL/HOLD correctness (0-100%)
- **Final Score**: Weighted combination of both scores
- **Validity Penalty**: Reduces score for old decisions

### 3. Symbol-by-Symbol Analysis

- **Rankings**: Symbols sorted by performance (best first)
- **Individual Metrics**: Accuracy, directional, and final scores per symbol
- **Decision Counts**: Analyzed vs skipped decisions per symbol
- **Performance Ratings**: EXCELLENT, VERY GOOD, GOOD, FAIR, NEEDS IMPROVEMENT

### 4. Automatic Saving

- **Timestamped Files**: `accuracy_test_YYYYMMDD_HHMMSS.json`
- **Complete Data**: All analysis results, metadata, and statistics
- **Newer First**: Files are sorted by timestamp (newest first)

## 📈 Sample Output

```
🎯 COMPREHENSIVE ACCURACY ANALYSIS REPORT
======================================================================
📊 Analysis Summary:
   📅 Timestamp: 2025-10-17T10:58:54
   📈 Period: Last 7 days
   🎯 Symbol Filter: All symbols
   🤖 Interactive Mode: OFF
   📊 Total Decisions Found: 68
   ✅ Valid Decisions Analyzed: 0
   ⏭️  Skipped Decisions: 68

🎯 Overall Performance:
   🎯 Accuracy Score: 97.1%
   📈 Directional Score: 22.8%
   ⭐ Final Accuracy: 74.9%
   🏆 Best Single Accuracy: 98.0%
   📉 Worst Single Accuracy: 95.8%

📋 Symbol-by-Symbol Analysis:
----------------------------------------------------------------------

1. /NQ:XCME{=h}:
   🎯 Accuracy Score: 97.4%
   📈 Directional Score: 58.7%
   ⭐ Final Score: 85.8%
   📊 Decisions: 10/10
   🏆 Best: 97.8% | 📉 Worst: 97.3%
   📊 Rating: ✅ VERY GOOD
```

## 🔧 Configuration Options

### Command Line Arguments

- `--symbol SYMBOL`: Analyze specific symbol only
- `--days N`: Analyze last N days (default: 7)
- `--non-interactive`: Skip user prompts, auto-skip old decisions

### Interactive Prompts

- **Symbol Selection**: All symbols or specific symbol
- **Analysis Period**: Number of days back to analyze
- **Mode Selection**: Interactive (prompts) or non-interactive (auto-skip)

## 📁 Test Results Structure

Each test result file contains:

```json
{
  "analysis_metadata": {
    "timestamp": "2025-10-17T10:58:54",
    "days_back": 7,
    "symbol_filter": null,
    "interactive_mode": false,
    "total_decisions_found": 68,
    "valid_decisions_analyzed": 0,
    "skipped_decisions": 68
  },
  "overall_statistics": {
    "overall_accuracy_score": 0,
    "overall_directional_score": 0,
    "overall_final_accuracy": 0,
    "best_accuracy": 0,
    "worst_accuracy": 0,
    "validity_penalty_applied": 0
  },
  "detailed_results": [...],
  "symbol_summary": {...}
}
```

## 🎯 Accuracy Scoring System

### Price Accuracy (0-100%)

- **100%**: Perfect prediction (0% MAPE)
- **90%**: 10% MAPE error
- **80%**: 20% MAPE error
- **Formula**: `max(0, 100 - MAPE)`

### Directional Accuracy (0-100%)

- **100%**: Correct BUY/SELL/HOLD prediction
- **0%**: Incorrect prediction
- **HOLD**: Within 2% price change = 100%, otherwise penalized

### Final Score (0-100%)

- **70%** weight on price accuracy
- **30%** weight on directional accuracy
- **Validity penalty** for decisions older than 12 hours

## 🚨 Troubleshooting

### No Decisions Found

- Check if AI signals have been generated
- Verify database connection
- Check symbol names (case-sensitive)

### All Decisions Skipped

- Decisions are older than 12 hours
- Use `--non-interactive` to see what would be analyzed
- Use `analyze_historical_accuracy.py` to ignore 12-hour rule

### Import Errors

- Ensure you're in the correct directory
- Check database connection
- Verify all dependencies are installed

## 📝 Best Practices

1. **Regular Testing**: Run accuracy tests daily to track performance
2. **Interactive Mode**: Use for detailed analysis of specific decisions
3. **Non-Interactive Mode**: Use for automated daily reports
4. **Historical Analysis**: Use to analyze older decisions for learning
5. **Symbol-Specific**: Focus on underperforming symbols for improvement

## 🎉 Success Indicators

- **Accuracy Score > 90%**: Excellent price prediction
- **Directional Score > 70%**: Good decision-making
- **Final Score > 80%**: Overall excellent performance
- **Low Validity Penalty**: Fresh, relevant decisions

---

**🎯 Ready to test your AI signals! Run `python run_accuracy_test.py` to get started.**
