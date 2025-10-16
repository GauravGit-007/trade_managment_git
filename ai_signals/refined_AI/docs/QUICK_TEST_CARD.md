# 🧪 Quick Test Reference Card

## One-Line Tests

| Test                 | Command                                               | What It Does                               |
| -------------------- | ----------------------------------------------------- | ------------------------------------------ |
| **System Health**    | `python test_system.py`                               | Check database, data, modules              |
| **Data Freshness**   | `python demo_freshness.py`                            | Show data age for all symbols              |
| **Interactive Mode** | `python test_interactive.py`                          | Test user prompts with simulated input     |
| **Non-Interactive**  | `python run_refined_ai.py generate --non-interactive` | Generate signals, skip outdated data       |
| **Interactive**      | `python run_refined_ai.py generate`                   | Generate signals, prompt for outdated data |
| **Accuracy Check**   | `python run_refined_ai.py check`                      | Check accuracy of generated signals        |
| **Full Workflow**    | `python run_refined_ai.py all --non-interactive`      | Run everything in sequence                 |

## Expected Results

### ✅ System Test Should Show:

- Database connection successful
- All tables exist
- Data available (1000+ records)
- All modules import successfully
- **Result**: "4/4 tests passed"

### ⚠️ Freshness Demo Should Show:

- Historical data: ~170 hours old (OUTDATED)
- News data: NOT AVAILABLE
- LSTM data: ~169 hours old (OUTDATED)
- **Result**: "0 fresh symbols, 8 outdated symbols"

### 🤖 Non-Interactive Should Show:

- Data freshness warnings for each symbol
- "Non-interactive mode: Skipping [symbol] due to outdated data"
- **Result**: "0 symbols processed, 8 symbols skipped"

### 🎯 Interactive Should Show:

- Data freshness warnings
- "Do you want to continue with this outdated data? (y/n):"
- **Result**: Waits for your input

## Quick Troubleshooting

| Problem                      | Solution                              |
| ---------------------------- | ------------------------------------- |
| "No AI decisions found"      | Normal - no fresh data to process     |
| "Database connection failed" | Check database file exists            |
| "Module import failed"       | Run `pip install -r requirements.txt` |
| "Data not fresh"             | Expected - data is 7 days old         |

## Success Indicators

- ✅ All tests pass without errors
- ✅ Clear status messages displayed
- ✅ Output files created (`daily_signals.json`)
- ✅ Database records saved
- ✅ Appropriate warnings shown for outdated data

## Test Sequence (Recommended)

1. `python test_system.py` - Verify system health
2. `python demo_freshness.py` - Check data status
3. `python run_refined_ai.py generate --non-interactive` - Test automated mode
4. `python run_refined_ai.py check` - Verify accuracy checker

**If all pass**: System is working correctly! 🎉
