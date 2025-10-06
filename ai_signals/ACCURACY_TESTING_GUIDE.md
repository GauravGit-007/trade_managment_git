# AI Signals Accuracy Testing Guide

## 🎯 **COMPLETE ACCURACY TESTING SYSTEM**

This guide covers all accuracy testing tools and methods for verifying your AI signals performance.

---

## **📊 ACCURACY TESTING FILES**

### **1. Quick Status Check** - `simple_accuracy.py`

```bash
python simple_accuracy.py
```

**What it shows:**

- Number of AI decisions in database
- Data availability status
- Whether accuracy can be calculated
- Current system status

**When to use:** Quick check to see if system is ready for accuracy calculation

---

### **2. Detailed Accuracy Calculator** - `accuracy_calculator.py`

```bash
# Calculate accuracy for specific time period
python accuracy_calculator.py --hours 48

# Live accuracy (last 6 hours)
python accuracy_calculator.py --live

# Daily accuracy (last 24 hours)
python accuracy_calculator.py --daily

# Weekly accuracy (last 7 days)
python accuracy_calculator.py --weekly
```

**What it shows:**

- Real directional accuracy percentage
- Win/loss rates
- P&L tracking
- Symbol performance breakdown
- Individual decision results

**When to use:** Main accuracy calculation tool

---

### **3. Prove Accuracy is Real** - `prove_real_accuracy.py`

```bash
python prove_real_accuracy.py
```

**What it shows:**

- Real SQL queries used
- Actual mathematical calculations
- Database table structures
- Proof of authentic accuracy measurement
- Step-by-step calculation process

**When to use:** To verify accuracy calculation is authentic and not fake

---

### **4. Data Status Check** - `check_data_status.py`

```bash
python check_data_status.py
```

**What it shows:**

- Historical data range
- AI decision range
- Data gaps or issues
- Recommendations for fixes
- Data quality analysis

**When to use:** When accuracy shows "insufficient data"

---

### **5. Quick Accuracy Check** - `quick_accuracy_check.py`

```bash
python quick_accuracy_check.py
```

**What it shows:**

- Detailed timing analysis
- Oldest decision timing
- Data availability after decisions
- Evaluation readiness status

**When to use:** To understand why accuracy can't be calculated yet

---

## **🔍 ACCURACY CALCULATION LOGIC**

### **How Real Accuracy is Calculated:**

#### **Step 1: Get AI Decision from Database**

```sql
SELECT symbol, action, current_price, decision_timestamp
FROM ai_decisions
WHERE datetime(decision_timestamp) >= datetime('now', '-48 hours')
```

#### **Step 2: Get Actual Price Data After Decision**

```sql
SELECT close, timestamp FROM historical_data_1h
WHERE symbol = ?
AND datetime(timestamp) > datetime(?)
ORDER BY timestamp ASC LIMIT 4
```

#### **Step 3: Calculate Price Change**

```python
entry_price = decision['current_price']  # From AI decision
final_price = price_data[-1]['close']    # From historical data
price_change = (final_price - entry_price) / entry_price
```

#### **Step 4: Determine if AI was Correct**

```python
if action in ['STRONG_BUY', 'BUY']:
    is_correct = price_change > 0.001  # 0.1% threshold
elif action in ['STRONG_SELL', 'SELL']:
    is_correct = price_change < -0.001
elif action == 'HOLD':
    is_correct = abs(price_change) <= 0.001
```

#### **Step 5: Calculate Overall Accuracy**

```python
accuracy = (correct_predictions / total_predictions) * 100
```

### **Real Example:**

```
AI Decision: STRONG_BUY at $4250.50
4 Hours Later: Price = $4260.25
Price Change: (4260.25 - 4250.50) / 4250.50 = 0.0023 (0.23%)
Result: 0.23% > 0.1% = CORRECT! ✅
```

---

## **📈 EXPECTED ACCURACY RESULTS**

### **Sample Output:**

```
📊 AI SIGNALS ACCURACY REPORT
======================================================================
⏰ Time Period: Last 48 hours
🕐 Calculated at: 2025-10-03 12:00:00 UTC
======================================================================
📈 Total Decisions: 24
✅ Evaluated Decisions: 20
🎯 Directional Accuracy: 72.5%  ← YOUR TARGET!
💪 Average Confidence: 0.78
💰 Total P&L: $1,250
🏆 Win Rate: 68%
✅ Profitable Trades: 14
❌ Losing Trades: 6

📊 SYMBOL PERFORMANCE:
   /ES:XCME{=h}: 75% accuracy (3/4) conf: 0.82
   /NQ:XCME{=h}: 70% accuracy (7/10) conf: 0.79
   /RTY:XCME{=h}: 80% accuracy (4/5) conf: 0.85
   /QM:XNYM{=h}: 60% accuracy (3/5) conf: 0.76

🎯 PERFORMANCE ASSESSMENT:
🎉 EXCELLENT! Accuracy of 72.5% exceeds 70% target!
```

---

## **🚨 TROUBLESHOOTING ACCURACY ISSUES**

### **Issue: "No performance data available"**

**Cause:** Decisions too recent (less than 4 hours old)
**Solution:** Wait 4+ hours, then run accuracy calculator

### **Issue: "Insufficient data"**

**Cause:** No future price data after AI decisions
**Solution:**

```bash
# Update historical data
cd .. && python services/historical_data.py
# Wait for data to accumulate
cd ai_signals && python accuracy_calculator.py
```

### **Issue: "No decisions found"**

**Cause:** No AI decisions in database
**Solution:**

```bash
# Generate AI signals first
python ai_workflow.py --mode signals
```

### **Issue: Low accuracy (<60%)**

**Causes & Solutions:**

- **Insufficient data:** Wait longer for more decisions
- **Market volatility:** Normal during high volatility periods
- **Data quality:** Check data status and update if needed
- **Symbol-specific issues:** Check individual symbol performance

---

## **📋 ACCURACY TESTING WORKFLOW**

### **Daily Testing Routine:**

```bash
# 1. Check system status
python simple_accuracy.py

# 2. Calculate daily accuracy
python accuracy_calculator.py --daily

# 3. Check data quality
python check_data_status.py

# 4. Verify calculations are real
python prove_real_accuracy.py
```

### **Weekly Testing Routine:**

```bash
# 1. Calculate weekly accuracy
python accuracy_calculator.py --weekly

# 2. Generate comprehensive report
python accuracy_calculator.py --hours 168

# 3. Check for data issues
python check_data_status.py
```

### **Real-Time Monitoring:**

```bash
# Monitor live accuracy
python accuracy_calculator.py --live

# Check every hour
watch -n 3600 "python accuracy_calculator.py --live"
```

---

## **🎯 ACCURACY TARGETS**

### **Performance Targets:**

- **Directional Accuracy:** >70% (your main target)
- **Win Rate:** >60%
- **Average Confidence:** >0.7
- **Symbol Performance:** >65% per symbol

### **Success Indicators:**

- ✅ Accuracy consistently >70%
- ✅ Win rate >60%
- ✅ High confidence signals performing well
- ✅ Positive P&L over time

### **Warning Signs:**

- ⚠️ Accuracy <60%
- ⚠️ Win rate <50%
- ⚠️ Low confidence signals
- ⚠️ Negative P&L trend

---

## **📊 ACCURACY FILES SUMMARY**

| File                      | Purpose             | When to Use            |
| ------------------------- | ------------------- | ---------------------- |
| `simple_accuracy.py`      | Quick status check  | Daily monitoring       |
| `accuracy_calculator.py`  | Main accuracy tool  | Calculate performance  |
| `prove_real_accuracy.py`  | Verify authenticity | Debugging/verification |
| `check_data_status.py`    | Data quality check  | Troubleshooting        |
| `quick_accuracy_check.py` | Timing analysis     | Understanding delays   |

---

## **🚀 QUICK START ACCURACY TESTING**

```bash
# 1. Check if system is ready
python simple_accuracy.py

# 2. Calculate current accuracy
python accuracy_calculator.py --hours 48

# 3. If no data, update historical data
cd .. && python services/historical_data.py

# 4. Wait 4+ hours, then check again
cd ai_signals && python accuracy_calculator.py --hours 48
```

**Your AI signals system is designed to achieve >70% accuracy with real, authentic calculations!** 🎯
