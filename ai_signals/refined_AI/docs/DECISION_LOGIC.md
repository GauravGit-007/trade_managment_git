### Refined AI Decision Logic (How a daily signal is made)

This note explains, at a high level, how the system turns data into a single daily trading decision per symbol.

---

### 1) Inputs and freshness gates

- **Historical price candles (1h)**: Close, volume and simple technicals are computed. Data must be fresh (<= ~3 hours).
- **News + sentiment**: Recent articles are scraped, mapped to instruments, and scored by FinBERT. If parent index news exists, micro contracts inherit that coverage. Data must be fresh.
- **LSTM predictions**: Short-horizon price direction/level forecast. Must be fresh.

If any critical input is stale and non‑interactive mode is used, that symbol is skipped. In interactive mode, the user can proceed with warnings.

---

### 2) Feature extraction (calculations)

- **Technical indicators (from historical):**

  - RSI (relative strength index) on closes.
  - Simple moving averages (e.g., SMA short vs long) to infer trend bias.
  - Volume ratio (latest vs recent average) to detect participation/interest.
  - Basic data hygiene: numeric coercion, NaN drops, minimum bars checks.

- **News sentiment (from FinBERT):**

  - Each article text → FinBERT → label ∈ {positive, negative, neutral} with a score.
  - Labels mapped to market signals: positive→Up, negative→Down, neutral→Flat.
  - Aggregate recent articles per symbol family (parent+micro, oil+micro) to a summary sentiment bias and average confidence.

- **Model forecast (LSTM):**
  - Next-horizon direction/level estimate (e.g., 6–12h). Extracted as predicted change or target price vs current.

---

### 3) Heuristics for signal construction

The system forms a single signal {BUY, SELL, HOLD} using consistent heuristics:

1. Start with a neutral baseline (HOLD).
2. Combine evidence:
   - If LSTM predicts meaningful up move and news bias is Up, and RSI is not overbought → tilt to BUY.
   - If LSTM predicts meaningful down move and news bias is Down, and RSI is not oversold → tilt to SELL.
   - If signals conflict (e.g., Up news, Down forecast) and technicals are neutral/mixed → HOLD.
3. Use volume/trend context:
   - Higher volume ratio or SMA alignment strengthens a BUY/SELL tilt.
   - Weak participation or mixed SMAs dampens conviction → HOLD more likely.

---

### 4) Confidence scoring (0–1)

Confidence is a blended score reflecting agreement and signal quality:

- Agreement bonus: LSTM direction agrees with news bias.
- Technical confirmation: RSI/trend context supports direction.
- Data quality: freshness and sufficient recent articles/candles.

Typical buckets used by the app:

- ~0.70–0.75: weak-to-moderate alignment, proceed with caution.
- ~0.80+: clear agreement across components and acceptable freshness.

---

### 5) Output and logging

For each symbol, the system saves a structured decision to SQLite in `Smart_AI_decisions` with fields like:

- `symbol`, `decision_timestamp`
- `signal` (BUY/SELL/HOLD)
- `confidence_score`
- `predicted_price`, `current_price` (when available)
- `reasoning` (concise natural-language rationale)
- `data_sources_used` (e.g., historical_data, news_sentiment, lstm_predictions)

This allows post‑trade accuracy evaluation once future candles arrive (e.g., 6–12 hours later or end‑of‑day).

---

### 6) Notes on instrument coverage

- News is first mapped via LLM + keyword fallbacks.
- Parent ↔ micro linking: `/ES` ↔ `/MES`, `/NQ` ↔ `/MNQ`, `/QM` ↔ `/MCL` ensure coverage continuity.
- Energy symbols use additional keyword detection; if no live articles exist, the system can rely on recent market context while avoiding hard‑coded outcomes.

---

### 7) When to check accuracy

- After a few post‑decision candles accumulate (1–3+), ideally at 6–12 hours or end‑of‑day, compare the decided direction vs realized move (direction accuracy, MAE/MAPE).
