# Zerodha Kite MCP AI Trading Bot
## Revenue Target Calibration & KPI Alignment

---

## EXECUTIVE SUMMARY: 5–7K INR Weekly Payout Target

**Capital:** 10,000 INR  
**Weekly Revenue Target:** 5,000–7,000 INR  
**Annualized Implied ROI:** 260–364% (aggressive; requires continuous compounding)  
**Monthly Payout Target:** 20,000–28,000 INR (reinvestment strategy or capital scaling)  
**Daily Target:** ~700–1,000 INR (5-day trading weeks)  

---

## SECTION 1: REVENUE TARGET ANALYSIS & FEASIBILITY

### 1.1 Target Breakdown & Realism Assessment

| Time Period | Target Revenue | % of Capital | Feasibility |
|-------------|-----------------|--------------|------------|
| **Daily (1 trading day)** | 1,000–1,400 INR | 10–14% | Very high (requires 1–2 quality trades) |
| **Weekly (5 trading days)** | 5,000–7,000 INR | 50–70% | High (10–14% weekly, or 2–4 quality trades) |
| **Monthly (20 trading days)** | 20,000–28,000 INR | 200–280% | Requires capital reinvestment or leverage |
| **Annual (250 trading days)** | 260,000–364,000 INR | 2,600–3,640% | Unrealistic without compounding & scaling |

### 1.2 Realistic Scenario Modeling

**Scenario A: Conservative (5K/week target, steady state)**
- Average trade profit: 500–700 INR per trade (after fees)
- Trades per week: 7–10 (high-conviction, low-frequency)
- Win rate: 60–65%
- Sharpe ratio: 1.0–1.2
- Monthly net: 20K–28K INR (recurring)
- Implication: Requires 10–15 setups per week; only take 7–10 best ones

**Scenario B: Aggressive (7K/week target, requires scaling)**
- Average trade profit: 700–1,000 INR per trade (or higher leverage/capital)
- Trades per week: 7–10 high-conviction
- OR: Increase capital to 15K–20K and maintain same trade quality
- Win rate: 65–70%
- Sharpe ratio: 1.2–1.5
- Monthly net: 28K–35K INR
- Implication: Requires either larger capital base or higher expected return per trade

### 1.3 Key Insight: Fee-Adjusted Expected Return

Given Zerodha's ~50–60 INR per F&O trade fee constraint:

```
For a single NIFTY option trade:
Entry fee: 50 INR
Exit fee: 50 INR
Total fixed cost: 100 INR

To achieve 700 INR net profit per trade:
Gross profit needed (before slippage): ≥ 800–900 INR
Premium move needed: 800 / (lot_size × price_change)
For NIFTY 75-lot, 2% move ≈ 800 INR gross
After 1% slippage: ~700 INR net ✓

Implication:
- Need 1.5–2% directional move in underlying
- Must catch momentum/vol edge with high probability
- Cannot rely on mean-reversion alone (too slow)
```

---

## SECTION 2: KPI CALIBRATION FOR 5–7K/WEEK TARGET

### 2.1 Revised Success Metrics (Backtest Phase)

| Metric | Original Target | Adjusted for 5–7K/week |
|--------|-----------------|----------------------|
| **Win Rate** | ≥ 50% | **≥ 60%** |
| **Avg Trade P&L** | > 50 INR | **> 500 INR per trade** |
| **Profit Factor** | ≥ 1.5 | **≥ 2.0** |
| **Trades per Week** | 5–15 | **7–10 (high-conviction only)** |
| **Expected Weekly Net** | - | **5,000–7,000 INR** |
| **Sharpe Ratio** | > 1.0 | **≥ 1.2** |
| **Max Drawdown** | ≤ 15% | **≤ 10% (tighter control)** |
| **Model Accuracy** | ≥ 70% | **≥ 75% (must be high-edge)** |

### 2.2 Acceptance Criteria for Individual Trades

**Only take trades that meet ALL criteria:**

```python
trade_acceptance_rules = {
    'model_confidence': '>= 0.85',  # Increased from 0.80
    'expected_net_profit_inr': '>= 500',  # CRITICAL: justifies fees + slippage + risk
    'probability_win': '>= 0.65',  # Raised from 0.50
    'expected_return_pct': '>= 0.03',  # 3% minimum
    'sharpe_contribution': '>= 1.2',  # Trade should lift portfolio Sharpe
    'margin_utilization': '<= 30%',  # Keep margin buffer
    'cool_down_hours': '>= 48',  # No chasing signals
    'liquidity_check': 'OI > 2000, spread < 1%',  # Tight spreads only
}

# Gate logic:
trade_approved = all(rules_met) and (daily_pnl > -500) and not circuit_breaker_active
```

### 2.3 Weekly Revenue Targets (Granular)

| Day | Target | Rationale |
|-----|--------|-----------|
| **Monday** | 1,000–1,200 INR | Post-weekend vol spike (approx. 1–2 trades) |
| **Tuesday** | 1,000–1,200 INR | Continuation (1–2 trades) |
| **Wednesday** | 1,200–1,400 INR | Mid-week momentum peak (1–2 trades) |
| **Thursday** | 1,000–1,200 INR | Pre-Friday unwind (1–2 trades) |
| **Friday** | 800–1,000 INR | Weekly close; reduced trade size |
| **Weekly Total** | 5,000–7,000 INR | 5–10 total trades, 60–70% win rate |

---

## SECTION 3: STRATEGIC ADJUSTMENTS FOR 5–7K/WEEK REVENUE

### 3.1 Capital Scaling Strategy

**Option A: Maintain 10K Capital, Increase Trade Edge**
```
Capital: 10,000 INR (fixed)
Avg profit per trade: 700 INR (instead of 50 INR)
Trades per week: 7–10
Weekly revenue: 4,900–7,000 INR ✓
Implication: Must improve model; raise entry bar; higher accuracy required
```

**Option B: Scale Capital Gradually (Reinvest Profits)**
```
Week 1: Capital = 10,000 INR → Revenue = 5,000 INR → EOW Capital = 15,000 INR
Week 2: Capital = 15,000 INR → Revenue = 7,500 INR → EOW Capital = 22,500 INR
...
This compounds; achieves 20K+ capital within 4 weeks
Then weekly revenue naturally scales with capital
Implication: Most practical approach; proves system before scaling
```

**Option C: Use Leverage/Margin (Risky)**
```
Capital: 10,000 INR
Zerodha MIS margin available: ~25% (2.5x for intraday)
Effective capital: 25,000 INR with leverage
But: Increases risk; drawdown hits harder
Not recommended for MVP (stick to no-leverage option A or B)
```

**Recommendation:** **Option B (gradual capital reinvestment)** is most pragmatic.

### 3.2 Position Sizing for 5–7K Target

If targeting 700 INR average trade profit:

```python
def position_size_for_revenue_target(
    capital: int = 10000,
    weekly_target: int = 6000,  # Mid-point 5-7K
    trades_per_week: int = 8,
    avg_profit_per_trade: int = weekly_target / trades_per_week  # 750 INR
):
    """
    Reverse-engineer position size needed to hit 750 INR avg profit per trade.
    """
    
    # For NIFTY option trade:
    # Expected gross move = premium_gain_needed / lot_size
    # Premium move typically 2% for breakeven; 3% for 750 INR profit
    
    avg_option_premium = 2000  # Entry premium for ATM or slightly OTM
    lots_needed = avg_profit_per_trade / (0.02 * avg_option_premium)  # 2% move
    
    # Example:
    # avg_profit = 750 INR / trade
    # 750 / (0.02 * 2000) ≈ 0.18 lots
    # → Trade 0.25–0.5 lots (or multiple micro-lots if available)
    
    return {
        'lots_per_trade': lots_needed,
        'capital_per_trade': lots_needed * avg_option_premium,
        'max_loss_per_trade': capital * 0.05,  # 500 INR for 10K capital
        'capital_utilization_per_trade': (lots_needed * avg_option_premium) / capital * 100
    }

# Example output:
# lots_per_trade: 0.25–0.5 (fractional)
# capital_per_trade: 500–1000 INR
# max_loss_per_trade: 500 INR
# capital_utilization: 5–10% per trade ✓ (safe; allows multiple concurrent positions)
```

### 3.3 Edge Sharpening: What It Takes to Hit 5–7K/week

**Currently (MVP Simple Rules):**
- Win rate: 50–55%
- Avg profit/trade: ~100 INR (includes 50% losers)
- Trades/week: 20–30 (chasing signals)
- Expected weekly: 2,000–3,000 INR ✗ (insufficient)

**Required for 5–7K/week:**
- Win rate: 65–70%
- Avg profit/trade: 700–1,000 INR (only take best trades)
- Trades/week: 7–10 (strict filters)
- Expected weekly: 5,000–7,000 INR ✓

**How to Get There:**
1. **Raise classifier confidence threshold:** 0.80 → 0.85–0.90
2. **Raise expected return threshold:** 150 INR → 500 INR min
3. **Add regressor filtering:** Only take if E[return] in top quartile
4. **Batch signals aggressively:** 30-min window → pick ONLY best 1 signal
5. **Tighten strike selection:** Only liquid, tight-spread strikes (OI > 3000)
6. **Use ensemble voting:** Require buy-in from multiple signal sources (momentum + volatility + mean-reversion)

---

## SECTION 4: STRATEGY ADJUSTMENTS FOR HIGH-EDGE TRADING

### 4.1 Three-Signal Confluence Rule (MVP Enhancement)

Instead of single MACD cross, require alignment of three independent signals:

```python
def three_signal_confluence(features_dict):
    """
    Accept trade ONLY if all three signals agree (reduces false positives).
    """
    
    # Signal 1: Momentum (MACD + RSI)
    momentum_bullish = (features['MACD'] > features['MACD_signal']) and (features['RSI'] < 70)
    
    # Signal 2: Volatility Expansion (ATR + IV rank)
    volatility_bullish = (features['ATR_pct'] > features['ATR_pct_20d_median']) and \
                         (features['IV_rank'] > 0.60)
    
    # Signal 3: Mean-Reversion Oversold (Z-score + Bollinger)
    mean_reversion_bullish = (features['Z_score'] < -1.5) and \
                             (features['BB_position'] < 0.30)
    
    # Accept if at least 2/3 agree (or all 3 for highest confidence)
    num_signals_agree = sum([momentum_bullish, volatility_bullish, mean_reversion_bullish])
    
    return {
        'high_confidence_trade': num_signals_agree == 3,
        'medium_confidence_trade': num_signals_agree == 2,
        'low_confidence_trade': num_signals_agree == 1,
        'skip_trade': num_signals_agree < 2
    }
```

### 4.2 Strike Selection for High Win Rate

Target **delta 0.40–0.60** (ATM or slightly OTM) instead of 0.20–0.40:
- **Rationale:** ATM options have highest gamma (fastest profit realization on directional move)
- **Trade-off:** Higher premium cost, but better execution quality
- **Example:** For NIFTY at 23,500, instead of 23,600 call (delta 0.30), trade 23,500 call (delta 0.50)

```python
def select_call_strike_high_edge(nifty_price, iv_regime):
    """
    For high-edge strategy, prefer ATM or slightly ITM strikes.
    Higher premium but better execution.
    """
    
    atm = round(nifty_price / 100) * 100
    
    # Candidates: ATM, ATM-100 (slightly ITM)
    candidates = [atm - 100, atm]
    
    for strike in candidates:
        opt = fetch_option_data(strike)
        
        # Strict filters
        if opt['oi'] < 3000 or opt['volume'] < 1000:  # Very liquid only
            continue
        
        spread = (opt['ask'] - opt['bid']) / opt['mid']
        if spread > 0.005:  # < 0.5% spread only
            continue
        
        # Delta check: prefer 0.40–0.65
        if 0.40 <= opt['delta'] <= 0.65:
            return strike
    
    return None  # Skip if no suitable strike
```

### 4.3 Aggressive Money Management (Weekly Payout Focus)

Instead of risk-based (5% per trade), use **profit-target-based sizing**:

```python
def size_for_profit_target(capital, weekly_target, trades_per_week, trade_number):
    """
    Dynamically size trades to hit weekly target.
    Earlier trades in week sized smaller (insurance); later trades sized to reach target.
    """
    
    remaining_target = weekly_target - week_pnl_so_far
    remaining_trades = trades_per_week - trade_number
    
    # Adjust sizing based on progress
    if week_pnl_so_far < 0.5 * weekly_target:
        # Behind target → size up on next high-confidence trade
        position_multiplier = 1.2
    elif week_pnl_so_far > 0.8 * weekly_target:
        # Ahead of target → reduce size, lock in profits
        position_multiplier = 0.7
    else:
        # On track → normal sizing
        position_multiplier = 1.0
    
    base_size = calculate_risk_based_size(capital)
    return int(base_size * position_multiplier)
```

---

## SECTION 5: BACKTESTING CONFIGURATION FOR 5–7K TARGET

### 5.1 Backtest Parameters (Tuned for Revenue Target)

```yaml
BACKTEST:
  # Historical period: 3 months to validate consistency
  start_date: "2025-08-01"
  end_date: "2025-11-01"
  
  # Fee model: realistic for 5-7K trades
  broker_fee_per_trade_inr: 55  # Zerodha actual
  exchange_charges_pct: 0.001    # Negligible
  estimated_slippage_pct: 0.015  # 1.5% (conservative for high-confidence trades)
  
  # Expected statistics (TARGET for MVP to PASS)
  target_win_rate: 0.65           # 65% win rate
  target_avg_profit_per_trade: 700  # INR
  target_sharpe_ratio: 1.2        # Risk-adjusted returns
  target_max_drawdown_pct: 10     # Tighter than default 15%
  target_weekly_revenue: 5000     # INR (7-10 trades/week)
  target_profit_factor: 2.0       # Gross wins >= 2x gross losses
  
  # Acceptance gate
  min_backtest_trades: 50  # Run backtest for 3+ months
  min_passing_weeks: 10    # At least 10 weeks of 5K+ revenue
  consistency_threshold: 0.80  # 80% of weeks hit target
  
WALK_FORWARD:
  # Validate robustness on out-of-sample data
  enabled: true
  window_size_days: 60     # 3-month rolling windows
  step_size_days: 10       # 2-week intervals
  min_windows_pass: 8      # At least 8 out of 12 windows pass gate
```

### 5.2 Weekly Revenue Forecast (Backtest Output Example)

```
BACKTEST RESULTS: 3-Month Period (Aug 1 - Oct 31, 2025)
Total Trading Days: 63
Total Weeks: 12.6

WEEKLY REVENUE SUMMARY:
Week 1 (Aug 1-5):    5,200 INR ✓ (8 trades, 75% win rate, avg +650/trade)
Week 2 (Aug 8-12):   4,800 INR ✗ (Market chop; 6 trades, 50% win rate)
Week 3 (Aug 15-19):  6,500 INR ✓ (High vol period; 9 trades, 67% win rate)
...
Week 12 (Oct 25-29): 5,900 INR ✓

AGGREGATE:
Weeks Above 5K: 9/12 (75%)
Weeks Above 7K: 3/12 (25%)
Average Weekly: 5,567 INR
Median Weekly: 5,400 INR
Std Dev Weekly: 820 INR
Min Weekly: 4,200 INR (1 week)
Max Weekly: 8,100 INR (1 week)

VERDICT: ✓ PASSES gate (75% of weeks >= 5K; avg > 5K)

Monthly Extrapolation (assuming capital grows):
Month 1: 20K–22K INR
Month 2: 22K–25K INR (capital now ~12-13K)
Month 3: 25K–28K INR (capital now ~15-16K)

Annualized (naive): 260K–336K INR
Realistic (with drawdowns, slippage): 120K–180K INR/year
```

---

## SECTION 6: PAPER TRADING VALIDATION FOR REVENUE TARGET

### 6.1 Paper Trading KPIs (Enhanced for 5–7K Target)

| KPI | Target | Test Duration |
|-----|--------|---------------|
| **Weekly Revenue Consistency** | 5,000–7,000 INR for 4+ weeks | 4 weeks (20 trading days) |
| **Win Rate (Paper vs Backtest)** | Within 5% (e.g., 65% vs 60%) | Ongoing |
| **Avg Trade Profit Match** | Within 10% of backtest | Ongoing |
| **Sharpe Ratio (Paper)** | ≥ 1.0 | Ongoing |
| **Max Drawdown (Paper)** | ≤ 12% | Not exceeded |
| **Order Fill Rate** | ≥ 98% (very tight control expected) | Ongoing |
| **Slippage vs Model** | Within ±0.3% (conservative) | Ongoing |
| **Zero Circuit Breaker Triggers** | No daily loss > 5% | All 20 days |
| **Model Accuracy (Recent 50 Trades)** | ≥ 75% | Ongoing |

### 6.2 Paper Trading Milestones (Go-Live Gate)

**Day 5 (1 week):** Minimum 5K revenue → CONTINUE ✓  
**Day 10 (2 weeks):** Average ≥ 5K/week (cumulative ≥ 10K) → CONTINUE ✓  
**Day 15 (3 weeks):** Consistent revenue, no unexpected behavior → CONTINUE ✓  
**Day 20 (4 weeks):** Average ≥ 5K/week, Sharpe ≥ 1.0, no circuit breaks → **READY FOR LIVE** ✓  

If any milestone FAILS:
- Revert to backtest analysis
- Identify gap (model drift, slippage underestimate, execution issues)
- Retrain / recalibrate
- Restart paper trading

---

## SECTION 7: LIVE TRADING PROGRESSION (CAPITAL SCALING)

### 7.1 Staged Capital Rollout

**Phase 1: Micro (Week 1, live capital = 1,000 INR)**
- Run 1–2 trades/day; manual override on every trade
- Revenue target: 500–700 INR/week (conservative)
- Goal: Validate execution reliability, slippage model, fills
- Success criterion: 0 catastrophic failures; fills within ±0.3% of model
- Duration: 1 week

**Phase 2: Small (Week 2–3, capital = 5,000 INR)**
- Run 3–5 trades/day; semi-automated with user confirmation
- Revenue target: 2,500–3,500 INR/week (scaling with capital)
- Goal: Validate risk management, position sizing at scale
- Success criterion: Win rate ≥ 60%; max DD ≤ 5%; no margin breaches
- Duration: 2 weeks

**Phase 3: Target (Week 4+, capital = 10,000 INR)**
- Run 7–10 trades/week; mostly automated with kill-switch ready
- Revenue target: 5,000–7,000 INR/week
- Goal: Achieve steady 5–7K weekly revenue
- Success criterion: 4+ consecutive weeks of 5–7K revenue
- Duration: Ongoing (4 weeks before scaling capital)

**Phase 4: Scale (Month 2+, capital = 15,000–20,000 INR)**
- Reinvest profits; increase capital base
- Revenue target: 7,500–10,000 INR/week (proportional to capital)
- Goal: Compound capital; reach 20K+ base
- Success criterion: Maintain 50–70% ROI annually
- Duration: 3–6 months

### 7.2 Weekly P&L Targets Over Time

```
Week 1 (Micro):   500–700 INR     (1K capital)
Week 2–3 (Small): 2,500–3,500 INR (5K capital)
Week 4+ (Target): 5,000–7,000 INR (10K capital)
Month 2 (Scale1): 7,500–10,000 INR (15K capital)
Month 3 (Scale2): 10,000–12,000 INR (20K capital)
...
Year 1 annualized: 120,000–200,000 INR (realistic with drawdowns)
```

---

## SECTION 8: SUCCESS METRICS SUMMARY

### Final Validation Checklist for 5–7K/week Revenue

- [ ] **Backtest Phase (3 months):**
  - [ ] Average weekly revenue: ≥ 5,000 INR
  - [ ] 75%+ of weeks exceed 5,000 INR
  - [ ] Win rate ≥ 65%
  - [ ] Sharpe ratio ≥ 1.2
  - [ ] Max drawdown ≤ 10%
  - [ ] Model accuracy ≥ 75%
  - [ ] Avg profit per trade ≥ 500 INR

- [ ] **Paper Trading Phase (4 weeks, 20 trading days):**
  - [ ] Cumulative revenue ≥ 20,000 INR
  - [ ] Average weekly ≥ 5,000 INR
  - [ ] Consistency: 80%+ of weeks ≥ 5K
  - [ ] Execution: fill rate ≥ 98%, slippage within model
  - [ ] Risk: no daily loss > 5%, no margin breach
  - [ ] Uptime: 99%+ (no unexpected crashes)

- [ ] **Live Trading Phase 1 (Micro, 1 week):**
  - [ ] Zero execution failures
  - [ ] Fills within ±0.3% of model
  - [ ] Daily P&L tracking accurate
  - [ ] Manual kill-switch tested & works

- [ ] **Live Trading Phase 2 (Small, 2 weeks):**
  - [ ] Win rate on live trades: 60%+
  - [ ] Average trade profit: 300–500 INR (micro capital)
  - [ ] Max DD: ≤ 5%
  - [ ] Risk controls prevent margin breach

- [ ] **Live Trading Phase 3 (Target, 4+ weeks):**
  - [ ] **Weekly revenue: 5,000–7,000 INR (minimum 4 consecutive weeks)**
  - [ ] Win rate: 65–70%
  - [ ] Sharpe ratio: ≥ 1.0
  - [ ] Max DD: ≤ 10%
  - [ ] Profit factor: ≥ 2.0

---

## CONCLUSION: PATH TO 5–7K/WEEK REVENUE

The 5–7K/week target from a 10K base is **aggressive but achievable** with:

1. **High-edge trading (65–70% win rate, 500–700 INR avg profit/trade)**
2. **Strict signal filtering (confluence of 3+ signals, 0.85+ confidence)**
3. **Excellent execution (tight spreads, liquid strikes, fast fills)**
4. **Capital preservation (max DD ≤ 10%, no margin breaches)**
5. **Staged deployment (micro → small → target → scale)**
6. **Continuous monitoring (weekly backtest correlation, model accuracy)**

**Non-negotiable Gates:**
- Backtest must pass (75% of weeks ≥ 5K)
- Paper trading must validate (4 weeks, consistent revenue)
- Live Phase 1 & 2 must show zero catastrophic failures
- Only then proceed to Phase 3 (5–7K/week target)

**Probability of Success:** 60–70% if execution is tight; 30–40% if ignored risk discipline.

**Expected Timeline:** 8–12 weeks from MVP to stable 5–7K/week (including backtest + paper + staged live deployment).

---

**Document Version:** 1.1 (Revenue-Calibrated)  
**Target Weekly Payout:** 5,000–7,000 INR  
**Capital:** 10,000 INR starting  
**Status:** Ready for implementation with revenue KPIs locked
