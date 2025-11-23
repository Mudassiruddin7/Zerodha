# Zerodha Kite MCP AI Trading Bot
## MVP Implementation Checklist & Deployment Guide

---

## SECTION 1: MVP SCOPE DEFINITION

### 1.1 Minimum Viable Product (MVP) Deliverable

The MVP focuses on **single-instrument (NIFTY)**, **semi-automated execution**, and **manual override always available**.

| Component | MVP Scope | Post-MVP |
|-----------|-----------|----------|
| **Instruments** | NIFTY index only | Add NIFTY-50 stocks |
| **F&O Strategies** | Intraday long options (CE, PE) | Swing options |
| **Equity Strategies** | Not in MVP | Swing + intraday |
| **Signal Model** | Rule-based + simple classifier | Ensemble ML (XGBoost) |
| **Execution Mode** | Semi-automated (manual confirm) | Fully automated with guardrails |
| **Data Sources** | Kite WebSocket + historical REST | Add order book L2/L3 |
| **Backtesting** | Simple fee/slippage model | Walk-forward validation |
| **Monitoring** | Basic P&L dashboard | Advanced analytics + alerts |
| **Deployment** | Local VS Code + MCP | Cloud with redundancy |

### 1.2 MVP Success Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Backtest Win Rate** | ≥ 50% | Historical simulation (3 months data) |
| **Backtest Sharpe** | ≥ 0.8 | Daily returns calculation |
| **Backtest Net Profit** | ≥ 200 INR | After fees and slippage |
| **Paper Trading Duration** | 10 trading days | Continuous operation |
| **Order Fill Rate** | ≥ 95% in 60s | Live fill tracking |
| **Model Inference Time** | < 100 ms | End-to-end latency |
| **UI Responsiveness** | < 500 ms | Page load + trade confirm |

---

## SECTION 2: WEEK-BY-WEEK IMPLEMENTATION ROADMAP

### Week 0–1: Foundation & Environment Setup

**Objectives:** Set up dev environment, Zerodha API credentials, basic data loading.

**Deliverables:**

1. **Environment Setup**
   - [ ] Python 3.10+ installed
   - [ ] VS Code with Python extension
   - [ ] Create virtual environment: `python -m venv venv`
   - [ ] Activate: `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows)
   - [ ] Install base deps: `pip install pandas numpy scikit-learn xgboost asyncio aiohttp pyyaml pydantic ta-lib`

2. **Zerodha Setup**
   - [ ] Register Kite Connect app at Zerodha Developer Console
   - [ ] Obtain API key and secret
   - [ ] Store in `config/secrets.env` (Git-ignored)
   - [ ] Test OAuth flow manually (browser login)

3. **Data Infrastructure**
   - [ ] Create `/data/historical/` directory structure
   - [ ] Download 3 months NIFTY 1m candles (2025-08-01 to 2025-11-01)
   - [ ] Save as Parquet: `data/historical/symbol=NIFTY/date=2025-11-05/candles.parquet`
   - [ ] Write `DataLoader` class to read historical candles
   - [ ] Verify data integrity: check for gaps, duplicates, NaNs

4. **Configuration Framework**
   - [ ] Define `BotConfig` (Pydantic model) with sections: capital, trading_rules, f_and_o, etc.
   - [ ] Create `config/default_config.yaml` with safe defaults
   - [ ] Create `config/config.yaml` (user overrides)
   - [ ] Test config loading and validation

**Code Skeleton:**

```python
# main.py (MVP stub)
import asyncio
from config.config_loader import BotConfig
from data.data_ingester import DataLoader

async def main():
    # Load config
    config = BotConfig.from_yaml('config/config.yaml')
    
    # Load historical data
    loader = DataLoader(config)
    nifty_candles = loader.load_candles(symbol='NIFTY', start='2025-08-01', end='2025-11-01', timeframe='1m')
    
    print(f"Loaded {len(nifty_candles)} NIFTY 1m candles")
    
    # TODO: Feature engineering next week

if __name__ == '__main__':
    asyncio.run(main())
```

---

### Week 1–2: Feature Engineering & Backtest Engine

**Objectives:** Build feature pipeline and basic vectorized backtester.

**Deliverables:**

1. **Feature Engineering**
   - [ ] Implement `FeatureComputer` class:
     - [ ] EMA(12, 26), MACD, MACD signal
     - [ ] RSI(14), ADX(14)
     - [ ] Bollinger Bands, Z-score
     - [ ] ATR(14), volatility
   - [ ] Create `features_df` from historical candles
   - [ ] Validate features: no NaNs, no infinite values
   - [ ] Plot sample features (matplotlib verification)

2. **Backtesting Infrastructure**
   - [ ] Create `BacktestEngine` class with:
     - [ ] Fee/slippage model (50 INR per trade + 0.1% slippage)
     - [ ] Position tracking (entry, exit, P&L)
     - [ ] Equity curve calculation
   - [ ] Implement `run_backtest(signals_df, price_df)` → returns trade_log + metrics
   - [ ] Calculate metrics: win_rate, profit_factor, sharpe_ratio, max_drawdown

3. **Rule-Based Signal Generator (MVP Simple)**
   - [ ] `generate_signals_simple(features_df)`:
     - [ ] IF EMA_12 > EMA_26 AND RSI < 70 → CALL signal
     - [ ] IF EMA_12 < EMA_26 AND RSI > 30 → PUT signal
   - [ ] Filter by confidence (simple heuristic: 0.60–0.80)
   - [ ] Apply cool-down (48 hours between F&O trades)
   - [ ] Output signals_df with signal_type, entry_price, expected_return_pct

4. **Backtest Runner**
   - [ ] Load historical candles + option chain snapshot
   - [ ] Generate signals for entire 3-month period
   - [ ] Run backtest
   - [ ] Print metrics:
     ```
     ===== BACKTEST RESULTS (3-month historical) =====
     Total Trades: 12
     Winning Trades: 7 (58% win rate)
     Profit Factor: 1.45
     Net Profit: 450 INR (4.5% ROI)
     Sharpe Ratio: 0.92
     Max Drawdown: 8.2%
     Avg Trade Profit: 37.5 INR
     ```

**Success Gate:** Backtest must show net profit > 0 and win rate ≥ 50%.

---

### Week 2–3: Kite MCP Integration & Order Execution

**Objectives:** Connect to Zerodha Kite MCP, test live market data, enable order placement.

**Deliverables:**

1. **Kite MCP Client Wrapper**
   - [ ] Create `KiteMCPClient` class with methods:
     - [ ] `initiate_oauth_flow()` → returns login URL
     - [ ] `exchange_token(request_token)` → returns access_token
     - [ ] `get_quote(symbol)` → returns bid/ask/last
     - [ ] `get_option_chain(symbol, expiry)` → returns strikes + IV + Greeks
     - [ ] `subscribe_ticks(symbols)` → start WebSocket
     - [ ] `place_order(order_dict)` → returns order_id
     - [ ] `get_order_history(order_id)` → returns status + fills

2. **OAuth Flow in VS Code**
   - [ ] Create simple Flask server (localhost:8000) for OAuth callback
   - [ ] On startup, redirect to Kite login if no token
   - [ ] Parse callback redirect_uri; extract request_token
   - [ ] Exchange for access_token
   - [ ] Store token securely (encrypted session)
   - [ ] Test: manually login → verify token works

3. **Market Data Subscription**
   - [ ] Subscribe to NIFTY 1m candles via WebSocket
   - [ ] Update feature store every minute
   - [ ] Verify real-time data matches historical backtest (sanity check)

4. **Order Executor**
   - [ ] Create `OrderExecutor` class:
     - [ ] `construct_order(signal, position_size, user_id)` → order_dict
     - [ ] `validate_order(order)` → checks margin, lot size, price sanity
     - [ ] `place_order(order)` with retry logic (exponential backoff)
     - [ ] `track_fill(order_id)` → poll status every 5s until COMPLETE/REJECTED
     - [ ] `calculate_slippage(fill_price, expected_price)` → log for model
   - [ ] Test paper trading mode (simulated fills)

5. **Paper Trading Mode**
   - [ ] Create `PaperTradingEnvironment`:
     - [ ] Simulate order fills with realistic slippage
     - [ ] Track simulated positions and P&L
     - [ ] NOT real money; for validation only
   - [ ] Run 5 days of paper trading; validate fills realistic

**Success Gate:** Successfully place and fill test orders via Kite MCP; confirm slippage model accuracy.

---

### Week 3–4: Strategy Rules & Position Sizing

**Objectives:** Implement F&O strategy with strike selection and position sizing.

**Deliverables:**

1. **F&O Strategy: Strike Selection**
   - [ ] Implement `select_call_strike(nifty_price, iv_regime, capital_available)`:
     - [ ] Candidate strikes: ATM, ATM+100, ATM+200
     - [ ] Filter by liquidity: OI > 1000, volume > 500, spread < 2%
     - [ ] Filter by delta: 0.20–0.40 preferred (OTM)
     - [ ] Score by: OI * (1 - spread%) / premium
     - [ ] Return highest-scored strike
   - [ ] Similar for PUT strikes (ATM, ATM-100, ATM-200)
   - [ ] Test on recent option chain data

2. **Position Sizing (Risk-Based)**
   - [ ] Implement `calculate_position_size(signal, capital, risk_pct)`:
     - [ ] Max loss per trade = capital * risk_pct (default 5%)
     - [ ] For options: loss = premium_per_lot
     - [ ] num_lots = max_loss / premium_per_lot
     - [ ] Respect max capital allocation (30% for F&O)
     - [ ] Return minimum 1 lot (or skip if not affordable)
   - [ ] Test: starting capital 10k, signal premium 2000, max risk 500 → should size 1 lot

3. **Exit Rules**
   - [ ] **Take Profit:** IF current_premium >= entry_premium * 1.50 → EXIT
   - [ ] **Stop Loss:** IF current_premium <= entry_premium * 0.40 → EXIT
   - [ ] **Time Exit:** IF days_to_expiry <= 1 → EXIT
   - [ ] **EOD Exit (intraday):** IF current_time >= 3:15 PM → EXIT

4. **Cool-Down & Batching**
   - [ ] Track last F&O trade time
   - [ ] Enforce 48-hour cool-down between new F&O trades
   - [ ] If multiple signals in 30-min window, select only highest-edge one

5. **Expected Profit Gate**
   - [ ] Implement `calculate_expected_net_profit(signal, position_size, fees, slippage)`:
     - [ ] gross_pnl = (exit_price - entry_price) * position_size
     - [ ] net_pnl = gross_pnl - (entry_fee + exit_fee + slippage)
     - [ ] Apply probability discount: expected_pnl = p_win * net_pnl_win + (1-p_win) * net_pnl_loss
     - [ ] Accept trade only if expected_pnl >= 150 INR (configurable)
   - [ ] Test: E[profit] = 150 INR → ACCEPT; 100 INR → REJECT

**Success Gate:** Strike selection algorithm works; position sizes calculated correctly; expected profit gate rejects low-edge trades.

---

### Week 4–5: ML Signal Model (Classifier + Regressor)

**Objectives:** Train XGBoost classifier (p_win) and regressor (expected_return).

**Deliverables:**

1. **Labeling Strategy**
   - [ ] For historical 3-month backtest data:
     - [ ] For each signal, simulate holding period (e.g., 3 days for swing)
     - [ ] Calculate realized P&L after fees/slippage
     - [ ] Label: 1 if net_pnl > 0 else 0 (for classifier)
     - [ ] Label: net_pnl (for regressor, in INR)
   - [ ] Handle look-ahead bias: use timestamp order (no mixing future into past)

2. **Feature Selection**
   - [ ] Compute 20–30 features from features_df:
     - [ ] Momentum: EMA, MACD, RSI, ADX
     - [ ] Mean-reversion: Z-score, Bollinger position, ROC
     - [ ] Volatility: ATR, std_dev, IV rank (from options)
     - [ ] Microstructure: bid-ask spread, volume trend
   - [ ] Drop features with near-zero variance
   - [ ] Check correlation; drop highly correlated features (> 0.95)

3. **Model Training**
   - [ ] **Classifier (p_win):**
     - [ ] Split: 80% train, 20% test (time-series aware; no lookahead)
     - [ ] Algorithm: XGBClassifier
     - [ ] Hyperparams (defaults): max_depth=6, learning_rate=0.05, n_estimators=100
     - [ ] Cross-validation: 5-fold time-series split
     - [ ] Metric: accuracy, precision, recall, F1
     - [ ] Target: accuracy ≥ 70%
   - [ ] **Regressor (expected_return):**
     - [ ] Target: net_pnl_in_INR (or expected_return_pct)
     - [ ] Similar split and hyperparams
     - [ ] Metric: R², MAE, RMSE
     - [ ] Target: R² ≥ 0.50

4. **Model Serialization & Loading**
   - [ ] Save trained models to disk (joblib or pickle):
     - [ ] `models/classifier.pkl`
     - [ ] `models/regressor.pkl`
   - [ ] Load at startup; use for predictions

5. **Prediction Pipeline**
   - [ ] On each new candle, compute features
   - [ ] Call classifier + regressor
   - [ ] Output: (p_win, expected_return_pct, confidence)
   - [ ] Apply thresholds: accept if p_win ≥ 0.80 AND expected_return_pct ≥ 0.02 (example)

**Success Gate:** Classifier accuracy ≥ 70%; regressor R² ≥ 0.50; predictions pass sanity checks.

---

### Week 5–6: Risk Management & Circuit Breaker

**Objectives:** Implement risk gates, daily loss limit, margin checks, circuit breaker.

**Deliverables:**

1. **Daily Loss Limiter**
   - [ ] Track `daily_pnl` (realized + unrealized)
   - [ ] On start of each day, reset `daily_pnl = 0`
   - [ ] If `daily_pnl < -500` (5% of 10k), pause all new trades
   - [ ] User can manually override or wait until next day

2. **Margin Validator**
   - [ ] Before placing order, check:
     - [ ] `available_margin > order_margin_requirement`
     - [ ] `available_margin > min_buffer (500 INR)`
   - [ ] If insufficient, reject order with alert

3. **Concurrent Position Limiter**
   - [ ] Maintain count of open positions
   - [ ] If count ≥ max_concurrent (default 3), don't place new trades
   - [ ] Close positions as they exit

4. **Circuit Breaker**
   - [ ] Triggers for pause:
     - [ ] Daily loss > 5%
     - [ ] API latency > 500 ms (consecutive)
     - [ ] Exceptions in last minute > 5
     - [ ] Market halted (exchange circuit breaker)
   - [ ] When triggered: pause all NEW trades; hold existing
   - [ ] Manual kill-switch: user can stop bot instantly

5. **Risk Manager Class**
   - [ ] Implement `RiskManager.check_trading_allowed(account_state)` → bool
   - [ ] Return reason for pause (for logging/alerts)
   - [ ] Call before every trade decision

**Success Gate:** Risk checks prevent margin breach; daily loss limit enforced; circuit breaker can be triggered and tested.

---

### Week 6–7: Monitoring, Logging & UI

**Objectives:** Add real-time dashboards, audit logging, alert system.

**Deliverables:**

1. **Structured JSON Logging**
   - [ ] All events logged to JSON (not plain text)
   - [ ] Fields: timestamp, event_type, order_id, symbol, status, user_id, etc.
   - [ ] Daily log rotation: `/logs/trading_bot_2025-11-05.json`
   - [ ] Parseable for post-trade analysis

2. **Metrics Collection**
   - [ ] Track: P&L, drawdown, order latency, fill rate, slippage
   - [ ] Update every 60 seconds
   - [ ] Export to CSV for backtesting analysis

3. **VS Code Webview UI**
   - [ ] Create simple HTML dashboard (VS Code Webview)
   - [ ] Display:
     - [ ] Current equity & daily P&L
     - [ ] Open positions (symbol, qty, entry, current, unrealized P&L)
     - [ ] Recent trades (filled today)
     - [ ] Risk metrics (margin %, drawdown %)
   - [ ] Trade preview: show signal details → [Confirm] [Reject] buttons
   - [ ] Manual override: ability to close position manually

4. **Alert System**
   - [ ] Email alerts on:
     - [ ] Daily loss limit breached
     - [ ] Margin warning (> 80% utilized)
     - [ ] High latency detected
     - [ ] Order fill failures
   - [ ] Use simple SMTP or third-party (e.g., SendGrid API key)

5. **Trade Audit Log**
   - [ ] Exportable as CSV: signal → recommendation → user confirm → execution → fill
   - [ ] For compliance and review

**Success Gate:** Dashboard loads, metrics update in real-time, alerts sent on events.

---

### Week 7–8: Paper Trading & Deployment

**Objectives:** Paper trade for 10+ days; validate against backtest; prepare for live (optional).

**Deliverables:**

1. **Paper Trading Mode**
   - [ ] Toggle in config: `enable_paper_trading_mode: true`
   - [ ] Simulate fills with realistic slippage
   - [ ] Do NOT place real orders to exchange
   - [ ] Run for 10 trading days
   - [ ] Compare paper results vs backtest (correlation ≥ 0.80)

2. **Slippage Calibration**
   - [ ] After paper trading, recalibrate slippage model:
     - [ ] Average slippage observed vs model prediction
     - [ ] Update `estimated_slippage_pct` in config
   - [ ] Ensure model conservative (over-estimate slippage)

3. **Deployment Checklist**
   - [ ] All unit tests passing: `pytest tests/ -v`
   - [ ] Integration tests passing (full trade cycles)
   - [ ] Backtest results documented
   - [ ] Paper trading results documented (10-day equity curve)
   - [ ] Risk guardrails tested (daily loss limit, margin breach scenarios)
   - [ ] Audit log verified complete and parseable
   - [ ] Dashboard UI functional

4. **Pre-Live Safety Checks (Optional)**
   - [ ] Run with 1,000 INR real capital instead of 10k (micro-trades)
   - [ ] Limit to 1–2 trades per day manually
   - [ ] Monitor for 5 trading days
   - [ ] If >90% match to paper trading → ready for full deployment

5. **Documentation**
   - [ ] README: how to set up, run, configure, interpret results
   - [ ] ARCHITECTURE: system design (link to earlier doc)
   - [ ] TROUBLESHOOTING: common issues + fixes

**Success Gate:** Paper trading ≥ 10 days without crashes; slippage calibrated; all tests passing; documentation complete.

---

## SECTION 3: MVP IMPLEMENTATION CHECKLIST

### Phase 0: Pre-Dev (Week 0)

- [ ] Zerodha API credentials obtained (api_key, secret, request_token endpoint)
- [ ] Dev environment: Python 3.10+ virtual environment
- [ ] Git repo initialized; `.gitignore` includes `secrets.env`, `/logs/*`, `/data/*`
- [ ] Starter code skeleton (main.py, config/, data/, models/, etc.)

### Phase 1: Data & Backtest (Weeks 1–2)

**Data Layer:**
- [ ] DataLoader class: load Parquet candles
- [ ] FeatureComputer: compute EMA, RSI, MACD, ATR, Bollinger, Z-score, IV, etc.
- [ ] Features saved to `features_df` (datetime index, features as columns)

**Backtesting:**
- [ ] BacktestEngine: simulate trades with fee/slippage
- [ ] `run_backtest(signals_df, price_df)` → trade_log, equity_curve, metrics
- [ ] Metrics: win_rate, profit_factor, sharpe_ratio, max_drawdown, avg_trade_pnl
- [ ] Historical 3-month backtest passing (net profit > 0, win_rate ≥ 50%)

**Rules-Based Signals (MVP Simple):**
- [ ] `generate_signals_simple(features_df)`: IF EMA > signal THEN CALL (simplified)
- [ ] Apply confidence threshold (0.60–0.80 heuristic)
- [ ] Cool-down enforcement (48 hours between F&O trades)
- [ ] Output: signals_df with signal_type, entry_price, expected_return_pct

### Phase 2: Execution & Market Integration (Weeks 3–4)

**Kite MCP Integration:**
- [ ] KiteMCPClient: OAuth, WebSocket subscribe, REST API calls
- [ ] Test OAuth flow: manual login → token obtained
- [ ] Subscribe to NIFTY 1m ticks; verify real-time data
- [ ] Paper trading environment: simulate fills with slippage

**Order Execution:**
- [ ] OrderExecutor: construct order → validate → place → track fill
- [ ] Place test orders (paper mode) and verify fills
- [ ] Slippage tracking: log fill_price vs expected_price

**F&O Strategy Rules:**
- [ ] Strike selection: prefer liquid, delta 0.20–0.40, tight spreads
- [ ] Position sizing: risk-based (5% max loss per trade, respect lot size)
- [ ] Exit rules: TP (50% premium), SL (40% loss), time-based (<1 day expiry)
- [ ] Expected profit gate: E[net_profit] ≥ 150 INR (after fees)

### Phase 3: ML Model & Risk (Weeks 4–6)

**Signal Model:**
- [ ] Label historical data: profitable trades = 1, loss = 0
- [ ] Train XGBoost classifier (p_win) on backtest data
- [ ] Train XGBoost regressor (expected_return) on net_pnl
- [ ] Model accuracy ≥ 70%; regressor R² ≥ 0.50
- [ ] Serialized models saved: `models/classifier.pkl`, `models/regressor.pkl`

**Risk Management:**
- [ ] Daily loss limit: if daily_pnl < -500 INR → pause trades
- [ ] Margin validator: check before every order
- [ ] Concurrent position limiter: max 3 open positions
- [ ] Circuit breaker: pause on high latency, too many exceptions
- [ ] RiskManager.check_trading_allowed() implemented and tested

### Phase 4: Monitoring & UI (Weeks 6–7)

**Logging & Metrics:**
- [ ] Structured JSON logging: all events to `/logs/trading_bot_YYYY-MM-DD.json`
- [ ] Metrics: P&L, drawdown, latency, fill rate, slippage
- [ ] Daily log rotation; parseable for analysis

**UI & Alerts:**
- [ ] VS Code Webview dashboard: equity, positions, recent trades, risk metrics
- [ ] Trade preview UI: signal details → [Confirm] [Reject] buttons
- [ ] Email alerts: daily loss, margin warning, high latency, fill failures
- [ ] Audit log: exportable CSV (signal → confirm → execution → fill)

### Phase 5: Paper Trading & Deployment (Weeks 7–8)

**Paper Trading:**
- [ ] Run 10+ trading days in paper mode (no real orders)
- [ ] Track paper P&L; compare to backtest (correlation ≥ 0.80)
- [ ] Recalibrate slippage model based on paper fills
- [ ] All guardrails tested (daily loss limit, margin checks)

**Final Checks:**
- [ ] Unit tests: `pytest tests/ -v` (100% pass)
- [ ] Integration tests: full trade cycles (entry → fill → exit)
- [ ] Documentation: README, ARCHITECTURE, TROUBLESHOOTING
- [ ] Code review & cleanup: no debug prints, clean code
- [ ] Git history clean; ready for deployment

---

## SECTION 4: DEPLOYMENT & OPERATIONS

### 4.1 Local Deployment (Recommended for MVP)

**Environment:**
```bash
# Clone repo
git clone https://github.com/yourname/zerodha_kite_bot.git
cd zerodha_kite_bot

# Create venv
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install deps
pip install -r requirements.txt

# Configure
cp config/default_config.yaml config/config.yaml
# Edit config/config.yaml with your settings

# Add secrets (create config/secrets.env or environment variables)
export KITE_API_KEY="your_api_key"
export KITE_SECRET="your_secret"
```

**Run MVP:**
```bash
# Start bot
python main.py

# In VS Code: Open extension panel (bot icon)
# Click "Connect to Zerodha" → OAuth login
# Confirm connection
# Dashboard shows "Connected. Equity: ₹10,000. Daily P&L: ₹0"
# Ready to trade (paper mode by default)
```

### 4.2 Process Management (Optional: Systemd or Supervisor)

**Systemd Service** (Linux):
```ini
# /etc/systemd/system/zerodha_bot.service
[Unit]
Description=Zerodha Kite AI Trading Bot
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/zerodha_kite_bot
ExecStart=/home/trader/zerodha_kite_bot/venv/bin/python main.py
Restart=on-failure
RestartSec=10
StandardOutput=append:/var/log/zerodha_bot.log
StandardError=append:/var/log/zerodha_bot_error.log

[Install]
WantedBy=multi-user.target
```

**Enable & Run:**
```bash
sudo systemctl enable zerodha_bot
sudo systemctl start zerodha_bot
sudo systemctl status zerodha_bot
sudo journalctl -u zerodha_bot -f  # Live logs
```

### 4.3 Monitoring & Maintenance

**Daily Pre-Market Checklist (before 9:15 AM):**
- [ ] Bot process running: `ps aux | grep main.py`
- [ ] Kite MCP connection active (dashboard shows "Connected")
- [ ] Paper trading enabled (if not ready for live)
- [ ] Account margin sufficient
- [ ] No errors in log from yesterday

**Weekly Maintenance:**
- [ ] Review trade log: all fills as expected?
- [ ] Check feature statistics (distributions, outliers)
- [ ] Verify model accuracy on recent trades (< 100 trades)
- [ ] Backup logs and trade data

**Monthly Review:**
- [ ] Retrain model on recent data (last month)
- [ ] Update slippage calibration
- [ ] Review backtest performance (rolling window)
- [ ] Adjust config if needed (thresholds, capital allocation, etc.)

---

## SECTION 5: TROUBLESHOOTING & FAQ

### Q1: Bot crashes with "API latency > 500ms"
**A:** Circuit breaker activated. Check Kite MCP server status. If hosted, may have temporary outage. Restart bot or switch to local MCP server.

### Q2: Orders not filling; keeps getting REJECTED
**A:** 
- Check margin availability (may be insufficient for lot size)
- Verify order price not > 20% away from last traded price (sanity check)
- Ensure trading hours (9:15 AM–3:30 PM IST, weekdays only)
- Review order log for error details

### Q3: Model predictions seem off
**A:**
- Retrain model if accuracy dropped > 10%
- Check feature distributions (may have changed market regime)
- Backtest on recent data vs. old data (compare)
- Consider increasing confidence threshold (p_win ≥ 0.85 instead of 0.80)

### Q4: How to switch from paper to live trading?
**A:**
1. Set `enable_paper_trading_mode: false` in config
2. Set `ui.mode: FULLY_AUTOMATED` only if confident
3. Reduce capital: start with 1,000 INR instead of 10,000
4. Manual kill-switch always available
5. Monitor first 5 live days closely

### Q5: How to interpret the audit log?
**A:**
Each row in `audit.csv`:
```
timestamp, event_type, signal_id, order_id, symbol, quantity, entry_price, fill_price, 
slippage_pct, fees, net_pnl, status, user_confirm, reason
```
Filter by status = 'COMPLETE' for executed trades; status = 'REJECTED' for failed orders.

---

## SECTION 6: NEXT STEPS POST-MVP

Once MVP is stable (paper trading 30+ days, metrics consistent):

1. **Add More Instruments:** NIFTY-50 stocks (top 10 by liquidity)
2. **Add Equity Swing Strategy:** Use similar rule-based approach
3. **Ensemble Models:** Combine multiple ML models (random forest, neural nets)
4. **Walk-Forward Optimization:** Systematically optimize thresholds
5. **Cloud Deployment:** Docker + AWS/GCP for 24/7 uptime
6. **Advanced Monitoring:** Prometheus + Grafana, PagerDuty alerts
7. **Automated Retraining:** Nightly model updates with recent data
8. **Risk Scaling:** Dynamic position sizing based on recent volatility

---

## SECTION 7: SUCCESS METRICS & FINAL VALIDATION

### Backtest Phase (Historical Data)
- Net Profit after fees/slippage: **> 200 INR** ✓
- Win Rate: **≥ 50%** ✓
- Sharpe Ratio: **> 0.8** ✓
- Max Drawdown: **< 15%** ✓
- Avg Trade P&L: **> 50 INR** ✓

### Paper Trading Phase (10 Trading Days)
- **Correlation to Backtest:** ≥ 0.80 ✓
- **Fill Rate:** ≥ 95% within 60s ✓
- **Slippage Match:** Within ±0.2% of backtest model ✓
- **Drawdown:** < 10% ✓
- **No Crashes:** Uptime 99%+ ✓

### Live Phase (Optional; First Month)
- **ROI Target:** 5–10% (conservative) or 20%+ (aggressive) ✓
- **Trades per Month:** 5–15 (low, high-conviction) ✓
- **Max Drawdown:** < 15% (hard limit) ✓
- **Sharpe Ratio:** > 0.8 (rolling 30d) ✓
- **User Satisfaction:** No unexpected behaviors; manual control always works ✓

---

## CONCLUSION

This checklist and deployment guide provide a **concrete, week-by-week roadmap** to build and validate the Zerodha Kite MCP AI trading bot. The MVP focuses on simplicity, safety, and testability. Each phase has clear success criteria and exit gates to prevent deployment of untested or risky code.

**Key Principles:**
1. **Paper First:** Never test strategies on real money until validated
2. **Minimal Viable:** Start simple (one instrument, rule-based signals); add complexity incrementally
3. **Risk-First:** All risk checks before execution logic
4. **Audit Everything:** Logged for compliance and post-trade review
5. **Manual Override:** Always available; bot is assistant, not autopilot

**Go-Live Readiness Checklist:**
- [ ] Backtest validated (win rate ≥ 50%, net profit > 0)
- [ ] Paper trading 10+ days with 0.80+ correlation to backtest
- [ ] All unit + integration tests passing
- [ ] Risk guardrails tested (circuit breaker, margin, daily loss)
- [ ] Monitoring + logging fully operational
- [ ] Documentation complete + reviewed
- [ ] Slippage model calibrated
- [ ] User trained on UI, manual controls, emergency stop
- [ ] Legal/compliance reviewed (if needed)
- [ ] Kill-switch tested and accessible

**Ready to deploy and trade!**

