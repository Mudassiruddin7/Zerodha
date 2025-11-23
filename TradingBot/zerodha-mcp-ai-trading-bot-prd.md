# Zerodha Kite MCP AI Trading Bot
## Comprehensive Product Requirements Document & Implementation Plan

---

## EXECUTIVE SUMMARY

**Project:** AI-driven algorithmic trading system integrated with Zerodha Kite MCP for VS Code environment.

**Starting Capital:** 10,000 INR  
**Primary KPI:** 100% ROI (target, not guaranteed)  
**Fee Budget:** 50–60 INR per F&O trade  
**Strategy Scope:** Long options (BUY CE/PE) in F&O + cash equity swing/intraday trades  
**Trade Philosophy:** High-confidence signals with minimal trade count; fee-aware execution  

**Success Definition:**
- Net profit after all fees/slippage > 0
- Trades per month ≤ 6–12 (configurable, low churn)
- Sharpe ratio > 1.0 (backtest)
- Max drawdown ≤ 15% (configurable)
- Execution latency < 100 ms for order placement
- Reproducible, auditable signal generation

---

## 1. PRODUCT VISION & OBJECTIVES

### 1.1 Vision
Build a **semi-automated, fee-conscious, AI-driven trading bot** that operates within strict risk guardrails and executes only high-confidence trades that justify their operational cost. The system prioritizes **capital preservation**, **low trade volume**, and **explainability** over flashy high-frequency strategies.

### 1.2 Ranked Objectives

| Rank | Objective | Metric |
|------|-----------|--------|
| **1** | Maximize net return after fees while minimizing trade count | Net P&L ÷ Trades per month |
| **2** | Keep drawdown acceptable; protect capital | Max DD ≤ 15%; Sortino ratio > 1.0 |
| **3** | Reproducible, auditable signals with manual override capability | Feature importance + audit log |
| **4** | Robust live/paper trading via MCP with clear monitoring | Backtest → Paper → Live validation |

### 1.3 Non-Goals
- High-frequency scalping (incompatible with fee constraint)
- Option selling or synthetic shorts (risk/margin intensive)
- Fully autonomous trading without user oversight (semi-automated + manual confirm default)
- Complex derivatives beyond options (keep scope manageable)

---

## 2. CONSTRAINTS & REQUIREMENTS

### 2.1 Operational Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| **Starting Capital** | 10,000 INR | Fixed budget; low leverage tolerance |
| **F&O Fee per Trade** | 50–60 INR | User-provided; must be justified by trade edge |
| **Allowed F&O Positions** | BUY CE only, BUY PE only | No option selling; reduce complexity & margin risk |
| **Cash Equity Trades** | Buy & sell (long only) | Intraday MIS + swing CNC/NRML allowed |
| **NIFTY Lot Size** | 75 | Standard NSE lot; margin & sizing impact |
| **Min Confidence Threshold** | p ≥ 0.80 (classifier) | Trade only if model ≥ 80% win probability |
| **Min Expected Net Profit** | ≥ 150 INR per trade (configurable) | Must exceed 2× baseline fee (50–60 INR) |
| **Cool-down Between F&O Trades** | 48 hours (swing); intraday exempt | Prevent fee churn on correlated signals |
| **Max Concurrent Open Positions** | 3 (configurable) | Operational simplicity & risk control |
| **Capital at Risk per Trade** | ≤ 5% (configurable) | Position size risk limit |
| **Daily Loss Limit** | 5–7% (configurable) | Stop trading if daily loss breached |
| **Max Drawdown (Backtest Target)** | 15% (configurable) | Capital preservation guardrail |

### 2.2 Technical Constraints

| Constraint | Specification |
|-----------|---------------|
| **Platform** | Zerodha Kite MCP (OAuth + REST API) |
| **IDE** | VS Code with MCP server integration |
| **Language** | Python 3.10+ |
| **Order Types** | LIMIT (default), SL-M (emergency), BO/CO (bracket) where supported |
| **Data Latency** | ≤ 1 second for 1m candles; ≤ 100 ms for tick data (where available) |
| **Order Placement Latency** | < 100 ms target; alert if > 500 ms |
| **Uptime Target** | 99.0% during market hours |
| **Storage** | Parquet (historical) + time-series DB for real-time (Timescale/InfluxDB optional) |

### 2.3 Regulatory & Compliance

- Respect Zerodha API rate limits (1000 req/min typical)
- No violating margin requirements (check before order placement)
- Maintain audit trail: signal → recommendation → user confirm → execution
- Mandatory disclaimers: "No guaranteed returns; trading involves risk"
- MCP OAuth scopes: read (market data) + write (trading) with explicit user consent
- Comply with NSE/BSE circuit breaker rules; auto-pause on market halts

---

## 3. SYSTEM ARCHITECTURE

### 3.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         VS Code Environment                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              UI / Control Layer                          │   │
│  │ • Trade ticket preview & confirm                         │   │
│  │ • Manual override panel                                  │   │
│  │ • Real-time P&L dashboard                               │   │
│  │ • Audit log viewer                                       │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                             │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │         Orchestration & State Management                 │   │
│  │ • Signal aggregator                                      │   │
│  │ • Position tracker                                       │   │
│  │ • Order queue manager                                    │   │
│  │ • Risk circuit breaker                                   │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                             │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │       Strategy & Execution Engine Layer                  │   │
│  │ ┌──────────────────────────────────────────────────────┐ │   │
│  │ │ Strategy Module (F&O, Equity, Rules)                │ │   │
│  │ │ • Strike selector (delta, IV, liquidity)            │ │   │
│  │ │ • Position sizer (risk-based)                       │ │   │
│  │ │ • Exit rules (TP, SL, time-based)                   │ │   │
│  │ │ • Expected-return calculator (fee-aware)            │ │   │
│  │ └──────────────────────────────────────────────────────┘ │   │
│  │ ┌──────────────────────────────────────────────────────┐ │   │
│  │ │ Signal Model Layer                                   │ │   │
│  │ │ • Feature engineering (momentum, microstructure)     │ │   │
│  │ │ • Classifier (p(win)) + regressor (E[return])        │ │   │
│  │ │ • Threshold application & filtering                  │ │   │
│  │ └──────────────────────────────────────────────────────┘ │   │
│  │ ┌──────────────────────────────────────────────────────┐ │   │
│  │ │ Execution Module                                     │ │   │
│  │ │ • Order construction & validation                    │ │   │
│  │ │ • MCP API calls (place, modify, cancel)              │ │   │
│  │ │ • Retry logic & reconciliation                       │ │   │
│  │ │ • Slippage tracking                                  │ │   │
│  │ └──────────────────────────────────────────────────────┘ │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                             │
│  ┌──────────────────▼───────────────────────────────────────┐   │
│  │              Data & Feature Store Layer                  │   │
│  │ • Parquet historical storage (candles, option chains)    │   │
│  │ • Real-time cache (current tick, option chain snapshot)  │   │
│  │ • Feature store (computed indicators, aggregate stats)   │   │
│  │ • Corporate events & calendar                            │   │
│  └──────────────────┬───────────────────────────────────────┘   │
│                     │                                             │
└─────────────────────┼─────────────────────────────────────────────┘
                      │
     ┌────────────────▼─────────────────┐
     │   Zerodha Kite MCP Client        │
     │ • OAuth & token management       │
     │ • WebSocket for tick/quote data  │
     │ • REST for order operations      │
     │ • PnL & position aggregation     │
     └────────────────┬─────────────────┘
                      │
     ┌────────────────▼─────────────────┐
     │   Zerodha Kite Backend           │
     │ • Market data (NSE/BSE)          │
     │ • Order execution                │
     │ • Account & holdings             │
     │ • Option chain data              │
     └──────────────────────────────────┘
```

### 3.2 Component List & Responsibilities

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Data Ingester** | Fetch historical + real-time candles, option chains, tick data | asyncio + aiohttp, Kite MCP WebSocket |
| **Feature Engine** | Compute momentum, mean-reversion, volatility, microstructure features | pandas, ta-lib (optional), numpy |
| **Signal Model** | Classifier + regressor for p(win) and E[return] | XGBoost / CatBoost + scikit-learn |
| **Strategy Router** | Route signals to F&O or equity module; apply filters | Python class-based design |
| **F&O Strategy** | Strike selection, position sizing, exit rules for options | Custom ruleset + model output |
| **Equity Strategy** | Momentum/mean-reversion for stocks; swing vs. intraday rules | Similar ruleset as F&O |
| **Position Sizer** | Risk-based sizing; respect lot sizes and capital limits | Custom optimizer (Kelly fraction variant) |
| **Execution Engine** | Construct orders, call MCP APIs, reconcile fills | asyncio + persistent queue |
| **Risk Manager** | Monitor drawdown, margin, concurrent positions; circuit break | Threshold-based state machine |
| **Backtest Engine** | Simulate trading with fee/slippage models; walk-forward validation | Custom vectorized or Backtrader-lite |
| **Monitoring & Logging** | Real-time P&L, latency, exceptions; persistent audit log | structured logging (JSON) + Prometheus (optional) |
| **UI / Dashboard** | Trade preview, manual override, monitoring | VS Code Webview or lightweight Flask/React |
| **Config Manager** | Load/validate YAML config; expose hyperparameters | pydantic + YAML |

---

## 4. DATA SPECIFICATION

### 4.1 Data Sources & Requirements

#### A. Candle Data (Historical & Real-time)

| Timeframe | Retention | Update Frequency | Use Case |
|-----------|-----------|------------------|----------|
| **1m** | 6 months | Per minute | Feature engineering, intraday exits |
| **5m** | 2 years | Per 5m | Swing entry confirmation |
| **15m** | 2 years | Per 15m | Medium-term trend |
| **1d** | 5+ years | End of day | Longer-term context, ML training |

**Fields per candle:** open, high, low, close, volume, dividend/split adjusted flag  
**Instruments:** NIFTY index + top 20 NIFTY-50 stocks (configurable watchlist)  
**Storage:** Parquet format, partitioned by symbol and date

#### B. Option Chain Data

| Field | Update Frequency | Precision |
|-------|------------------|-----------|
| Strike price, Expiry | Per trading day | Standard NSE expiry format |
| Bid/Ask (price, qty) | Per tick | L1 or L2 (L2 preferred for liquidity) |
| Last traded price, OI, Volume | Per tick | Real-time |
| Implied Volatility (IV) | Per tick | 0.01% granularity |
| Greeks (delta, gamma, vega, theta) | Per second (or computed) | 0.01 precision |

**Storage:** Time-series DB (InfluxDB/Timescale) or Parquet snapshots (every 5m)  
**Fetch Strategy:** Subscribe via Kite WebSocket; snapshot every 5m for backtest

#### C. Tick Data & Order Book (L2/L3)

| Field | Frequency | Use Case |
|-------|-----------|----------|
| Best bid/ask prices & sizes | Per tick | Slippage modeling, liquidity checks |
| Depth (5-level L2 or deeper L3) | Per tick | Order-book imbalance features |
| Trade tick (price, qty, side) | Per tick | Microstructure signals |

**Storage:** Streaming cache (Redis) + archival parquet (daily)  
**Latency:** < 100 ms for feature computation

#### D. Corporate Events & Calendar

| Event Type | Data Point | Delivery |
|-----------|-----------|----------|
| Earnings | Symbol, date, EPS, guidance | Via corporate actions feed or manual CSV |
| Dividends | Ex-date, record date, amount | From NSE website or Kite API |
| Market holidays | Date, description | Hardcoded calendar + NSE updates |
| Expiry dates | Weekly/monthly NIFTY expiry | NSE derivatives calendar |
| Index rebalancing | Date, new composition | NIFTY index changes (quarterly) |

### 4.2 Data Quality & Validation

- **Missing values:** Forward-fill for candles (up to 1 hour); skip if longer gap
- **Outliers:** Flag bid-ask spreads > 5% of midprice; filter ticks with 0 volume
- **Reconciliation:** Compare Kite tick data with NSE official close every 3:30 PM (market close); alert if > 0.2% mismatch
- **Freshness:** If real-time data > 5 min stale, pause new trade signals; trigger alert

---

## 5. FEATURE ENGINEERING & SIGNAL DESIGN

### 5.1 Feature Categories

#### A. Momentum Features

```
EMA_12 = exponential moving average (price, period=12, timeframe=1m)
EMA_26 = exponential moving average (price, period=26, timeframe=1m)
MACD = EMA_12 - EMA_26
MACD_signal = EMA(MACD, period=9)
RSI = relative strength index (price, period=14)
ADX = average directional index (period=14)
```

**Lookback:** 100 candles minimum per timeframe

#### B. Mean-Reversion Features

```
Z_score = (price - SMA_20) / std_dev(price, period=20)
BB_position = (price - lower_band) / (upper_band - lower_band)
  where bands = SMA ± (k * std_dev), k=2
ROC = rate of change (price, period=20)
```

#### C. Volatility Features

```
ATR = average true range (period=14)
Std_dev = standard deviation (close, period=20)
HV_realized = historical volatility over 20-day window
IV = implied volatility (from option chain)
IV_rank = (IV - IV_min_252) / (IV_max_252 - IV_min_252)
IV_skew = (IV_call_OTM - IV_ATM) / IV_ATM
```

#### D. Microstructure Features (Order-Book Based)

```
OB_imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
Depth_ratio = qty_at_5bp_bid / qty_at_5bp_ask
Trade_flow = sum(buy_vol) - sum(sell_vol) over 5m window
Large_trade = count(trades with qty > 0.5 * typical_qty)
Bid_ask_spread_pct = (ask - bid) / midprice
```

#### E. Option-Specific Features

```
IV_vs_realized = IV - HV_realized (volatility expectations gap)
Delta_band = delta of selected strike (0.20–0.40 preferred)
Time_value = option_price - intrinsic_value
Theta_decay = theta * days_to_expiry (time decay impact)
Vega_exposure = vega * 1pp change in IV
OI_trend = % change in open interest over 5 trading days
Volume_OI_ratio = volume / OI (liquidity proxy)
```

### 5.2 Signal Model Architecture

#### A. Labeling Strategy (For Model Training)

**For Options:**
```python
# Definition: trade is profitable if net P&L > 0 after fees
holding_period = 3 days (configurable)
entry_price = bid price at signal time
entry_fee = 50 INR (broker fee)
entry_slippage = estimated from order-book depth

# Simulation: close at market (exit_price) after holding_period
exit_price = market ask price at holding_period end
exit_fee = 50 INR
exit_slippage = estimated 0.1% (conservative)

net_pnl = (exit_price - entry_price) * lot_size - entry_fee - exit_fee - slippage_cost

label = 1 if net_pnl > 0 else 0
expected_return = net_pnl / capital_allocated_for_trade
```

**For Equities:**
```python
holding_period = 5 days (for swing trades)
label = 1 if (exit_price - entry_price) * qty > 2 * transaction_cost else 0
expected_return_pct = (exit_price - entry_price) / entry_price
```

#### B. Model Ensemble

**Primary Classifier (P(Win)):**
- Algorithm: XGBoost or CatBoost
- Features: Top 15–20 from feature set above
- Train/test split: 80/20 with time-series aware split
- Hyperparameters (defaults): max_depth=6, learning_rate=0.05, n_estimators=100
- Cross-validation: 5-fold time-series with no look-ahead
- Threshold: p ≥ 0.80 for trade acceptance

**Secondary Regressor (E[Return]):**
- Algorithm: XGBoost regressor
- Target: net_pnl_in_INR (multioutput for multiple holding periods)
- Ensures expected profit in INR > 150 INR before trade acceptance

**Rule-Based Pre-Filter (Always Applied):**
- Liquidity check: bid-ask spread < 2% of midprice, OI > 1000 contracts, volume > 500
- Delta band: 0.20 ≤ delta ≤ 0.40 for OTM preference (or 0.35–0.65 for ATM)
- IV filter: IV > IV_20day_percentile_30 (prefer high IV for long vol) OR IV < percentile_70 (mean reversion mode)
- Margin check: required margin < available margin × 0.80 (safety buffer)
- Circuit breaker: if daily loss > threshold, skip all new signals

### 5.3 Signal Scoring & Thresholding

```
score = w1 * p_win(features) + w2 * normalized_expected_return + w3 * liquidity_score

where:
  w1 = 0.5 (classifier probability weight)
  w2 = 0.3 (expected return weight)
  w3 = 0.2 (liquidity weight)
  
trade_accepted = True if:
  AND score > threshold (default 0.75)
  AND p_win >= 0.80
  AND expected_net_profit_INR >= 150
  AND all pre-filters pass
  AND cool-down period satisfied
```

### 5.4 Model Retraining Schedule

- **Backtest model:** trained once on 3-year historical data
- **Paper trading:** retrain weekly (Friday EOD) using last 1 year of data
- **Live trading:** retrain every 2 weeks; monitor model drift (track prediction accuracy on live data)
- **Retraining trigger:** if classification accuracy drops > 10% on recent 100 trades, force retrain

---

## 6. STRATEGY SPECIFICATIONS

### 6.1 F&O Strategy: Long Options (BUY CE / BUY PE)

#### A. Entry Rules

**Intraday (MIS) Entry:**
```python
if MACD > signal_line and RSI < 70:
    signal_type = "CALL"  # Bullish
    
if MACD < signal_line and RSI > 30:
    signal_type = "PUT"   # Bearish
    
# Additional gate: must pass ML classifier
if classifier_prob >= 0.80 and expected_net_profit_INR >= 150:
    entry_approved = True
```

**Swing Entry (CNC for expiry selection + NRML for hold):**
```python
# Require multiple timeframe alignment
if EMA_12 > EMA_26 on 15m AND EMA_12 > EMA_26 on 1h AND ADX > 25:
    signal_type = "CALL"
    swing_entry = True
```

#### B. Strike Selection Algorithm

```python
def select_strike(underlying_price, signal_type, iv_regime, capital_available):
    """
    Prefer liquid, OTM-to-ATM strikes with favorable delta.
    """
    
    # 1. Determine strike offset from ATM
    atm_strike = round(underlying_price / 100) * 100
    
    if signal_type == "CALL":
        candidate_strikes = [atm_strike, atm_strike + 100, atm_strike + 200]
    else:  # PUT
        candidate_strikes = [atm_strike, atm_strike - 100, atm_strike - 200]
    
    # 2. Filter by liquidity and delta constraints
    valid_strikes = []
    for strike in candidate_strikes:
        option_data = fetch_option_chain(symbol="NIFTY", strike=strike, expiry=nearest_expiry)
        
        # Liquidity check
        if option_data.oi < 1000 or option_data.volume < 500:
            continue
        
        # Bid-ask spread check
        spread_pct = (option_data.ask - option_data.bid) / option_data.mid_price
        if spread_pct > 0.02:
            continue
        
        # Delta check: prefer 0.20–0.40 for directional (OTM)
        if not (0.20 <= abs(option_data.delta) <= 0.40):
            if not (0.35 <= abs(option_data.delta) <= 0.65):  # Allow ATM as fallback
                continue
        
        # IV filter
        if iv_regime == "high_iv":
            iv_percentile = (option_data.iv - iv_min_252) / (iv_max_252 - iv_min_252)
            if iv_percentile < 0.60:
                continue
        
        valid_strikes.append({
            'strike': strike,
            'premium': option_data.last_price,
            'delta': option_data.delta,
            'iv': option_data.iv,
            'oi': option_data.oi,
            'spread_pct': spread_pct
        })
    
    # 3. Rank by premium affordability + liquidity + delta
    best_strike = sorted(
        valid_strikes,
        key=lambda x: (x['oi'] * (1 - x['spread_pct']), -x['premium'])
    )[0]
    
    return best_strike['strike']
```

#### C. Position Sizing

```python
def calculate_option_position_size(
    capital=10000,
    premium_per_contract=2000,  # Example: 2000 INR per NIFTY call
    signal_confidence=0.85,
    max_capital_allocation_pct=30,  # Max 30% of capital in options
    lot_size=75  # NIFTY
):
    """
    Size position to maximize expected utility while respecting capital allocation limits.
    """
    
    # Capital allocation constraint
    max_capital_for_options = capital * max_capital_allocation_pct / 100
    
    # Risk-based sizing
    max_loss_per_trade = capital * 0.05  # Max 5% at risk per trade
    
    # Expected loss = premium * lot_size (worst case: full premium loss)
    cost_per_lot = premium_per_contract
    num_lots = min(
        max_loss_per_trade / cost_per_lot,
        max_capital_for_options / cost_per_lot,
        1  # Start with single lot for MVP
    )
    
    # Round down to integer lots
    num_lots = int(num_lots)
    
    if num_lots < 1:
        return 0  # Insufficient capital; skip trade
    
    return num_lots
```

#### D. Exit Rules

**Take Profit (TP):**
```python
if current_option_price >= entry_premium * 1.50:  # 50% gain on premium
    exit_signal = "TAKE_PROFIT"
    exit_price = market_bid_price
elif current_option_price >= entry_premium + 100:  # 100 INR absolute gain
    exit_signal = "TAKE_PROFIT"
    exit_price = market_bid_price
```

**Stop Loss (SL):**
```python
if current_option_price <= entry_premium * 0.40:  # Lost 60% of premium
    exit_signal = "STOP_LOSS"
    exit_price = market_bid_price
elif current_option_price <= entry_premium - 100:  # 100 INR absolute loss
    exit_signal = "STOP_LOSS"
    exit_price = market_bid_price
```

**Time-Based Exit:**
```python
days_to_expiry = (option_expiry_date - today).days

if days_to_expiry == 1:  # Last day before expiry
    exit_signal = "TIME_DECAY"
    exit_price = market_bid_price
    reason = "Avoid theta bleed at expiry"

if signal_type == "INTRADAY" and current_time >= 15:15:  # 3:15 PM
    exit_signal = "EOD_EXIT"
    exit_price = market_bid_price
    reason = "Close intraday positions before market close"
```

### 6.2 Cash Equity Strategy: Swing & Intraday

#### A. Entry Rules

**Swing Entry (Hold 3–7 days):**
```python
if price_crosses_above_EMA_20 and RSI > 50 and volume > avg_volume_20d:
    signal_type = "BUY"
    order_type = "CNC"  # Delivery position
    expected_hold_days = 5

if price_crosses_below_EMA_20 and RSI < 50:
    signal_type = "SELL"  # Exit existing long
```

**Intraday Entry (Hold 30min–4hrs):**
```python
if MACD > signal_line on 5m AND RSI(5m) < 70:
    signal_type = "BUY_INTRADAY"
    order_type = "MIS"  # Intraday square-off

if MACD < signal_line on 5m AND RSI(5m) > 30:
    signal_type = "SELL_INTRADAY"
    order_type = "MIS"
```

#### B. Position Sizing (Equity)

```python
def calculate_equity_position_size(
    capital=10000,
    stock_price=1000,  # Entry price
    stop_loss_price=950,  # SL 50 INR away
    risk_per_trade_pct=5,
    max_position_size_pct=10,  # Max 10% of capital per stock
    typical_qty_per_trade=10  # Adjust based on capital
):
    """
    Position size for equity trades: risk-based.
    """
    
    risk_per_trade_INR = capital * risk_per_trade_pct / 100
    risk_per_share = stock_price - stop_loss_price
    
    num_shares = risk_per_trade_INR / risk_per_share
    
    # Capital allocation constraint
    max_notional = capital * max_position_size_pct / 100
    max_shares_by_capital = max_notional / stock_price
    
    num_shares = min(num_shares, max_shares_by_capital)
    
    return int(num_shares)
```

#### C. Exit Rules (Equity)

**Swing Exit:**
```python
TP = entry_price * (1 + target_return_pct)  # e.g., 5% gain
SL = entry_price * (1 - stop_loss_pct)      # e.g., 3% loss
days_held = (today - entry_date).days

if current_price >= TP:
    exit_signal = "TAKE_PROFIT"
    exit_reason = f"Reached target +{target_return_pct*100}%"

if current_price <= SL:
    exit_signal = "STOP_LOSS"
    exit_reason = f"Hit stop loss -{stop_loss_pct*100}%"

if days_held >= 7:
    exit_signal = "TIME_EXIT"
    exit_reason = "Max swing hold period reached"
```

### 6.3 Cool-Down & Trade Batching Rules

```python
def check_trade_acceptance(signal, recent_trades, config):
    """
    Apply cool-down and batching logic to prevent fee churn.
    """
    
    # Cool-down: no new F&O trades within 48 hours
    if signal['instrument_type'] == 'F&O':
        last_f_and_o_trade = get_last_trade_time(instrument_type='F&O', trades=recent_trades)
        time_since_last_trade = now - last_f_and_o_trade
        
        if time_since_last_trade < timedelta(hours=48):
            return False, reason="F&O cool-down period active"
    
    # Batching: if multiple correlated signals in 30-min window, take highest edge
    correlated_signals = find_correlated_signals(signal, recent_trades, window=30)
    
    if len(correlated_signals) > 1:
        # Sort by expected net profit; accept only the top one
        best_signal = sorted(correlated_signals, key=lambda x: x['expected_return'], reverse=True)[0]
        if signal != best_signal:
            return False, reason="Lower-confidence signal batched out; top signal selected"
    
    return True, reason="Trade accepted"
```

---

## 7. MONEY MANAGEMENT & POSITION SIZING

### 7.1 Risk-Based Position Sizing Framework

```python
class PositionSizer:
    
    def __init__(self, capital=10000, max_risk_per_trade=0.05, max_concurrent=3):
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade  # 5% of capital at risk per trade
        self.max_concurrent = max_concurrent
        self.leverage = 1.0  # No leverage in MVP
    
    def calculate_size(self, signal, current_prices, margin_requirements):
        """
        Calculate position size given signal, current market state, and constraints.
        
        Returns: {num_contracts, notional_exposure, margin_required, expected_loss_at_sl}
        """
        
        # 1. Risk-based size
        max_loss_per_trade = self.capital * self.max_risk_per_trade
        
        if signal['type'] == 'OPTION':
            entry_premium = current_prices['option_premium']
            num_lots = max_loss_per_trade / entry_premium
            notional = entry_premium * num_lots * signal['lot_size']
            
        else:  # EQUITY
            entry_price = current_prices['stock_price']
            sl_price = signal['stop_loss']
            risk_per_share = entry_price - sl_price
            num_shares = max_loss_per_trade / risk_per_share
            notional = entry_price * num_shares
        
        # 2. Margin check
        margin_req = margin_requirements.get(signal['symbol'], notional * 0.20)
        if margin_req > self.capital * 0.80:
            return None  # Insufficient margin
        
        # 3. Capital allocation constraint (options max 30%, equity max 10% per stock)
        if signal['type'] == 'OPTION':
            max_alloc = self.capital * 0.30
        else:
            max_alloc = self.capital * 0.10
        
        if notional > max_alloc:
            num_units = int(max_alloc / (entry_premium if signal['type'] == 'OPTION' else entry_price))
        
        # 4. Concurrent position limit
        current_open_positions = len(get_open_positions())
        if current_open_positions >= self.max_concurrent:
            return None
        
        return {
            'num_units': num_units,
            'notional_exposure': notional,
            'margin_required': margin_req,
            'expected_loss_at_sl': max_loss_per_trade
        }
```

### 7.2 Expected Return & Fee-Aware Filtering

```python
def calculate_expected_net_profit(signal, model_output, current_market_data, config):
    """
    Estimate net profit in INR after all fees and slippage.
    This is THE critical filter: trade only if E[net_profit] >> fees.
    """
    
    symbol = signal['symbol']
    signal_type = signal['type']  # 'OPTION' or 'EQUITY'
    p_win = model_output['classifier_prob']  # e.g., 0.82
    expected_return_pct = model_output['expected_return_pct']  # e.g., 0.05 (5%)
    
    # 1. Position size
    position_size = calculate_position_size(signal, config)
    if position_size is None:
        return {'net_profit_expected_inr': 0, 'reason': 'Position size calc failed'}
    
    entry_price = current_market_data['entry_price']
    
    # 2. Entry cost
    entry_fee = config['broker_fee_per_trade']  # 50–60 INR
    entry_slippage_inr = estimate_slippage(symbol, signal_type, config)
    
    # 3. Expected exit price (probabilistic)
    expected_move = entry_price * expected_return_pct
    
    # 4. Exit cost
    exit_fee = config['broker_fee_per_trade']
    exit_slippage_inr = entry_slippage_inr  # Symmetric assumption
    
    # 5. Gross P&L (at expected return)
    if signal_type == 'OPTION':
        gross_pnl = (expected_move) * position_size['num_units'] * config['lot_size']
    else:
        gross_pnl = expected_move * position_size['num_units']
    
    # 6. Net P&L after costs
    total_cost = entry_fee + exit_fee + (entry_slippage_inr + exit_slippage_inr)
    net_pnl_expected = gross_pnl - total_cost
    
    # 7. Apply probability discount
    net_pnl_risk_adjusted = net_pnl_expected * p_win + (-position_size['expected_loss_at_sl']) * (1 - p_win)
    
    # 8. Acceptance threshold
    min_net_profit = config['min_expected_net_profit_per_trade']  # e.g., 150 INR
    
    trade_accepted = net_pnl_risk_adjusted >= min_net_profit
    
    return {
        'net_profit_expected_inr': net_pnl_risk_adjusted,
        'gross_pnl': gross_pnl,
        'total_cost': total_cost,
        'trade_accepted': trade_accepted,
        'reason': f"E[net_profit]={net_pnl_risk_adjusted:.0f} INR, {'ACCEPT' if trade_accepted else 'REJECT'}"
    }

def estimate_slippage(symbol, signal_type, config):
    """
    Estimate slippage in INR based on bid-ask spread and historical fills.
    """
    if signal_type == 'OPTION':
        bid_ask_spread_pct = 0.01  # ~1% typical for liquid options
        midprice = get_current_midprice(symbol)
        slippage_inr = midprice * bid_ask_spread_pct * 0.5  # Half-spread as slippage estimate
    else:
        bid_ask_spread_pct = 0.005  # ~0.5% for equities
        price = get_current_price(symbol)
        slippage_inr = price * bid_ask_spread_pct * 0.5
    
    return slippage_inr
```

---

## 8. EXECUTION LOGIC & KITE MCP INTEGRATION

### 8.1 Order Construction & Placement

```python
class OrderExecutor:
    
    def __init__(self, kite_mcp_client, config):
        self.client = kite_mcp_client
        self.config = config
        self.order_queue = PersistentQueue('order_queue.db')
        self.audit_log = AuditLogger('audit.json')
    
    def construct_order(self, signal, position_size, approval_user_id):
        """
        Build order dict for MCP API call.
        """
        
        order = {
            'variety': 'regular',  # 'regular', 'bo', 'co', etc.
            'order_type': 'LIMIT',  # Default to LIMIT for control
            'tradingsymbol': signal['symbol'],
            'exchange': 'NFO' if signal['instrument_type'] == 'F&O' else 'NSE',
            'transaction_type': 'BUY',
            'quantity': position_size['num_units'],
            'price': signal['entry_price'],  # LIMIT order price
            'product': 'MIS' if signal['holding_period'] == 'intraday' else 'NRML',
            'validity': 'DAY',
            'tag': f"signal_{signal['signal_id']}_user_{approval_user_id}"
        }
        
        # Add stop-loss for intraday (bracket order if supported)
        if signal['holding_period'] == 'intraday' and self.config.get('use_bracket_orders'):
            order['variety'] = 'bo'
            order['parent_order_id'] = None
            order['trigger_price'] = signal['stop_loss_price']
            order['trailing_stop_loss'] = self.config.get('trailing_sl_pct', 2)
        
        return order
    
    def place_order(self, order, user_confirm=True):
        """
        Place order via Kite MCP with optional user confirmation.
        """
        
        if user_confirm:
            # Semi-automated: show preview, wait for user confirm
            confirmation = self._prompt_user_confirmation(order)
            if not confirmation:
                self.audit_log.log({
                    'event': 'ORDER_REJECTED_BY_USER',
                    'order': order,
                    'timestamp': datetime.now()
                })
                return {'success': False, 'reason': 'User rejected'}
        
        # Validate order
        validation = self._validate_order(order)
        if not validation['valid']:
            self.audit_log.log({
                'event': 'ORDER_VALIDATION_FAILED',
                'order': order,
                'error': validation['error']
            })
            return {'success': False, 'reason': validation['error']}
        
        # Place order
        try:
            response = self.client.place_order(
                variety=order['variety'],
                exchange=order['exchange'],
                tradingsymbol=order['tradingsymbol'],
                transaction_type=order['transaction_type'],
                quantity=order['quantity'],
                price=order['price'],
                order_type=order['order_type'],
                product=order['product'],
                validity=order['validity'],
                tag=order['tag']
            )
            
            order_id = response['order_id']
            
            self.audit_log.log({
                'event': 'ORDER_PLACED',
                'order_id': order_id,
                'order': order,
                'timestamp': datetime.now()
            })
            
            return {'success': True, 'order_id': order_id}
        
        except Exception as e:
            self.audit_log.log({
                'event': 'ORDER_PLACEMENT_FAILED',
                'order': order,
                'error': str(e)
            })
            return {'success': False, 'reason': str(e)}
    
    def _validate_order(self, order):
        """
        Pre-flight checks before sending to exchange.
        """
        
        checks = []
        
        # 1. Margin check
        available_margin = self.client.get_account_margin()['available']
        estimated_margin = self._estimate_margin(order)
        checks.append(available_margin >= estimated_margin)
        
        # 2. Price sanity check (not > 20% away from last price)
        last_price = self.client.get_quote(order['exchange'], order['tradingsymbol'])['last_price']
        price_deviation = abs(order['price'] - last_price) / last_price
        checks.append(price_deviation < 0.20)
        
        # 3. Lot size validation
        if order['exchange'] == 'NFO':
            lot_size = self.config['lot_size'][order['tradingsymbol']]
            checks.append(order['quantity'] % lot_size == 0)
        
        # 4. Concurrent position limit
        open_positions = self.client.get_holdings()
        checks.append(len(open_positions) < self.config['max_concurrent_positions'])
        
        all_passed = all(checks)
        return {
            'valid': all_passed,
            'error': 'Validation failed' if not all_passed else None
        }
    
    def _estimate_margin(self, order):
        """
        Rough margin estimate for pre-flight check.
        """
        
        if order['product'] == 'MIS':
            return order['quantity'] * order['price'] * 0.25  # ~25% for equity intraday
        else:
            return order['quantity'] * order['price'] * 0.20  # ~20% for derivatives
```

### 8.2 Fill Reconciliation & Slippage Tracking

```python
class FillReconciler:
    
    def __init__(self, client, config):
        self.client = client
        self.config = config
        self.pending_orders = {}
    
    def track_order_fill(self, order_id, expected_price, expected_qty):
        """
        Monitor order status; reconcile fill price vs expected.
        """
        
        max_wait_seconds = self.config.get('order_wait_timeout', 300)
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            order_status = self.client.get_order_history(order_id)
            
            if order_status['status'] == 'COMPLETE':
                # Order filled
                filled_qty = order_status['filled_quantity']
                fill_price = order_status['average_price']
                
                slippage = fill_price - expected_price
                slippage_pct = slippage / expected_price
                
                self._log_fill(order_id, expected_price, fill_price, slippage_pct)
                
                return {
                    'filled': True,
                    'filled_qty': filled_qty,
                    'fill_price': fill_price,
                    'slippage': slippage,
                    'slippage_pct': slippage_pct
                }
            
            elif order_status['status'] == 'REJECTED' or order_status['status'] == 'CANCELLED':
                return {'filled': False, 'reason': order_status['status']}
            
            time.sleep(5)  # Poll every 5 seconds
        
        # Timeout: cancel order and retry or escalate
        self.client.cancel_order(order_id)
        return {'filled': False, 'reason': 'TIMEOUT'}
    
    def _log_fill(self, order_id, expected_price, fill_price, slippage_pct):
        """
        Log to slippage tracking DB for model calibration.
        """
        
        log_entry = {
            'order_id': order_id,
            'expected_price': expected_price,
            'fill_price': fill_price,
            'slippage_pct': slippage_pct,
            'timestamp': datetime.now()
        }
        
        # Append to parquet or time-series DB
        self._persist_fill_log(log_entry)
        
        # Alert if slippage > threshold
        if slippage_pct > self.config.get('max_slippage_pct_alert', 0.005):
            logger.warning(f"High slippage on {order_id}: {slippage_pct*100:.2f}%")
```

### 8.3 Order Retry & Fail-Safe Logic

```python
def retry_order_placement(order, max_retries=3, backoff_base=2):
    """
    Exponential backoff retry for failed orders.
    """
    
    for attempt in range(max_retries):
        try:
            result = executor.place_order(order, user_confirm=False)
            if result['success']:
                return result
        except Exception as e:
            wait_time = backoff_base ** attempt
            logger.warning(f"Order placement attempt {attempt+1} failed. Retry in {wait_time}s: {e}")
            time.sleep(wait_time)
    
    # Final attempt failed
    logger.error(f"Order placement failed after {max_retries} retries: {order}")
    alert_user("Order placement failed. Manual intervention required.")
    return {'success': False, 'reason': 'Max retries exceeded'}

def fail_safe_pause_trading():
    """
    Circuit breaker: pause automation on critical failures.
    """
    
    # Triggers
    if api_latency_ms > 500:
        logger.critical("API latency > 500ms. Pausing trading.")
        toggle_trading(enabled=False)
    
    if daily_loss_pct > config['daily_loss_limit']:
        logger.critical("Daily loss limit breached. Pausing trading.")
        toggle_trading(enabled=False)
    
    if not is_market_open():
        logger.info("Market closed. Pausing trading.")
        toggle_trading(enabled=False)
    
    if exception_count_in_last_minute > 5:
        logger.critical("High exception rate. Pausing trading.")
        toggle_trading(enabled=False)
```

---

## 9. BACKTESTING FRAMEWORK

### 9.1 Backtest Engine Specification

```python
class BacktestEngine:
    """
    Vectorized backtester with fee/slippage model and walk-forward validation.
    """
    
    def __init__(self, config):
        self.config = config
        self.trade_log = []
        self.equity_curve = []
    
    def run_backtest(self, signals_df, price_data, option_data, start_date, end_date):
        """
        Execute backtest over historical data.
        
        Inputs:
          signals_df: DataFrame with datetime, symbol, signal_type, p_win, expected_return
          price_data: Dict[symbol] = OHLCV candles
          option_data: Dict[symbol][expiry][strike] = option chain snapshots
          start_date, end_date: date range for backtest
        
        Returns:
          BacktestResult with trade log, equity curve, metrics
        """
        
        starting_capital = self.config['starting_capital']
        current_capital = starting_capital
        portfolio = {}  # {position_id: {symbol, qty, entry_price, entry_date, ...}}
        
        for index, row in signals_df.iterrows():
            if row['datetime'] < start_date or row['datetime'] > end_date:
                continue
            
            signal = row
            current_price = price_data[signal['symbol']].loc[signal['datetime'], 'close']
            
            # Check acceptance thresholds
            if signal['p_win'] < self.config['min_confidence'] or \
               signal['expected_return_pct'] * current_price < self.config['min_expected_profit_inr']:
                continue
            
            # Calculate position size
            position_size = self._calc_position_size(signal, current_capital)
            if position_size is None:
                continue
            
            # Entry cost (commission + slippage)
            entry_slippage = current_price * self.config['slippage_pct']
            entry_cost = self.config['broker_fee'] + entry_slippage
            
            # Create position
            position_id = f"{signal['symbol']}_{signal['datetime']}"
            portfolio[position_id] = {
                'symbol': signal['symbol'],
                'qty': position_size,
                'entry_price': current_price,
                'entry_date': signal['datetime'],
                'entry_cost': entry_cost,
                'exit_price': None,
                'exit_date': None,
                'pnl': None
            }
            
            current_capital -= entry_cost  # Deduct commission
        
        # Mark positions to market daily; evaluate exits
        for date in pd.date_range(start_date, end_date, freq='D'):
            if date.weekday() > 4:  # Skip weekends
                continue
            
            for position_id, position in list(portfolio.items()):
                current_price = price_data[position['symbol']].loc[date, 'close']
                
                # Check exit conditions
                exit_condition = self._check_exit_condition(position, current_price, date)
                
                if exit_condition:
                    # Exit position
                    exit_slippage = current_price * self.config['slippage_pct']
                    exit_cost = self.config['broker_fee'] + exit_slippage
                    
                    gross_pnl = (current_price - position['entry_price']) * position['qty']
                    net_pnl = gross_pnl - exit_cost
                    
                    position['exit_price'] = current_price
                    position['exit_date'] = date
                    position['pnl'] = net_pnl
                    
                    current_capital += gross_pnl  # Add P&L back
                    
                    self.trade_log.append(position)
                    del portfolio[position_id]
            
            # Record daily equity
            unrealized_pnl = sum([
                (price_data[pos['symbol']].loc[date, 'close'] - pos['entry_price']) * pos['qty']
                for pos in portfolio.values()
            ]) if portfolio else 0
            
            self.equity_curve.append({
                'date': date,
                'equity': current_capital + unrealized_pnl
            })
        
        # Finalize: close remaining positions at end date
        final_date = end_date
        for position_id, position in portfolio.items():
            final_price = price_data[position['symbol']].loc[final_date, 'close']
            position['exit_price'] = final_price
            position['exit_date'] = final_date
            position['pnl'] = (final_price - position['entry_price']) * position['qty'] - self.config['broker_fee']
            self.trade_log.append(position)
        
        return self._compute_backtest_results()
    
    def _compute_backtest_results(self):
        """
        Calculate backtest metrics from trade log.
        """
        
        if len(self.trade_log) == 0:
            return None
        
        trade_df = pd.DataFrame(self.trade_log)
        
        # Metrics
        total_trades = len(trade_df)
        winning_trades = (trade_df['pnl'] > 0).sum()
        losing_trades = (trade_df['pnl'] <= 0).sum()
        
        gross_profit = trade_df[trade_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trade_df[trade_df['pnl'] <= 0]['pnl'].sum())
        
        net_profit = trade_df['pnl'].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Drawdown
        equity_curve = pd.Series([x['equity'] for x in self.equity_curve])
        running_max = equity_curve.cummax()
        drawdown = (running_max - equity_curve) / running_max
        max_drawdown = drawdown.max()
        
        # Sharpe ratio
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'net_profit': net_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_trade_profit': net_profit / total_trades,
            'trade_log': trade_df
        }
```

### 9.2 Walk-Forward Validation

```python
def walk_forward_validation(signals_df, price_data, config, window_size=252, step=20):
    """
    Time-series aware cross-validation: train on past, test on future (no lookahead).
    """
    
    results = []
    
    for train_end in range(window_size, len(signals_df), step):
        train_start = max(0, train_end - window_size * 2)
        test_end = min(len(signals_df), train_end + window_size)
        
        train_signals = signals_df.iloc[train_start:train_end]
        test_signals = signals_df.iloc[train_end:test_end]
        
        # Retrain model on train_signals
        model = train_signal_model(train_signals, price_data)
        
        # Backtest on test_signals (no lookahead)
        test_signals_with_preds = model.predict(test_signals)
        backtest_result = backtest_engine.run_backtest(test_signals_with_preds, price_data, test_signals.index[0], test_signals.index[-1])
        
        results.append({
            'train_period': (train_start, train_end),
            'test_period': (train_end, test_end),
            'net_profit': backtest_result['net_profit'],
            'sharpe': backtest_result['sharpe_ratio'],
            'max_dd': backtest_result['max_drawdown']
        })
    
    # Aggregate results
    mean_profit = np.mean([r['net_profit'] for r in results])
    mean_sharpe = np.mean([r['sharpe'] for r in results])
    robustness = np.std([r['net_profit'] for r in results])  # Lower is more robust
    
    return {
        'walk_forward_results': results,
        'mean_profit': mean_profit,
        'mean_sharpe': mean_sharpe,
        'robustness_score': 1 / (1 + robustness / abs(mean_profit))
    }
```

---

## 10. RISK MANAGEMENT & CIRCUIT BREAKER

### 10.1 Risk Rules Engine

```python
class RiskManager:
    
    def __init__(self, config):
        self.config = config
        self.daily_pnl = 0
        self.max_concurrent_positions = 0
        self.circuit_breaker_active = False
    
    def check_trading_allowed(self, account_state):
        """
        Gate function: return True if trading should proceed, False if circuit breaker active.
        """
        
        checks = []
        
        # 1. Daily loss limit
        self.daily_pnl = account_state['daily_pnl']
        daily_loss_pct = self.daily_pnl / account_state['starting_capital']
        if daily_loss_pct < -self.config['daily_loss_limit_pct']:
            logger.warning(f"Daily loss {daily_loss_pct*100:.2f}% exceeds limit. Trading paused.")
            return False
        checks.append(True)
        
        # 2. Concurrent position limit
        num_open = len(account_state['open_positions'])
        if num_open >= self.config['max_concurrent_positions']:
            logger.debug(f"Max concurrent positions ({num_open}) reached.")
            return False
        checks.append(True)
        
        # 3. Margin availability
        margin_available = account_state['available_margin']
        if margin_available < self.config['min_margin_buffer']:
            logger.warning(f"Available margin {margin_available} < buffer. Trading paused.")
            return False
        checks.append(True)
        
        # 4. API latency check
        if account_state.get('api_latency_ms', 0) > self.config['max_api_latency_ms']:
            logger.error(f"API latency {account_state['api_latency_ms']}ms exceeds threshold.")
            return False
        checks.append(True)
        
        # 5. System health (exceptions in last minute)
        exception_count = account_state.get('exceptions_last_minute', 0)
        if exception_count > self.config['max_exceptions_threshold']:
            logger.critical(f"Too many exceptions ({exception_count}). Pausing trading.")
            return False
        checks.append(True)
        
        # 6. Market hours check
        if not self._is_market_open():
            return False
        checks.append(True)
        
        return all(checks)
    
    def _is_market_open(self):
        """
        Check if market is currently open (NSE/BSE hours: 9:15 AM - 3:30 PM IST).
        """
        now = datetime.now(tz=pytz.timezone('Asia/Kolkata'))
        market_open = now.time() >= time.fromisoformat('09:15:00')
        market_close = now.time() <= time.fromisoformat('15:30:00')
        return market_open and market_close and now.weekday() < 5  # Not weekend
```

### 10.2 Per-Trade Risk Limits

```python
def validate_position_risk(position, account_state, config):
    """
    Pre-trade checks on risk exposure.
    """
    
    # 1. Max notional exposure per symbol
    symbol_exposure = sum([p['notional'] for p in account_state['open_positions'] if p['symbol'] == position['symbol']])
    max_exposure = config['max_notional_per_symbol_pct'] * account_state['capital'] / 100
    
    if symbol_exposure + position['notional'] > max_exposure:
        return False, f"Symbol exposure would exceed {max_exposure} INR"
    
    # 2. Max portfolio notional
    total_notional = sum([p['notional'] for p in account_state['open_positions']]) + position['notional']
    max_portfolio_notional = config['max_total_notional_pct'] * account_state['capital'] / 100
    
    if total_notional > max_portfolio_notional:
        return False, f"Total notional would exceed {max_portfolio_notional} INR"
    
    # 3. Capital at risk (position size * potential loss)
    capital_at_risk = position['qty'] * abs(position['entry_price'] - position['stop_loss_price'])
    max_capital_at_risk = account_state['capital'] * config['max_capital_at_risk_pct'] / 100
    
    if capital_at_risk > max_capital_at_risk:
        return False, f"Capital at risk {capital_at_risk} INR exceeds max {max_capital_at_risk} INR"
    
    return True, "All checks passed"
```

---

## 11. MONITORING, LOGGING & ALERTING

### 11.1 Metrics to Track

| Metric | Frequency | Threshold for Alert |
|--------|-----------|-------------------|
| **Real-time P&L** | Every 1 min | Daily loss > 5–7% |
| **Drawdown** | Every 1 min | Max DD > 15% |
| **Open positions** | Every 1 min | Count > 3 |
| **Order latency** | Per order | > 500 ms |
| **Fill rate** | Per order | < 95% within 60s |
| **Slippage** | Per fill | > 0.5% |
| **API response time** | Every 10s | > 200 ms avg |
| **Margin utilization** | Every 1 min | > 80% of available |
| **Model accuracy** | Daily | < 70% on recent 100 trades |

### 11.2 Logging Format (Structured JSON)

```json
{
  "timestamp": "2025-11-05T10:30:45.123Z",
  "event_type": "ORDER_PLACED",
  "order_id": "order_123456",
  "symbol": "NIFTYNOVFUT",
  "transaction_type": "BUY",
  "quantity": 1,
  "price": 23500.0,
  "status": "success",
  "execution_latency_ms": 45,
  "user_id": "user_001",
  "signal_id": "signal_abc123",
  "reason_code": "MOMENTUM_SIGNAL",
  "model_confidence": 0.85,
  "expected_profit_inr": 250.0
}
```

### 11.3 Alert Configuration

```yaml
ALERTS:
  - alert_id: "DAILY_LOSS_LIMIT"
    condition: "daily_pnl < -500"
    severity: "CRITICAL"
    action: "PAUSE_TRADING"
    notify: ["email", "dashboard"]
  
  - alert_id: "HIGH_LATENCY"
    condition: "api_latency_ms > 500"
    severity: "WARNING"
    action: "LOG_AND_MONITOR"
    notify: ["dashboard"]
  
  - alert_id: "MODEL_DRIFT"
    condition: "recent_accuracy < 0.70"
    severity: "WARNING"
    action: "RETRAIN_MODEL"
    notify: ["email"]
  
  - alert_id: "MARGIN_BREACH"
    condition: "available_margin < 0"
    severity: "CRITICAL"
    action: "EMERGENCY_LIQUIDATE"
    notify: ["email", "sms"]
```

---

## 12. CONFIGURATION TEMPLATE (YAML)

```yaml
# ==============================================================================
# Zerodha Kite MCP AI Trading Bot - Configuration Template
# ==============================================================================

# PLATFORM & AUTHENTICATION
PLATFORM:
  broker: "zerodha"
  mcp_endpoint: "https://mcp.kite.trade"  # or local MCP server
  api_key: "${KITE_API_KEY}"
  oauth_callback_url: "http://localhost:8000/callback"
  
# CAPITAL & RISK
CAPITAL:
  starting_capital_inr: 10000
  target_roi_pct: 100  # 100% ROI target (informational)
  daily_loss_limit_pct: 5  # Stop trading if loss exceeds 5%
  max_drawdown_pct: 15
  
# POSITION MANAGEMENT
POSITION_SIZING:
  max_risk_per_trade_pct: 5  # 5% of capital at risk per trade
  max_concurrent_positions: 3
  max_capital_allocation_f_and_o_pct: 30  # Max 30% of capital in options
  max_capital_allocation_equity_pct: 10   # Max 10% per stock
  
# TRADING RULES
TRADING_RULES:
  min_confidence_threshold: 0.80  # Classifier probability
  min_expected_net_profit_inr: 150  # Minimum expected profit per trade
  broker_fee_per_trade_inr: 55  # Average of 50–60 INR
  estimated_slippage_pct: 0.01  # 0.1% conservative estimate
  
# F&O STRATEGY
F_AND_O:
  enabled: true
  allowed_instruments: ["NIFTYNOV2025C", "NIFTYNOV2025P"]  # CE & PE only
  nifty_lot_size: 75
  strike_selection:
    delta_preference_min: 0.20
    delta_preference_max: 0.40
    min_oi: 1000
    min_volume: 500
    max_bid_ask_spread_pct: 0.02
  expiry_preference: "nearest_two"
  
  # Entry signals
  intraday_entry:
    macd_cross: true
    rsi_threshold_bearish: 70
    rsi_threshold_bullish: 30
  
  swing_entry:
    ema_cross: true
    adx_threshold: 25
    timeframe_alignment: "15m,1h"
  
  # Exit rules
  exits:
    take_profit_pct: 50  # 50% gain on premium
    stop_loss_pct: 40  # 60% loss on premium
    time_based_days_to_expiry: 1  # Exit 1 day before expiry
    intraday_eod_close: true
  
  cool_down_hours: 48  # Min time between new F&O trades
  
# EQUITY STRATEGY
EQUITY:
  enabled: true
  watchlist: ["INFY", "TCS", "WIPRO", "LT", "HDFC"]
  
  swing_trades:
    enabled: true
    holding_period_days: 5
    target_return_pct: 5
    stop_loss_pct: 3
  
  intraday_trades:
    enabled: true
    max_position_duration_hours: 4
    target_return_pct: 2
    stop_loss_pct: 1.5
  
  position_sizing:
    method: "risk_based"
    max_allocation_pct_per_stock: 10

# SIGNAL MODEL
SIGNAL_MODEL:
  model_type: "ensemble"  # XGBoost classifier + regressor
  features:
    - "momentum"
    - "mean_reversion"
    - "volatility"
    - "microstructure"
    - "option_specific"
  
  classifier:
    min_prob_threshold: 0.80
    algorithm: "xgboost"
    hyperparams:
      max_depth: 6
      learning_rate: 0.05
      n_estimators: 100
  
  regressor:
    target: "expected_return_pct"
    algorithm: "xgboost"
  
  retraining:
    backtest_mode: "offline"  # Retrain every 2 weeks
    paper_trading_retrain_frequency_days: 7
    live_retrain_trigger: "model_drift_detected"

# EXECUTION
EXECUTION:
  order_type_default: "LIMIT"
  order_type_emergency: "SL_M"
  order_validity: "DAY"
  order_wait_timeout_seconds: 300
  use_bracket_orders: false  # For MVP
  
  order_placement:
    batch_size: 1  # Place 1 order at a time
    rate_limit_per_minute: 10
  
  retry_logic:
    max_retries: 3
    exponential_backoff_base: 2

# DATA
DATA:
  sources:
    - "kite_websocket"  # Real-time tick data
    - "kite_rest"       # Option chain, historical candles
  
  historical_storage:
    format: "parquet"
    location: "/data/historical/"
    retention_days: 1825  # 5+ years
  
  candles:
    timeframes: ["1m", "5m", "15m", "1d"]
    lookback_periods:
      "1m": 100
      "5m": 100
      "15m": 50
      "1d": 252

  option_chain:
    update_frequency_ms: 1000
    snapshot_frequency_min: 5

# MONITORING & LOGGING
MONITORING:
  log_level: "INFO"
  log_format: "json"
  log_destination: "/logs/trading_bot.log"
  
  metrics:
    track_p_and_l_interval_sec: 60
    track_latency_interval_sec: 10
    track_fill_rate_interval_sec: 60
  
  alerts:
    email_recipients: ["user@example.com"]
    sms_enabled: false
    slack_webhook: ""
    
  persistence:
    audit_log: "/logs/audit.json"
    trade_log: "/logs/trades.parquet"

# BACKTEST & VALIDATION
BACKTEST:
  enabled: true
  start_date: "2022-01-01"
  end_date: "2025-10-31"
  fee_model: "fixed"
  fee_per_trade: 55
  slippage_model: "percentage"
  slippage_pct: 0.01
  
  walk_forward:
    enabled: true
    window_size_days: 252
    step_size_days: 20

# UI / CONTROL
UI:
  mode: "SEMI_AUTOMATED"  # "SEMI_AUTOMATED" or "FULLY_AUTOMATED"
  trade_confirmation_timeout_sec: 30
  port: 8000
  host: "127.0.0.1"

# MISC
MISC:
  timezone: "Asia/Kolkata"
  market_open_time: "09:15:00"
  market_close_time: "15:30:00"
  enable_paper_trading_mode: true
  enable_debug_mode: false
```

---

## 13. TESTING PLAN

### 13.1 Unit Tests

| Test | Scenario | Expected Outcome |
|------|----------|------------------|
| `test_strike_selection` | Multiple liquid strikes available | Selects highest OI, tightest spread |
| `test_position_sizing` | Capital = 10k, trade cost = 200 INR | Allocates 1 lot respecting 5% risk |
| `test_expected_profit_calc` | Gross P&L 300, fees 50 each side | Net profit = 200 INR |
| `test_fee_check` | Expected profit 100 INR, min required 150 | Trade rejected |
| `test_order_validation` | Invalid lot size (not multiple of 75) | Order rejected |
| `test_margin_check` | Insufficient margin | Order rejected before placement |

### 13.2 Integration Tests

| Test | Scenario | Expected Outcome |
|------|----------|------------------|
| `test_full_trade_cycle` | Signal → position size → order placement → fill reconciliation → exit | All steps executed, P&L logged |
| `test_mcp_oauth_flow` | Authenticate via MCP OAuth | Token obtained, market data accessible |
| `test_concurrent_positions` | 3 positions open, new signal arrives | Trade accepted; triggers when position closes |
| `test_cool_down` | F&O trade placed, another signal within 48h | Second trade rejected |
| `test_circuit_breaker_daily_loss` | Daily loss > 5% | New trades paused, alert sent |

### 13.3 Edge Cases & Stress Tests

| Scenario | Action | Expected Behavior |
|----------|--------|------------------|
| **Market open gap** | Limit order not filled at market open | Cancel, re-evaluate at next signal |
| **Large slippage** | Fill price 2% worse than expected | Log and recalibrate slippage model |
| **Exchange halt** | NSE trading halted mid-day | Pause automation, hold existing positions |
| **Margin call** | Available margin becomes negative | Emergency liquidate, pause trading |
| **API outage** | Kite MCP unavailable for 5 minutes | Retry with backoff; if > 10 min, alert user |
| **High volatility** | IV jumps 50% in seconds | Reassess position; may hit stops faster |
| **Liquidity dry-up** | Bid-ask spread widens > 5% | Skip entry; wait for better spreads |

---

## 14. DEPLOYMENT & OPERATIONS

### 14.1 MVP (Weeks 0–8)

**Phase 1: Core Infrastructure (Weeks 0–2)**
- Data ingester: Kite WebSocket + historical data loader
- Basic feature store (in-memory cache)
- Backtest engine (vectorized)
- Validate fee/slippage model against historical fills

**Phase 2: Strategy Implementation (Weeks 2–4)**
- F&O rules engine: strike selection, position sizing, exits
- Equity swing rules
- Cool-down & batching logic
- Run historical sims; target > 50% win rate in backtest

**Phase 3: ML & Signals (Weeks 4–6)**
- Train XGBoost classifier + regressor
- Feature engineering pipeline
- Walk-forward validation
- Backtest with ML signals (target: net profit > 0, Sharpe > 1)

**Phase 4: MCP Integration & UI (Weeks 6–8)**
- Kite MCP client setup in VS Code
- Order construction, placement, reconciliation
- Trade preview UI (semi-automated mode)
- Paper trading mode (simulator)

**Phase 5: Hardening & Launch (Weeks 8–12)**
- Risk guardrails: circuit breaker, margin checks, drawdown limits
- Monitoring dashboards, logging, alerting
- Audit trail & compliance checks
- 30–90 days paper trading validation
- Selective manual override + gradual auto mode enable

### 14.2 Deployment Checklist

- [ ] All dependencies installed (Python 3.10+, pandas, xgboost, asyncio, aiohttp)
- [ ] Kite API credentials configured & OAuth flow tested
- [ ] Historical data ingested (3+ years for backtest)
- [ ] Backtest results reviewed (win rate, Sharpe, drawdown metrics)
- [ ] Walk-forward validation passed (robustness > 0.6)
- [ ] Paper trading run 30+ days with minimal drawdown
- [ ] Live slippage model calibrated
- [ ] All risk checks operational (daily loss, margin, concurrent)
- [ ] Monitoring & alerting configured
- [ ] Audit log and trade log persistent storage verified
- [ ] Manual kill-switch accessible
- [ ] Disclaimers displayed to user; consent obtained
- [ ] Automated restart script in place (cron + systemd)

---

## 15. SUCCESS METRICS & ACCEPTANCE CRITERIA

### 15.1 Backtest Requirements (Gate to Paper Trading)

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Net Profit** | > 500 INR (5% ROI) | Covers fees + slippage; demonstrates profitability |
| **Trade Count** | ≤ 20 total over backtest | Low trade volume (fee conscious) |
| **Win Rate** | ≥ 50% | Majority of trades profitable |
| **Profit Factor** | ≥ 1.5 | Gross wins ≥ 1.5 × gross losses |
| **Sharpe Ratio** | ≥ 1.0 | Positive risk-adjusted returns |
| **Max Drawdown** | ≤ 15% | Capital preservation |
| **Avg Trade Profit** | ≥ 50 INR after fees | Justifies execution cost |

### 15.2 Paper Trading Requirements (Gate to Live)

| Metric | Target | Duration |
|--------|--------|----------|
| **Consecutive Profitable Days** | ≥ 10 out of 20 trading days | 1 month |
| **Backtest vs Paper Correlation** | ≥ 0.80 | Backtest reflects live behavior |
| **Order Fill Rate** | ≥ 95% within 60s | Execution reliability |
| **Slippage vs Model** | Within ±0.2% | Model calibration accuracy |
| **Drawdown (Paper)** | ≤ 10% | Risk control working |

### 15.3 Live Trading KPIs

| KPI | Target | Measurement |
|-----|--------|-------------|
| **Monthly Net Profit** | > 100 INR/month (1% ROI) | Cumulative |
| **Trades per Month** | 5–15 | Low count, high conviction |
| **Avg Profit per Trade** | ≥ 100 INR | After all costs |
| **Sharpe Ratio (rolling 60d)** | ≥ 0.8 | Risk-adjusted performance |
| **Max Drawdown (rolling)** | ≤ 15% | Capital preservation |
| **Model Accuracy (recent 100 trades)** | ≥ 75% | Signal quality |

---

## 16. IMPLEMENTATION ROADMAP & MVP CHECKLIST

### 16.1 Week-by-Week Roadmap

| Week | Milestone | Deliverables |
|------|-----------|--------------|
| **0–2** | Data infrastructure | Historical data (3y), Feature store, Backtest engine |
| **2–4** | Rules engine | F&O strike selection, Position sizing, Exit logic, Cool-down |
| **4–6** | ML pipeline | Feature engineering, XGBoost models, Walk-forward validation |
| **6–8** | MCP integration | Kite OAuth, Order placement, Reconciliation, Semi-auto UI |
| **8–10** | Risk & monitoring | Circuit breaker, Drawdown limits, Logging, Dashboards |
| **10–12** | Validation & deploy | Paper trading 30d, Slippage calibration, Live-readiness checklist |

### 16.2 MVP Minimal Viable Automated Pipeline

**Scope:** Single-instrument (NIFTY), single strategy (intraday), semi-automated

```python
# MVP Entry Point: main.py

import asyncio
from kite_mcp_client import KiteMCPClient
from data_engine import DataEngine
from signal_model import SignalModel
from order_executor import OrderExecutor
from risk_manager import RiskManager

async def main():
    # 1. Initialize components
    kite_client = KiteMCPClient(api_key=config['kite_api_key'])
    data_engine = DataEngine(kite_client, config)
    signal_model = SignalModel(config)
    executor = OrderExecutor(kite_client, config)
    risk_mgr = RiskManager(config)
    
    # 2. Load historical data & train model
    await data_engine.load_historical_data(start_date='2022-01-01', end_date=today)
    signal_model.train(data_engine.get_training_data())
    
    # 3. Real-time loop
    while True:
        # Fetch latest data
        market_data = await data_engine.fetch_realtime_data(symbols=['NIFTY50', 'NIFTYNOV50C', 'NIFTYNOV50P'])
        
        # Generate signals
        signals = signal_model.predict(market_data)
        
        # Filter & validate
        for signal in signals:
            if not signal['valid'] or not risk_mgr.check_trading_allowed():
                continue
            
            # Position sizing
            pos_size = calculate_position_size(signal)
            if pos_size is None:
                continue
            
            # Expected profit check
            expected_pnl = calculate_expected_net_profit(signal, pos_size, market_data)
            if expected_pnl < config['min_expected_net_profit']:
                continue
            
            # User confirmation (semi-auto)
            order_preview = executor.construct_order(signal, pos_size)
            user_confirm = await ui.prompt_user(order_preview, timeout_sec=30)
            
            if user_confirm:
                result = executor.place_order(order_preview)
                if result['success']:
                    logger.info(f"Order placed: {result['order_id']}")
        
        await asyncio.sleep(60)  # Poll every 60 seconds

if __name__ == '__main__':
    asyncio.run(main())
```

### 16.3 MVP Checklist

- [ ] Kite MCP OAuth working in VS Code
- [ ] Historical data (3 years) loaded to Parquet
- [ ] Basic candle feature extraction (EMA, RSI, MACD)
- [ ] XGBoost classifier trained (p(win) ≥ 0.80 threshold)
- [ ] Backtest engine running; metrics displayed
- [ ] F&O position sizing for NIFTY working
- [ ] Order construction & validation logic implemented
- [ ] Semi-automated UI (trade preview + confirm button)
- [ ] Paper trading mode (simulated fills) working
- [ ] P&L dashboard updating in real-time
- [ ] Audit log capturing all trades
- [ ] Risk circuit breaker (daily loss limit) active
- [ ] Alert system (email/dashboard) configured

---

## 17. PSEUDO-CODE FOR CRITICAL FUNCTIONS

### 17.1 Strike Selection (Detailed)

```python
def select_optimal_strike(
    underlying_price: float,
    signal_type: str,  # 'CALL' or 'PUT'
    iv_regime: str,  # 'high' or 'low'
    capital_available: float,
    option_chain_snapshot: dict
) -> dict:
    """
    Select the best option strike based on delta, liquidity, and affordability.
    """
    
    atm = round(underlying_price / 100) * 100
    
    # Generate candidate strikes
    if signal_type == 'CALL':
        candidates = [atm - 200, atm - 100, atm, atm + 100, atm + 200]
    else:  # PUT
        candidates = [atm + 200, atm + 100, atm, atm - 100, atm - 200]
    
    valid_strikes = []
    
    for strike in candidates:
        if strike not in option_chain_snapshot:
            continue
        
        opt = option_chain_snapshot[strike]
        
        # Liquidity filter
        if opt['oi'] < 1000 or opt['volume'] < 500:
            continue
        
        # Spread filter
        spread = (opt['ask'] - opt['bid']) / opt['midprice']
        if spread > 0.02:  # 2%
            continue
        
        # Delta filter (prefer OTM)
        if not (0.20 <= abs(opt['delta']) <= 0.40):
            continue
        
        # IV filter
        if iv_regime == 'high':
            iv_pct = (opt['iv'] - iv_min_252) / (iv_max_252 - iv_min_252)
            if iv_pct < 0.60:
                continue
        
        # Affordability
        premium = opt['last_price']
        if premium * 75 > capital_available:  # 75 = lot size
            continue
        
        valid_strikes.append({
            'strike': strike,
            'premium': premium,
            'delta': opt['delta'],
            'oi': opt['oi'],
            'spread_pct': spread,
            'score': opt['oi'] * (1 - spread) / premium  # Higher OI, lower spread, lower premium = better
        })
    
    if not valid_strikes:
        return None
    
    # Return best strike by score
    return sorted(valid_strikes, key=lambda x: x['score'], reverse=True)[0]
```

### 17.2 Expected Net Profit Calculation

```python
def calculate_expected_net_profit(
    signal: dict,
    model_prediction: dict,  # {'classifier_prob': 0.85, 'expected_return_pct': 0.05}
    position_size: int,
    entry_price: float,
    stop_loss_price: float,
    config: dict
) -> dict:
    """
    Calculate expected net profit in INR after fees and slippage.
    This is the core decision gate: trade only if E[profit] >> fees.
    """
    
    p_win = model_prediction['classifier_prob']  # e.g., 0.85
    exp_return_pct = model_prediction['expected_return_pct']  # e.g., 0.05
    
    # Win scenario: profit by exp_return_pct
    exit_price_win = entry_price * (1 + exp_return_pct)
    gross_pnl_win = (exit_price_win - entry_price) * position_size
    
    # Loss scenario: loss to stop-loss
    gross_pnl_loss = (stop_loss_price - entry_price) * position_size
    
    # Costs
    entry_fee = config['broker_fee_per_trade']
    exit_fee = config['broker_fee_per_trade']
    entry_slippage = estimate_slippage(entry_price, config['slippage_pct'])
    exit_slippage = estimate_slippage(exit_price_win, config['slippage_pct'])
    total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage
    
    # Risk-adjusted expected value
    net_pnl_win = gross_pnl_win - total_cost
    net_pnl_loss = gross_pnl_loss - total_cost
    
    expected_net_pnl = p_win * net_pnl_win + (1 - p_win) * net_pnl_loss
    
    # Decision gate
    min_expected_profit = config['min_expected_net_profit_per_trade']
    trade_approved = expected_net_pnl >= min_expected_profit
    
    return {
        'expected_net_pnl': expected_net_pnl,
        'net_pnl_win': net_pnl_win,
        'net_pnl_loss': net_pnl_loss,
        'total_cost': total_cost,
        'trade_approved': trade_approved,
        'approval_reason': f"E[profit]={expected_net_pnl:.0f} INR; Required >= {min_expected_profit} INR"
    }
```

### 17.3 Order Placement with Reconciliation

```python
async def place_and_reconcile_order(
    order_dict: dict,
    kite_client: KiteMCPClient,
    expected_fill_price: float,
    config: dict
) -> dict:
    """
    Place order, wait for fill, reconcile against expected price.
    """
    
    # Step 1: Place order
    try:
        response = kite_client.place_order(
            variety=order_dict['variety'],
            exchange=order_dict['exchange'],
            tradingsymbol=order_dict['tradingsymbol'],
            transaction_type=order_dict['transaction_type'],
            quantity=order_dict['quantity'],
            price=order_dict['price'],
            order_type=order_dict['order_type'],
            product=order_dict['product'],
            validity=order_dict['validity']
        )
        order_id = response['order_id']
        logger.info(f"Order placed: {order_id}")
    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return {'success': False, 'error': str(e)}
    
    # Step 2: Monitor fill
    max_wait = config['order_wait_timeout_seconds']
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        order_status = kite_client.get_order_history(order_id)
        
        if order_status['status'] == 'COMPLETE':
            filled_qty = order_status['filled_quantity']
            fill_price = order_status['average_price']
            fill_time = order_status['exchange_timestamp']
            
            # Step 3: Reconcile
            slippage = fill_price - expected_fill_price
            slippage_pct = slippage / expected_fill_price
            
            logger.info(f"Order {order_id} filled: qty={filled_qty}, price={fill_price}, slippage={slippage_pct*100:.3f}%")
            
            # Log for slippage model recalibration
            slippage_log.append({
                'order_id': order_id,
                'expected_price': expected_fill_price,
                'fill_price': fill_price,
                'slippage_pct': slippage_pct,
                'timestamp': fill_time
            })
            
            return {
                'success': True,
                'order_id': order_id,
                'filled_qty': filled_qty,
                'fill_price': fill_price,
                'slippage_pct': slippage_pct
            }
        
        elif order_status['status'] in ['REJECTED', 'CANCELLED']:
            logger.warning(f"Order {order_id} {order_status['status']}")
            return {'success': False, 'reason': order_status['status']}
        
        await asyncio.sleep(5)  # Poll every 5 seconds
    
    # Timeout: cancel order
    logger.warning(f"Order {order_id} not filled within {max_wait}s. Cancelling.")
    kite_client.cancel_order(order_id)
    return {'success': False, 'reason': 'TIMEOUT'}
```

---

## 18. SAFETY, COMPLIANCE & DISCLAIMERS

### 18.1 MCP OAuth Scopes & Permissions

**Requested Scopes:**
- `read:market_data` – Fetch candles, option chain, tick data
- `read:account` – Check holdings, margin, account balance
- `write:orders` – Place, modify, cancel orders
- `read:orders` – Track order status and fills

**User Consent Flow:**
```
1. User opens bot in VS Code
2. Bot displays OAuth consent dialog with requested scopes
3. User approves via Zerodha login
4. Token obtained; stored securely (short-lived, refresh available)
5. Confirmation: "Connected to Zerodha Kite. Ready to trade."
```

### 18.2 Mandatory Disclaimers

```
═══════════════════════════════════════════════════════════════════
⚠ TRADING RISK DISCLAIMER ⚠

This AI trading bot is provided AS-IS for EDUCATIONAL and AUTOMATED
TRADING purposes. By using this bot, you acknowledge and accept the
following:

1. NO GUARANTEED RETURNS: Trading in financial markets involves
   substantial risk of loss. Past performance does not guarantee
   future results. This bot may incur losses.

2. CAPITAL AT RISK: You may lose some or all of your capital. Use
   only money you can afford to lose completely.

3. NO PROFESSIONAL ADVICE: This bot does not constitute financial,
   investment, or trading advice. Consult a professional advisor
   before trading.

4. BOT FAILURES: The bot may fail to execute trades, may place
   unexpected trades, or may behave unpredictably due to bugs,
   API outages, or market conditions.

5. USER RESPONSIBILITY: You are fully responsible for all trades
   executed via this bot, including losses and tax implications.

6. MARKET RULES: You are responsible for complying with all
   applicable laws, regulations, and exchange rules (NSE, BSE).

═══════════════════════════════════════════════════════════════════

I have read and understood the above disclaimer and accept full
responsibility for trading decisions made by this bot.

[ I AGREE ]  [ I DECLINE ]
```

### 18.3 API Rate Limit & Compliance

- Zerodha Kite: 1000 requests/minute typical limit
- Bot enforces: max 10 orders/minute (safety margin)
- Auto-backoff on rate limit 429 response

---

## 19. CONFIGURATION EXAMPLES & SCENARIOS

### 19.1 Conservative (Low Risk)

```yaml
CAPITAL:
  starting_capital_inr: 10000
  daily_loss_limit_pct: 3
  max_drawdown_pct: 10

TRADING_RULES:
  min_confidence_threshold: 0.85
  min_expected_net_profit_inr: 200

POSITION_SIZING:
  max_risk_per_trade_pct: 3
  max_concurrent_positions: 2

F_AND_O:
  cool_down_hours: 72
  exits:
    take_profit_pct: 75
    stop_loss_pct: 30

BACKTEST:
  slippage_pct: 0.02  # Higher slippage assumption
```

**Expected:** Very few trades, low volatility, slower profit accumulation.

### 19.2 Moderate (Balanced)

```yaml
# Use default config from section 12
```

**Expected:** 5–10 trades/month, ~10% max drawdown, steady profit.

### 19.3 Aggressive (High Conviction)

```yaml
TRADING_RULES:
  min_confidence_threshold: 0.75
  min_expected_net_profit_inr: 100

POSITION_SIZING:
  max_risk_per_trade_pct: 7
  max_concurrent_positions: 5

F_AND_O:
  cool_down_hours: 24
  exits:
    take_profit_pct: 30
    stop_loss_pct: 50

CAPITAL:
  daily_loss_limit_pct: 10
  max_drawdown_pct: 20
```

**Expected:** 15–20 trades/month, higher drawdown, faster profit (or loss).

---

## 20. CONCLUSION & NEXT STEPS

This PRD provides a **complete, explicit, and engineer-ready specification** for building a Zerodha Kite MCP-driven AI trading bot in VS Code. The design emphasizes:

✅ **Fee consciousness:** Every trade justified by expected return > 2× broker fee  
✅ **Capital preservation:** Strict drawdown limits and circuit breakers  
✅ **Reproducibility:** Auditable signals, backtested logic, walk-forward validation  
✅ **Operational simplicity:** Semi-automated MVP; manual override always available  
✅ **Explainability:** Feature importance, per-trade justification trace  

**Immediate Next Steps:**

1. **Clone repo** with starter code (data ingester, backtest stub)
2. **Load historical data** (3 years NIFTY + top stocks)
3. **Train baseline classifier** (target: 75%+ accuracy)
4. **Run backtest** (target: 5–10% ROI, Sharpe > 1, max DD < 15%)
5. **Integrate Kite MCP** (OAuth flow, order API)
6. **Paper trade 30 days** (validate slippage model, refine signals)
7. **Go live** (small capital, gradual scale, continuous monitoring)

**Success Probability:** High, given disciplined risk management, low trade count, and realistic expectations (100% ROI is aspirational; 20–50% annual ROI is more probable).

---

**Document Version:** 1.0  
**Date:** November 5, 2025  
**Author:** Quantitative Trading Systems Design  
**Status:** Ready for Implementation
