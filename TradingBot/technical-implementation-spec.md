# Zerodha Kite MCP AI Trading Bot
## Technical Implementation Specification & Code Architecture

---

## PART A: TECHNICAL ARCHITECTURE & COMPONENT DESIGN

### A1. System Components Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ZERODHA KITE MCP AI TRADING BOT                       │
│                         (VS Code Environment)                            │
└─────────────────────────────────────────────────────────────────────────┘

LAYER 1: USER INTERFACE & CONTROL
├── Trade Confirmation Panel (VS Code Webview)
├── Real-time Dashboard (P&L, positions, alerts)
├── Manual Override Controls
└── Audit Log Viewer

LAYER 2: ORCHESTRATION & STATE MANAGEMENT
├── Signal Aggregator (deduplicate, batch signals)
├── Position Tracker (maintain open positions, P&L)
├── Order Queue Manager (FIFO persistent queue)
├── Risk Circuit Breaker (daily loss, margin, latency)
└── Account State Manager (cache capital, holdings, margin)

LAYER 3: STRATEGY EXECUTION ENGINE
├── F&O Strategy Module
│   ├── Strike Selector
│   ├── Option Entry Filter
│   ├── Position Sizer (options)
│   └── Exit Logic (TP, SL, time-based)
├── Equity Strategy Module
│   ├── Stock Filter
│   ├── Swing/Intraday Entry Logic
│   ├── Position Sizer (equity)
│   └── Exit Logic (TP, SL)
├── Signal Model (ML + Rules)
│   ├── Feature Pipeline
│   ├── XGBoost Classifier (p_win)
│   ├── XGBoost Regressor (expected_return)
│   └── Threshold Application
└── Expected Profit Calculator (fee-aware)

LAYER 4: EXECUTION ENGINE
├── Kite MCP Client Wrapper
│   ├── OAuth & Token Management
│   ├── WebSocket Subscribe/Unsubscribe
│   └── REST API Calls (place, modify, cancel, history)
├── Order Constructor (build order dict)
├── Order Validator (margin, lot size, price sanity)
├── Order Placer (with retry logic)
├── Fill Reconciler (track fills, slippage)
└── Execution Auditor (log all events)

LAYER 5: DATA & FEATURE LAYER
├── Data Ingester
│   ├── Kite WebSocket Listener
│   ├── Historical Data Loader (Parquet)
│   ├── Option Chain Fetcher
│   └── Market Calendar
├── Feature Store
│   ├── Real-time Cache (Redis or in-memory)
│   ├── Feature Computer (EMA, RSI, MACD, etc.)
│   └── Option Greeks Calculator (approximation)
└── Historical Storage
    ├── Parquet Files (candles, option snapshots)
    ├── Time-series DB (InfluxDB/Timescale optional)
    └── Trade Log (persistent)

LAYER 6: RISK & MONITORING
├── Risk Manager
│   ├── Daily Loss Checker
│   ├── Margin Validator
│   ├── Concurrent Position Limiter
│   └── Circuit Breaker
├── Monitoring & Logging
│   ├── Structured JSON Logger
│   ├── Metrics Collector (P&L, latency, fills)
│   ├── Alert Router (email, dashboard, SMS)
│   └── Performance Dashboard
└── Backtest Engine
    ├── Vectorized Simulator
    ├── Fee/Slippage Model
    └── Walk-Forward Validator
```

### A2. File Structure & Module Organization

```
zerodha_kite_bot/
├── main.py                          # Entry point: async event loop orchestrator
├── config/
│   ├── config.yaml                  # User configuration (strategies, capital, etc.)
│   ├── default_config.yaml          # Default safe values
│   └── secrets.env                  # API keys (git-ignored)
├── data/
│   ├── data_ingester.py             # Kite WebSocket + historical data loader
│   ├── feature_store.py             # Real-time feature computation
│   ├── data_validator.py            # Data quality checks
│   └── storage/                     # Parquet files, logs
│       ├── historical_candles/
│       ├── option_chains/
│       └── audit_logs/
├── models/
│   ├── signal_model.py              # XGBoost classifier + regressor
│   ├── model_trainer.py             # Training pipeline
│   └── model_utils.py               # Feature importance, SHAP
├── strategies/
│   ├── strategy_base.py             # Abstract base class
│   ├── f_and_o_strategy.py          # Options strategy (long CE/PE)
│   ├── equity_strategy.py           # Stock strategy (swing/intraday)
│   └── signal_aggregator.py         # Combine signals, apply filters
├── execution/
│   ├── kite_mcp_client.py           # Kite OAuth + API wrapper
│   ├── order_executor.py            # Order construction, placement, reconciliation
│   ├── position_manager.py          # Maintain open positions
│   └── fill_reconciler.py           # Track fills, slippage model
├── risk/
│   ├── risk_manager.py              # Daily loss, margin, concurrent checks
│   ├── position_sizer.py            # Risk-based sizing
│   ├── expected_profit_calc.py      # Fee-aware profit estimation
│   └── circuit_breaker.py           # Pause trading on anomalies
├── monitoring/
│   ├── logger_config.py             # Structured JSON logging
│   ├── metrics_collector.py         # Real-time metrics (P&L, latency)
│   ├── alert_manager.py             # Email, dashboard alerts
│   └── dashboards/
│       ├── realtime_dashboard.html  # Live P&L, positions
│       └── backtest_report.html     # Historical performance
├── backtest/
│   ├── backtest_engine.py           # Vectorized simulator
│   ├── fee_slippage_model.py        # Commission + slippage
│   └── walk_forward.py              # Time-series CV
├── tests/
│   ├── test_strike_selection.py
│   ├── test_position_sizing.py
│   ├── test_expected_profit.py
│   ├── test_order_executor.py
│   ├── test_risk_manager.py
│   └── test_integration.py
├── ui/
│   ├── vs_code_panel.py             # VS Code Webview integration
│   ├── trade_preview.html           # Trade ticket preview
│   └── dashboard.html               # Monitoring dashboard
├── utils/
│   ├── config_loader.py             # YAML config parser + validation
│   ├── timezone_utils.py            # IST handling
│   ├── db_utils.py                  # Parquet, InfluxDB connectors
│   └── notification_utils.py        # Email, SMS senders
├── requirements.txt                 # Dependencies
├── README.md                        # Usage guide
├── ARCHITECTURE.md                  # This file
└── docker-compose.yml               # Optional: containerize MCP server
```

### A3. Core Class Definitions & Interfaces

#### A3.1 KiteMCPClient Wrapper

```python
class KiteMCPClient:
    """
    Wrapper around Zerodha Kite Connect API with MCP integration for VS Code.
    Handles OAuth, token refresh, WebSocket subscription, and REST calls.
    """
    
    def __init__(self, api_key: str, oauth_redirect_uri: str, config: dict):
        self.api_key = api_key
        self.oauth_redirect_uri = oauth_redirect_uri
        self.config = config
        self.access_token = None
        self.user_id = None
        self.websocket = None
        self.request_token = None
    
    # OAuth Flow
    async def initiate_oauth_flow(self) -> str:
        """Return login URL for user to authorize."""
    
    async def exchange_token(self, request_token: str) -> dict:
        """Exchange request token for access token."""
    
    async def refresh_access_token(self) -> None:
        """Refresh expired access token."""
    
    # Market Data (REST)
    async def get_quote(self, exchange: str, symbol: str) -> dict:
        """Fetch bid/ask/last price."""
    
    async def get_historical_data(
        self, exchange: str, symbol: str, interval: str, 
        from_date: str, to_date: str
    ) -> list:
        """Fetch historical candles (1m, 5m, 15m, 1d)."""
    
    async def get_option_chain(self, symbol: str, expiry: str) -> dict:
        """Fetch option chain (all strikes, IV, OI, Greeks)."""
    
    # Subscriptions (WebSocket)
    async def subscribe_candles(self, symbols: list, interval: str) -> None:
        """Subscribe to real-time candles."""
    
    async def subscribe_ticks(self, symbols: list) -> None:
        """Subscribe to tick-by-tick data."""
    
    async def unsubscribe(self, symbols: list) -> None:
        """Unsubscribe from updates."""
    
    # Orders (REST)
    async def place_order(self, order_dict: dict) -> dict:
        """Place order; return order_id."""
    
    async def modify_order(self, order_id: str, **kwargs) -> dict:
        """Modify existing order."""
    
    async def cancel_order(self, order_id: str) -> dict:
        """Cancel order."""
    
    async def get_order_history(self, order_id: str) -> dict:
        """Get order status and fills."""
    
    async def get_trades_today(self) -> list:
        """Get executed trades for the day."""
    
    # Account
    async def get_account_info(self) -> dict:
        """Get account status, equity, margin."""
    
    async def get_holdings(self) -> list:
        """Get current holdings (T+1, delivery positions)."""
    
    async def get_positions(self) -> list:
        """Get intraday positions (MIS, NRML)."""
    
    async def get_margins(self) -> dict:
        """Get margin availability breakdown."""
```

#### A3.2 StrategyBase (Abstract)

```python
class StrategyBase(ABC):
    """
    Abstract base class for all strategies (F&O, Equity).
    Enforces consistent interface: entry rules, exit rules, sizing.
    """
    
    def __init__(self, name: str, config: dict, signal_model: 'SignalModel'):
        self.name = name
        self.config = config
        self.signal_model = signal_model
    
    @abstractmethod
    def generate_signals(self, market_data: dict, account_state: dict) -> list:
        """
        Generate list of signal dicts.
        Each signal: {symbol, type, entry_price, stop_loss_price, 
                      expected_return_pct, confidence, reason}
        """
        pass
    
    @abstractmethod
    def apply_entry_filters(self, signals: list, account_state: dict) -> list:
        """Filter signals by liquidity, capital, confidence thresholds."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: dict, account_state: dict) -> int:
        """Return number of units (lots or shares) to trade."""
        pass
    
    @abstractmethod
    def get_exit_conditions(self, position: dict, market_data: dict) -> dict:
        """Return {'should_exit': bool, 'reason': str, 'exit_price': float}"""
        pass
```

#### A3.3 F&OStrategy (Long Options Only)

```python
class F_and_OStrategy(StrategyBase):
    """
    F&O strategy: long call (CE) and long put (PE) only.
    No option selling; strictly capital preservation + high-edge trades.
    """
    
    def __init__(self, config: dict, signal_model: 'SignalModel', kite_client: 'KiteMCPClient'):
        super().__init__("F&O_LongOptions", config, signal_model)
        self.kite_client = kite_client
        self.option_chain_cache = {}
        self.iv_history = {}
    
    def generate_signals(self, market_data: dict, account_state: dict) -> list:
        """
        1. Fetch NIFTY price & option chain
        2. Compute momentum/mean-reversion on 1m, 5m, 15m
        3. Classify: CALL (bullish) or PUT (bearish) or SKIP
        4. Select strike (liquid, delta 0.20-0.40 preferred)
        5. Return signal list
        """
        signals = []
        
        # Intraday entry
        if self._macd_above_signal_line(market_data['1m']):
            if market_data['1m']['rsi'] < 70:
                call_strike = self._select_call_strike(market_data, account_state)
                if call_strike:
                    signal = self._build_signal(
                        symbol=f"NIFTY{expiry}C{call_strike}",
                        signal_type="CALL",
                        entry_price=market_data[call_strike]['bid'],
                        stop_loss_price=None,  # % based later
                        reason="MACD bullish cross"
                    )
                    signals.append(signal)
        
        # Similar for PUT signals...
        
        return signals
    
    def _select_call_strike(self, market_data: dict, account_state: dict) -> int:
        """Detailed strike selection logic (see pseudo-code section)."""
        pass
    
    def apply_entry_filters(self, signals: list, account_state: dict) -> list:
        """
        Filter: confidence >= 0.80, expected_profit >= 150 INR, 
        cool-down satisfied, concurrent limit.
        """
        filtered = []
        for signal in signals:
            # 1. Confidence check
            if signal['confidence'] < self.config['min_confidence']:
                continue
            
            # 2. Expected profit check
            profit_calc = calculate_expected_net_profit(signal, account_state, self.config)
            if profit_calc['net_profit_expected'] < self.config['min_expected_net_profit_inr']:
                continue
            
            # 3. Cool-down check (48 hours between separate F&O trades)
            if not self._cool_down_satisfied():
                continue
            
            # 4. Concurrent position check
            if len(account_state['open_f_and_o_positions']) >= self.config['max_concurrent_positions']:
                continue
            
            filtered.append(signal)
        
        return filtered
    
    def calculate_position_size(self, signal: dict, account_state: dict) -> int:
        """Size position using risk-based approach, respecting lot size constraints."""
        pass
    
    def get_exit_conditions(self, position: dict, market_data: dict) -> dict:
        """
        Evaluate: TP (50% premium gain), SL (40% loss), time-based (< 1 day to expiry).
        """
        pass
```

#### A3.4 OrderExecutor

```python
class OrderExecutor:
    """
    Constructs, validates, places, and reconciles orders via Kite MCP.
    Maintains audit trail, handles retries, tracks fills and slippage.
    """
    
    def __init__(self, kite_client: 'KiteMCPClient', config: dict, logger: 'Logger'):
        self.kite_client = kite_client
        self.config = config
        self.logger = logger
        self.pending_orders = {}  # {order_id: {signal_info, expected_price, ...}}
        self.fill_log = []  # For slippage recalibration
    
    async def place_and_reconcile_order(
        self,
        signal: dict,
        position_size: int,
        user_id: str,
        user_confirm: bool = True
    ) -> dict:
        """
        Full cycle: construct → validate → [user confirm] → place → reconcile fill.
        Returns: {'success': bool, 'order_id': str, 'fill_price': float, 'slippage_pct': float, ...}
        """
        
        # 1. Construct order dict
        order = self._construct_order(signal, position_size, user_id)
        
        # 2. Validate
        validation = self._validate_order(order)
        if not validation['valid']:
            self.logger.error(f"Order validation failed: {validation['error']}")
            return {'success': False, 'reason': validation['error']}
        
        # 3. User confirmation (semi-automated mode)
        if user_confirm:
            preview = self._generate_order_preview(order, signal)
            confirmed = await self._prompt_user_confirmation(preview)
            if not confirmed:
                self.logger.info(f"Order rejected by user: {order['tag']}")
                return {'success': False, 'reason': 'USER_REJECTED'}
        
        # 4. Place order with retry logic
        result = await self._place_order_with_retries(order)
        if not result['success']:
            return result
        
        order_id = result['order_id']
        self.pending_orders[order_id] = {
            'signal': signal,
            'expected_price': signal['entry_price'],
            'order_dict': order,
            'placed_time': datetime.now()
        }
        
        # 5. Reconcile fill
        fill_result = await self._track_and_reconcile_fill(order_id)
        
        # Log to audit
        self.logger.info(f"Order {order_id} completed: {fill_result}")
        
        return fill_result
    
    def _construct_order(self, signal: dict, position_size: int, user_id: str) -> dict:
        """Build order dict for Kite MCP API."""
        return {
            'variety': 'regular',
            'exchange': signal['exchange'],  # 'NFO' for F&O, 'NSE' for equity
            'tradingsymbol': signal['symbol'],
            'transaction_type': 'BUY',
            'quantity': position_size,
            'price': signal['entry_price'],
            'order_type': 'LIMIT',
            'product': 'MIS' if signal['holding_period'] == 'intraday' else 'NRML',
            'validity': 'DAY',
            'tag': f"signal_{signal['signal_id']}_user_{user_id}_time_{datetime.now().isoformat()}"
        }
    
    def _validate_order(self, order: dict) -> dict:
        """Pre-flight checks: margin, lot size, price sanity."""
        pass
    
    async def _place_order_with_retries(self, order: dict, max_retries: int = 3) -> dict:
        """Place order with exponential backoff on failure."""
        pass
    
    async def _track_and_reconcile_fill(self, order_id: str) -> dict:
        """Poll order status; return when filled; calculate slippage."""
        pass
```

#### A3.5 RiskManager

```python
class RiskManager:
    """
    Gate keeper: check if trading is allowed given current account state.
    Implements circuit breaker: daily loss limit, margin checks, latency, exceptions.
    """
    
    def __init__(self, config: dict, logger: 'Logger'):
        self.config = config
        self.logger = logger
        self.circuit_breaker_active = False
        self.circuit_break_reason = None
    
    async def check_trading_allowed(self, account_state: dict) -> tuple:
        """
        Returns: (allowed: bool, reason: str)
        Reasons: 'OK', 'DAILY_LOSS_LIMIT', 'MARGIN_INSUFFICIENT', 'HIGH_LATENCY', 
                 'TOO_MANY_EXCEPTIONS', 'MARKET_CLOSED', ...
        """
        
        checks = []
        
        # 1. Daily loss limit
        if account_state['daily_pnl'] < -self.config['daily_loss_limit_inr']:
            self.logger.critical(f"Daily loss limit breached: {account_state['daily_pnl']} INR")
            self.circuit_breaker_active = True
            self.circuit_break_reason = "DAILY_LOSS_LIMIT"
            return False, "DAILY_LOSS_LIMIT"
        
        # 2. Margin check
        if account_state['available_margin'] < self.config['min_margin_buffer_inr']:
            self.logger.warning("Margin insufficient for new trades")
            return False, "MARGIN_INSUFFICIENT"
        
        # 3. Concurrent position limit (soft gate, not circuit break)
        if len(account_state['open_positions']) >= self.config['max_concurrent_positions']:
            self.logger.debug("Max concurrent positions reached")
            return False, "MAX_CONCURRENT_REACHED"
        
        # 4. API latency (hard gate: if > 500ms, pause)
        if account_state['api_latency_ms'] > self.config['max_api_latency_ms']:
            self.logger.error(f"API latency {account_state['api_latency_ms']}ms exceeds threshold")
            return False, "HIGH_LATENCY"
        
        # 5. System health (exceptions)
        if account_state['exceptions_last_minute'] > self.config['max_exceptions_threshold']:
            self.logger.critical(f"Too many exceptions: {account_state['exceptions_last_minute']}")
            self.circuit_breaker_active = True
            return False, "TOO_MANY_EXCEPTIONS"
        
        # 6. Market hours
        if not self._is_market_open():
            return False, "MARKET_CLOSED"
        
        return True, "OK"
    
    def validate_position_risk(self, signal: dict, account_state: dict) -> tuple:
        """
        Per-trade risk checks: capital at risk, per-symbol exposure, portfolio notional.
        Returns: (valid: bool, error_message: str)
        """
        pass
    
    def _is_market_open(self) -> bool:
        """NSE market open: 9:15 AM - 3:30 PM IST on weekdays."""
        pass
```

#### A3.6 SignalModel (ML Layer)

```python
class SignalModel:
    """
    ML-based signal generation: XGBoost classifier (p_win) + regressor (expected_return).
    Also wraps rule-based pre-filters (liquidity, delta, IV).
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.classifier = None  # XGBoost fitted model
        self.regressor = None
        self.feature_names = None
        self.scaler = StandardScaler()
    
    def train(self, training_data: pd.DataFrame, target: str = 'trade_profitable') -> dict:
        """
        Train classifier and regressor on historical backtest-derived labels.
        
        training_data: rows = historical candles with features + label
        Returns: {accuracy, f1_score, feature_importance, ...}
        """
        
        # Prepare features
        feature_cols = [c for c in training_data.columns if c.startswith('feat_')]
        X = training_data[feature_cols].values
        y_classifier = (training_data['net_pnl'] > 0).astype(int).values
        y_regressor = training_data['expected_return_pct'].values
        
        # Train classifier
        self.classifier = XGBClassifier(
            max_depth=self.config['model']['classifier']['hyperparams']['max_depth'],
            learning_rate=self.config['model']['classifier']['hyperparams']['learning_rate'],
            n_estimators=self.config['model']['classifier']['hyperparams']['n_estimators'],
            random_state=42
        )
        self.classifier.fit(X, y_classifier)
        
        # Train regressor
        self.regressor = XGBRegressor(
            max_depth=self.config['model']['regressor']['hyperparams']['max_depth'],
            learning_rate=self.config['model']['regressor']['hyperparams']['learning_rate'],
            n_estimators=self.config['model']['regressor']['hyperparams']['n_estimators'],
            random_state=42
        )
        self.regressor.fit(X, y_regressor)
        
        # Evaluate
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y_classifier, y_pred)
        
        return {
            'classifier_accuracy': accuracy,
            'feature_importance': self.classifier.feature_importances_.tolist(),
            'regressor_r2': self.regressor.score(X, y_regressor)
        }
    
    def predict(self, features_df: pd.DataFrame) -> dict:
        """
        Predict for new data.
        Returns: {p_win: float [0, 1], expected_return_pct: float, confidence_score: float}
        """
        
        feature_cols = [c for c in features_df.columns if c.startswith('feat_')]
        X = features_df[feature_cols].values
        
        p_win = self.classifier.predict_proba(X)[:, 1]
        expected_return = self.regressor.predict(X)
        
        return {
            'p_win': p_win[0],
            'expected_return_pct': expected_return[0],
            'confidence_score': max(p_win[0], 1 - p_win[0])  # Confidence in prediction direction
        }
```

---

## PART B: DATA FLOW & EXECUTION SEQUENCES

### B1. Real-Time Trading Loop (Async Event Orchestration)

```python
async def trading_loop_orchestrator(
    kite_client: KiteMCPClient,
    strategies: list,  # [F&OStrategy, EquityStrategy]
    signal_model: SignalModel,
    executor: OrderExecutor,
    risk_mgr: RiskManager,
    config: dict,
    logger: Logger
):
    """
    Main async loop: runs every 60 seconds (configurable).
    1. Fetch latest market data
    2. Compute features
    3. Generate signals per strategy
    4. Aggregate and de-duplicate
    5. Risk gates
    6. Preview and execute (semi-auto)
    7. Monitor fills and P&L
    8. Log metrics
    """
    
    account_state = await kite_client.get_account_info()
    
    while True:
        try:
            # 1. Fetch data
            market_data = await _fetch_latest_market_data(kite_client)
            account_state = await kite_client.get_account_info()
            
            # 2. Check circuit breaker
            allowed, reason = await risk_mgr.check_trading_allowed(account_state)
            if not allowed:
                logger.warning(f"Trading paused: {reason}")
                await asyncio.sleep(config['loop_interval_seconds'])
                continue
            
            # 3. Generate signals per strategy
            all_signals = []
            for strategy in strategies:
                signals = strategy.generate_signals(market_data, account_state)
                all_signals.extend(signals)
            
            if not all_signals:
                logger.debug("No signals generated this cycle")
                await asyncio.sleep(config['loop_interval_seconds'])
                continue
            
            # 4. Filter signals (confidence, expected profit, cool-down)
            filtered_signals = _filter_and_rank_signals(all_signals, account_state, config)
            
            # 5. Risk validation per signal
            approved_signals = []
            for signal in filtered_signals:
                valid, error = risk_mgr.validate_position_risk(signal, account_state)
                if valid:
                    approved_signals.append(signal)
                else:
                    logger.debug(f"Signal rejected by risk check: {error}")
            
            # 6. Execute trades (semi-automated)
            for signal in approved_signals:
                strategy = _get_strategy_for_signal(signal, strategies)
                pos_size = strategy.calculate_position_size(signal, account_state)
                
                # Placement (with user confirm in semi-auto mode)
                result = await executor.place_and_reconcile_order(
                    signal=signal,
                    position_size=pos_size,
                    user_id=account_state['user_id'],
                    user_confirm=config['ui']['mode'] == 'SEMI_AUTOMATED'
                )
                
                if result['success']:
                    logger.info(f"Order executed: {result['order_id']}")
                    account_state['recent_orders'].append(result)
                else:
                    logger.warning(f"Order failed: {result['reason']}")
            
            # 7. Check exits for open positions
            for position in account_state['open_positions']:
                strategy = _get_strategy_for_position(position, strategies)
                exit_cond = strategy.get_exit_conditions(position, market_data)
                
                if exit_cond['should_exit']:
                    exit_result = await executor.place_and_reconcile_order(
                        signal={'symbol': position['symbol'], 'transaction_type': 'SELL', ...},
                        position_size=position['quantity'],
                        user_id=account_state['user_id'],
                        user_confirm=False  # Auto-exit
                    )
                    logger.info(f"Position exited: {exit_result['reason']}")
            
            # 8. Log metrics
            await _update_monitoring_dashboard(account_state, logger)
            
        except Exception as e:
            logger.error(f"Orchestrator exception: {e}", exc_info=True)
            account_state['exceptions_last_minute'] += 1
        
        # Sleep before next cycle
        await asyncio.sleep(config['loop_interval_seconds'])
```

### B2. Order Placement & Fill Reconciliation Sequence

```
User/Signal → Order Preview (UI) → User Confirm → API Call → Order ID
                                                          ↓
                                                    Poll Status (every 5s)
                                                    COMPLETE? → Calculate Slippage
                                                    │         → Log Fill
                                                    REJECTED  → Alert & Retry
                                                    TIMEOUT   → Cancel & Escalate
```

### B3. Position Lifecycle

```
ENTRY:  Signal → Size → Validate → Place → Fill → Add to Portfolio
HOLD:   Monitor Market → Check Exits (TP, SL, Time) → Update P&L
EXIT:   Trigger Condition → Place Exit Order → Fill → Close Position
```

---

## PART C: DATA MODELS & SCHEMAS

### C1. Signal Schema

```python
@dataclass
class Signal:
    signal_id: str  # Unique identifier
    symbol: str  # "NIFTYNOV2025C23500" or "INFY"
    exchange: str  # "NFO" or "NSE"
    signal_type: str  # "CALL", "PUT", "BUY", "SELL"
    entry_price: float
    stop_loss_price: float
    target_profit_pct: float  # Or fixed INR amount
    expected_return_pct: float  # ML regressor output
    confidence: float  # Classifier probability
    reason: str  # Human-readable explanation
    timestamp: datetime
    holding_period: str  # "intraday", "swing"
    strategy_source: str  # "F&O", "Equity"
    model_features: dict  # {momentum_score, volatility, etc.}
    audit_trail: str  # "MACD>Signal, RSI<70, P(win)=0.85"
```

### C2. Position Schema

```python
@dataclass
class Position:
    position_id: str
    symbol: str
    exchange: str
    quantity: int
    entry_price: float
    entry_time: datetime
    entry_order_id: str
    entry_fee: float
    entry_slippage: float
    
    current_price: float  # Mark-to-market
    current_time: datetime
    unrealized_pnl: float
    unrealized_pnl_pct: float
    
    stop_loss_price: float
    take_profit_price: float
    trailing_stop: bool
    
    position_type: str  # "LONG", "SHORT"
    product: str  # "MIS", "NRML", "CNC"
    
    exit_signal: Optional[dict]  # When to exit
    exit_time: Optional[datetime]
    exit_order_id: Optional[str]
    exit_price: Optional[float]
    exit_fee: Optional[float]
    exit_slippage: Optional[float]
    
    realized_pnl: Optional[float]
    trade_duration_minutes: Optional[int]
```

### C3. Order Execution Schema

```python
@dataclass
class OrderExecution:
    order_id: str
    signal_id: str
    symbol: str
    transaction_type: str  # "BUY", "SELL"
    quantity: int
    order_price: float  # LIMIT price placed
    order_type: str  # "LIMIT", "MARKET", "SL-M"
    
    placed_time: datetime
    placed_by: str  # User ID
    user_confirmed: bool
    
    # Fill details
    fill_time: Optional[datetime]
    filled_quantity: int
    average_fill_price: float
    slippage: float  # fill_price - expected_price
    slippage_pct: float
    
    # Status tracking
    status: str  # "PENDING", "COMPLETE", "REJECTED", "CANCELLED"
    reject_reason: Optional[str]
    
    # Audit
    api_latency_ms: float
    audit_log_entry: str
```

### C4. Feature Schema (for ML)

```python
@dataclass
class FeatureVector:
    timestamp: datetime
    symbol: str
    
    # Momentum
    ema_12: float
    ema_26: float
    macd: float
    macd_signal: float
    rsi_14: float
    adx_14: float
    
    # Mean-reversion
    bb_position: float  # (price - lower_band) / (upper - lower)
    z_score_20: float
    roc_20: float
    
    # Volatility
    atr_14: float
    std_dev_20: float
    historical_vol_20: float
    iv_from_options: Optional[float]
    iv_rank: Optional[float]
    iv_skew: Optional[float]
    
    # Microstructure
    ob_imbalance: float
    trade_flow_5m: float
    bid_ask_spread_pct: float
    
    # Option-specific (if applicable)
    delta: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    time_value: Optional[float]
    oi_trend: Optional[float]
```

---

## PART D: API INTEGRATION SPECIFICS

### D1. Kite MCP OAuth Flow (VS Code)

```
1. User clicks "Connect to Zerodha" in VS Code extension
2. VS Code opens browser → Kite OAuth login page
3. User enters credentials → Kite redirects to localhost:8000/callback?request_token=xxxxx
4. Local server exchanges request_token for access_token
5. Token stored securely (session-based, short-lived)
6. WebSocket reconnects with access_token
7. UI confirms: "Connected to Zerodha Kite. Account: xxxxx. Equity: ₹10,000"
```

**Code:**

```python
async def oauth_callback_handler(request_token: str) -> dict:
    """Zerodha OAuth callback handler."""
    
    response = kite_client.generate_session(
        api_key=config['kite_api_key'],
        request_token=request_token,
        secret=config['kite_secret']
    )
    
    access_token = response['access_token']
    user_id = response['user_id']
    
    # Store securely (e.g., encrypted cookie or secure storage)
    session['access_token'] = access_token
    session['user_id'] = user_id
    
    return {'success': True, 'user_id': user_id, 'token_expiry': '24 hours'}
```

### D2. WebSocket Subscription for Real-Time Data

```python
async def subscribe_to_live_data(kite_client: KiteMCPClient, symbols: list) -> None:
    """Subscribe to tick data via WebSocket for low-latency feature updates."""
    
    def on_tick(ticks):
        """Called every tick."""
        for tick in ticks:
            symbol = tick['instrument_token']
            # Update real-time cache
            feature_store.update_tick(symbol, {
                'bid': tick['bid'],
                'ask': tick['ask'],
                'last_price': tick['last_price'],
                'volume': tick['volume'],
                'timestamp': tick['timestamp']
            })
    
    def on_connect():
        logger.info("WebSocket connected")
    
    def on_close():
        logger.warning("WebSocket disconnected; attempting reconnect...")
        # Implement exponential backoff reconnect
    
    # Set up websocket
    kite_client.websocket.on_tick = on_tick
    kite_client.websocket.on_connect = on_connect
    kite_client.websocket.on_close = on_close
    
    # Subscribe to symbols
    kite_client.websocket.subscribe(symbols)
    
    # Keep running
    await asyncio.Event().wait()
```

### D3. Order Placement via REST API

```python
async def place_order_kite(
    kite_client: KiteMCPClient,
    symbol: str,
    exchange: str,
    qty: int,
    price: float,
    order_type: str = 'LIMIT',
    transaction_type: str = 'BUY',
    product: str = 'MIS',
    tag: str = ''
) -> dict:
    """Place order via Kite Connect REST API."""
    
    response = await kite_client.place_order(
        variety='regular',
        exchange=exchange,
        tradingsymbol=symbol,
        transaction_type=transaction_type,
        quantity=qty,
        price=price,
        order_type=order_type,
        product=product,
        validity='DAY',
        tag=tag
    )
    
    return {
        'order_id': response['order_id'],
        'status': 'PLACED',
        'timestamp': datetime.now()
    }
```

---

## PART E: PERFORMANCE & OPTIMIZATION CONSIDERATIONS

### E1. Latency Budget

| Component | Budget (ms) | Notes |
|-----------|------------|-------|
| Market data fetch (WebSocket tick) | 10 | Real-time |
| Feature computation | 50 | 20 indicators in-memory |
| Model inference (XGBoost predict) | 5 | Fast tree-based model |
| Strategy signal generation | 20 | Rule-based filters |
| Risk checks | 10 | Cache account state |
| Order construction & validation | 20 | Local checks |
| API call (place order) | 200 | Network latency |
| **Total E2E Target** | **315 ms** | < 500 ms alert threshold |

### E2. Memory Optimization

- **Candle data:** Cache only last 100 candles per symbol in memory; older data on disk (Parquet)
- **Option chain:** Update snapshot every 5m; keep only current expiry + next expiry
- **Feature store:** Rolling window (last 500 ticks); auto-evict older entries
- **Model weights:** Load once at startup; keep in memory

### E3. Database Schema (Parquet for Historical Data)

```
historical_candles/
├── symbol=NIFTY/date=2025-11-05/
│   └── candles_NIFTY_2025-11-05.parquet
├── symbol=INFY/date=2025-11-05/
│   └── candles_INFY_2025-11-05.parquet
└── ...

option_chain/
├── symbol=NIFTYNOV2025/date=2025-11-05/
│   └── chain_NIFTYNOV2025_2025-11-05.parquet
└── ...

audit_logs/
├── 2025-11-05/
│   └── audit_2025-11-05_0900_1630.json (daily rolling log)
└── ...

trade_logs/
├── trades_2025-11-05.parquet
└── ...
```

---

## PART F: DEPLOYMENT & CONTAINERIZATION (OPTIONAL)

### F1. Docker Setup for MCP Server (Optional; can use hosted mcp.kite.trade)

```dockerfile
# Dockerfile for local MCP server
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.9'
services:
  mcp_server:
    build: .
    ports:
      - "8000:8000"
    environment:
      - KITE_API_KEY=${KITE_API_KEY}
      - KITE_SECRET=${KITE_SECRET}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
```

---

## PART G: TESTING FRAMEWORK

### G1. Unit Tests Example

```python
# test_position_sizer.py
import pytest
from risk.position_sizer import PositionSizer

def test_position_size_calculation():
    sizer = PositionSizer(capital=10000, max_risk_per_trade=0.05)
    
    signal = {
        'symbol': 'NIFTYNOV2025C23500',
        'entry_price': 250,
        'stop_loss_price': 200
    }
    
    size = sizer.calculate_position_size(signal)
    assert size == 2  # 2 lots of 75 = 150 qty; expected_loss = (250-200)*150 = 7500, but max allowed = 500 INR
    # This should be adjusted...
```

### G2. Integration Test: Full Trade Cycle

```python
# test_integration_full_trade_cycle.py
@pytest.mark.asyncio
async def test_full_trade_cycle_paper_mode(paper_trading_env):
    """Full E2E test: signal → order → fill → P&L."""
    
    # 1. Inject a mock signal
    signal = Signal(...)
    
    # 2. Execute via paper trading (simulator)
    result = await paper_trading_env.place_and_reconcile_order(signal, pos_size=1)
    
    # 3. Assert
    assert result['success'] == True
    assert result['fill_price'] > 0
    assert result['slippage_pct'] < 0.01
```

---

## PART H: CONFIGURATION VALIDATION & DEFAULTS

```python
# utils/config_loader.py
from pydantic import BaseModel, validator

class CapitalConfig(BaseModel):
    starting_capital_inr: int
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    
    @validator('starting_capital_inr')
    def capital_positive(cls, v):
        assert v > 0, "Capital must be positive"
        return v

class BotConfig(BaseModel):
    capital: CapitalConfig
    # ... more sections
    
    @classmethod
    def from_yaml(cls, filepath: str):
        with open(filepath) as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Usage
config = BotConfig.from_yaml('config/config.yaml')
# Validation errors raised if config invalid
```

---

## CONCLUSION

This technical specification provides the detailed component architecture, data models, API integration points, and implementation patterns needed to build the Zerodha Kite MCP AI trading bot. Each component is designed to be:

- **Modular:** Swap implementations (e.g., strategy, model) without breaking others
- **Testable:** Dependency injection enables unit + integration testing
- **Observable:** Comprehensive logging and metrics collection
- **Safe:** Risk management gates at multiple levels
- **Extensible:** Easy to add new strategies, models, or data sources

**Next Steps:** Convert this spec into actual code, starting with the data ingester and backtest engine, then integrate MCP, and validate with paper trading.

