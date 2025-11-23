# Zerodha Kite MCP AI Trading Bot

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-MVP%20Development-yellow.svg)

Semi-automated, fee-conscious, ML-driven trading bot for Zerodha Kite with strict risk management and manual confirmation for all trades.

---

## 📋 Project Overview

This trading bot implements a comprehensive algorithmic trading system with:

- **Semi-Automated Trading**: Manual confirmation required for all trades via VS Code UI
- **ML-Driven Signals**: XGBoost models for win probability and expected return prediction
- **Multi-Strategy Support**: F&O (long options) and equity (swing/intraday) strategies
- **Strict Risk Management**: Daily loss limits, circuit breakers, position sizing
- **Fee-Conscious Design**: Only trades with expected profit ≥ 150 INR after fees
- **Comprehensive Backtesting**: Vectorized engine with walk-forward validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              VS Code UI (Manual Override)           │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│         Trading Orchestrator (main.py)              │
├─────────────────────────────────────────────────────┤
│  • Signal Aggregation  • Risk Checks  • Execution   │
└────┬──────────┬──────────┬─────────────┬───────────┘
     │          │          │             │
┌────▼────┐ ┌──▼──────┐ ┌─▼────────┐ ┌─▼──────────┐
│Strategy │ │ML Models│ │  Risk    │ │ Execution  │
│ Engine  │ │(XGBoost)│ │ Manager  │ │  Engine    │
└────┬────┘ └──┬──────┘ └─┬────────┘ └─┬──────────┘
     │          │          │             │
┌────▼──────────▼──────────▼─────────────▼───────────┐
│          Data Layer (Historical + Live)             │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│          Zerodha Kite Connect API                   │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Zerodha Kite Connect API credentials
- Windows/Linux/macOS with VS Code

### Installation

1. **Clone the repository**
```powershell
cd c:\Users\mohdm\Downloads\Zerodha\TradingBot
```

2. **Create virtual environment**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

4. **Configure environment**
```powershell
copy .env.example .env
# Edit .env with your Kite API credentials
```

5. **Set up Kite API credentials**

- Register app at: https://developers.kite.trade/
- Get API Key and Secret
- Update `.env` file:
```
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_USER_ID=your_user_id
```

---

## 📁 Project Structure

```
TradingBot/
├── config/                    # Configuration files
│   ├── config.yaml            # Main configuration (200+ parameters)
│   └── config_loader.py       # Config validation with Pydantic
│
├── data/                      # Data ingestion & processing
│   ├── loader.py              # Historical & live data loading
│   ├── feature_computer.py    # 50+ technical indicators
│   └── storage.py             # Parquet storage handler
│
├── execution/                 # Order execution
│   ├── kite_client.py         # Kite Connect API wrapper
│   └── order_executor.py      # Order placement & reconciliation
│
├── models/                    # ML models
│   ├── signal_model.py        # XGBoost classifier & regressor
│   └── training_pipeline.py   # Model training workflow
│
├── strategies/                # Trading strategies
│   ├── fo_strategy.py         # F&O strategy (long options)
│   ├── equity_strategy.py     # Equity swing/intraday
│   └── base_strategy.py       # Abstract base class
│
├── risk/                      # Risk management
│   └── risk_manager.py        # Circuit breaker, limits, cool-down
│
├── monitoring/                # Logging & monitoring
│   ├── logger.py              # Audit logging (JSON)
│   └── dashboard.py           # P&L tracking
│
├── backtest/                  # Backtesting framework
│   ├── engine.py              # Vectorized backtest engine
│   └── fee_model.py           # Brokerage fee calculation
│
├── orchestration/             # Main orchestrator
│   └── trading_orchestrator.py
│
├── tests/                     # Unit & integration tests
│
├── ui/                        # VS Code extension (TBD)
│
├── main.py                    # Entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## ⚙️ Configuration

The bot is highly configurable via `config/config.yaml`:

### Key Parameters

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Capital** | `starting_capital` | 10,000 INR | Initial trading capital |
| | `max_daily_loss_pct` | 7% | Daily loss circuit breaker |
| **F&O Strategy** | `min_delta` | 0.20 | Minimum option delta |
| | `max_delta` | 0.40 | Maximum option delta |
| | `take_profit_pct` | 50% | Exit at 50% premium gain |
| | `stop_loss_pct` | 60% | Exit at 60% premium loss |
| **ML Models** | `min_model_confidence` | 0.80 | Minimum p(win) threshold |
| | `target_accuracy` | 0.70 | Model training target |
| **Risk** | `max_concurrent_positions` | 3 | Max open positions |
| | `cool_down_hours_fo` | 48 | F&O trade cool-down |

See `config/config.yaml` for full configuration reference (200+ parameters).

---

## 🔧 Usage

### 1. Authentication

First run requires Kite Connect OAuth:

```powershell
python main.py
```

Follow the login URL, authorize the app, and paste the `request_token`.

### 2. Paper Trading (Recommended)

Start with paper trading to validate strategies:

```powershell
python main.py --paper
```

### 3. Backtesting

Run backtest on historical data:

```powershell
python -m backtest.engine --instrument NIFTY --days 90
```

### 4. Live Trading (After Validation)

⚠️ **Only after 30+ days of successful paper trading!**

Edit `config/config.yaml`:
```yaml
trading_rules:
  enable_live_trading: true
  require_manual_confirmation: true  # Keep this ON!
```

```powershell
python main.py
```

---

## 📊 Features Implemented

### ✅ Completed (MVP Phase 1)

- [x] Project structure and configuration system
- [x] Kite Connect API integration (OAuth, REST, WebSocket)
- [x] Data loader with Parquet storage
- [x] Feature computation (50+ technical indicators)
- [x] Risk manager with circuit breaker
- [x] Audit logging system
- [x] Basic orchestration framework

### 🚧 In Progress

- [ ] ML model training pipeline (XGBoost)
- [ ] F&O strategy implementation
- [ ] Equity strategy implementation
- [ ] Order executor with fill reconciliation
- [ ] Backtesting engine

### 📅 Planned

- [ ] VS Code UI extension
- [ ] Walk-forward validation
- [ ] Position tracking & P&L dashboard
- [ ] Alert system (email/SMS)
- [ ] Advanced slippage modeling

---

## 🧪 Testing

Run unit tests:

```powershell
pytest tests/
```

Run with coverage:

```powershell
pytest --cov=. --cov-report=html tests/
```

---

## 📈 Success Metrics

### Backtest Gates (Must Pass Before Paper Trading)

- Win rate ≥ 50%
- Sharpe ratio ≥ 0.80
- Max drawdown ≤ 15%
- Net profit > 200 INR (after fees)
- Minimum 30 trades

### Paper Trading Gates (Must Pass Before Live)

- Win rate ≥ 60%
- Sharpe ratio ≥ 1.0
- Max drawdown ≤ 10%
- Correlation to backtest ≥ 0.80
- 30-day validation period

---

## ⚠️ Risk Warnings

1. **This is experimental software** - Use at your own risk
2. **Start with small capital** - Validate thoroughly before scaling
3. **Never disable manual confirmation** in live trading
4. **Monitor daily** - Check logs and P&L regularly
5. **Respect loss limits** - Don't override circuit breakers

**Past performance does not guarantee future results.**

---

## 🛠️ Development Roadmap

### Week 1-2: Data & Backtesting
- [x] Data infrastructure
- [ ] Backtest engine with fee model
- [ ] Feature engineering validation

### Week 3-4: Strategies & ML
- [ ] F&O strategy rules
- [ ] XGBoost model training
- [ ] Walk-forward validation

### Week 5-6: Risk & Execution
- [x] Risk management system
- [ ] Order execution engine
- [ ] Position tracking

### Week 7-8: Testing & UI
- [ ] Comprehensive test suite
- [ ] VS Code UI extension
- [ ] 30-day paper trading

---

## 📚 Documentation

- **PRD**: See `zerodha-mcp-ai-trading-bot-prd.md` (2,227 lines)
- **Technical Spec**: See `technical-implementation-spec.md` (1,166 lines)
- **Implementation Plan**: See `mvp-implementation-checklist.md` (714 lines)
- **Revenue Calibration**: See `revenue-target-calibration.md` (664 lines)

---

## 📝 License

MIT License - See LICENSE file for details

---

## 🤝 Contributing

This is a personal project. Contributions welcome but please open an issue first to discuss changes.

---

## 📧 Support

For issues or questions:
1. Check documentation files in repository
2. Review audit logs in `logs/audit/`
3. Enable debug logging in `config/config.yaml`

---

## 🔗 Links

- [Kite Connect API Docs](https://kite.trade/docs/connect/v3/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [TA-Lib Manual](https://ta-lib.org/function.html)

---

**Built with ❤️ for algorithmic trading**

**Status**: MVP Development - Not Production Ready

**Last Updated**: November 23, 2025
