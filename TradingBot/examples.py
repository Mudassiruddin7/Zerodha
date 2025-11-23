"""
Quick start guide and example usage scripts.
"""

# Example 1: Train ML Models
"""
python train_models.py --instrument NIFTY --days 180 --type equity
"""

# Example 2: Run in Paper Trading Mode
"""
python main.py --paper
"""

# Example 3: Test Configuration
"""
from config.config_loader import ConfigLoader

config = ConfigLoader("config/config.yaml")
print(f"Starting capital: {config.get('capital.starting_capital')}")
print(f"Daily loss limit: {config.get('capital.max_daily_loss_pct')}%")
"""

# Example 4: Load Historical Data
"""
import asyncio
from config.config_loader import ConfigLoader
from execution.kite_client import KiteMCPClient
from data.loader import DataLoader
from datetime import datetime, timedelta

async def load_data():
    config = ConfigLoader("config/config.yaml")
    kite = KiteMCPClient(config)
    await kite.initialize()
    
    loader = DataLoader(config, kite)
    
    # Search for NIFTY
    instruments = await loader.search_instruments("NIFTY", exchange="NSE")
    token = instruments[0]["instrument_token"]
    
    # Load 90 days of data
    to_date = datetime.now()
    from_date = to_date - timedelta(days=90)
    
    df = await loader.load_historical_data(token, from_date, to_date)
    print(f"Loaded {len(df)} candles")
    print(df.head())

asyncio.run(load_data())
"""

# Example 5: Compute Features
"""
from data.feature_computer import FeatureComputer
from config.config_loader import ConfigLoader
import pandas as pd

config = ConfigLoader("config/config.yaml")
fc = FeatureComputer(config)

# Assuming df is a DataFrame with OHLCV data
df_with_features = fc.compute_all_features(df, instrument_type="equity")

# Get feature names
feature_names = fc.get_feature_names("equity")
print(f"Computed {len(feature_names)} features")
"""

# Example 6: Test Strategy Signal Generation
"""
import asyncio
from config.config_loader import ConfigLoader
from strategies.equity_strategy import EquityStrategy

async def test_strategy():
    config = ConfigLoader("config/config.yaml")
    strategy = EquityStrategy(config)
    
    market_data = {
        "tradingsymbol": "RELIANCE",
        "instrument_token": 738561,
        "last_price": 2500.0
    }
    
    features = {
        "ema_9": 2480,
        "ema_21": 2450,
        "rsi_14": 55,
        "macd": 5.2,
        "macd_signal": 4.8
    }
    
    ml_prediction = {
        "p_win": 0.78,
        "expected_return": 3.5
    }
    
    signals = await strategy.generate_signals(market_data, features, ml_prediction)
    
    for signal in signals:
        print(signal)

asyncio.run(test_strategy())
"""

# Example 7: Calculate Fees
"""
from execution.order_executor import OrderExecutor
from config.config_loader import ConfigLoader

config = ConfigLoader("config/config.yaml")
# Note: kite_client required but can be None for fee calculation

executor = OrderExecutor(config, None)
fees = executor._calculate_fees(trade_value=10000, strategy_type="equity")
print(f"Fees for 10,000 INR trade: {fees:.2f} INR")
"""

# Example 8: Risk Manager Check
"""
from config.config_loader import ConfigLoader
from risk.risk_manager import RiskManager

config = ConfigLoader("config/config.yaml")
rm = RiskManager(config)

# Check if can trade
can_trade = rm.can_trade("equity")
print(f"Can trade: {can_trade}")

# Simulate a losing trade
rm.update_pnl(-500)
print(f"Daily P&L: {rm.daily_pnl}")
print(f"Daily losses: {rm.daily_losses}")
"""

print(__doc__)
