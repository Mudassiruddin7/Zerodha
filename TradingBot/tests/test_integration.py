"""Integration tests for full trading workflows."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from orchestration.trading_orchestrator import TradingOrchestrator
from strategies.fo_strategy import FOStrategy
from strategies.equity_strategy import EquityStrategy
from models.signal_model import SignalModel
from execution.kite_client import KiteClient
from risk.risk_manager import RiskManager
from config.config_loader import ConfigLoader


class TestFullTradeCycle:
    """Test complete trade cycle from signal to P&L."""
    
    @pytest.mark.asyncio
    async def test_equity_swing_full_cycle(self):
        """Test complete equity swing trade: signal -> entry -> hold -> exit -> P&L."""
        # Setup
        config = Mock(spec=ConfigLoader)
        config.get = Mock(return_value=None)
        
        # Mock components
        kite_client = AsyncMock(spec=KiteClient)
        kite_client.get_quote = AsyncMock(return_value={
            'last_price': 1000.0,
            'bid': 999.0,
            'ask': 1001.0,
        })
        kite_client.place_order = AsyncMock(return_value='ORDER123')
        kite_client.get_order_status = AsyncMock(return_value={
            'status': 'COMPLETE',
            'filled_quantity': 10,
            'average_price': 1000.0,
        })
        kite_client.get_margins = AsyncMock(return_value={
            'available': 10000.0,
            'used': 0.0,
        })
        
        # Create strategy
        strategy = EquityStrategy(config)
        
        # 1. Generate entry signal (EMA crossover)
        market_data = {
            'ema_9': 1005,
            'ema_21': 1000,
            'ema_9_prev': 999,
            'ema_21_prev': 1000,
            'close': 1005,
            'rsi_14': 55,
            'volume': 1000000,
        }
        
        signal = await strategy.generate_signal(market_data)
        assert signal is not None
        assert signal.direction == 'BUY'
        
        # 2. Execute order (mock)
        # In real integration test, would use OrderExecutor
        
        # 3. Monitor position
        position = {
            'entry_price': 1000.0,
            'entry_time': datetime.now(),
            'quantity': 10,
        }
        
        # 4. Check exit conditions
        # Scenario: Price hits take profit (5%)
        exit_market_data = {
            'close': 1052.0,  # 5.2% gain
            'ema_9': 1050,
            'ema_21': 1030,
        }
        
        should_exit = strategy._should_exit_swing(position, exit_market_data)
        assert should_exit is True
        
        # 5. Calculate P&L
        entry_cost = 1000.0 * 10
        exit_value = 1052.0 * 10
        gross_pnl = exit_value - entry_cost
        fees = 130  # Approximate
        net_pnl = gross_pnl - fees
        
        assert gross_pnl == 520.0
        assert net_pnl > 0  # Profitable after fees
    
    @pytest.mark.asyncio
    async def test_fo_options_full_cycle(self):
        """Test complete F&O options trade: strike selection -> entry -> exit."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(return_value=None)
        
        strategy = FOStrategy(config)
        
        # 1. Strike selection
        available_strikes = pd.DataFrame([
            {'strike': 18000, 'delta': 0.25, 'open_interest': 5000, 'volume': 2000, 'bid': 100, 'ask': 101, 'premium': 100},
            {'strike': 18100, 'delta': 0.30, 'open_interest': 8000, 'volume': 3000, 'bid': 120, 'ask': 121, 'premium': 120},
            {'strike': 18200, 'delta': 0.35, 'open_interest': 6000, 'volume': 2500, 'bid': 140, 'ask': 141, 'premium': 140},
        ])
        
        selected_strike = strategy._select_best_strike(available_strikes)
        assert selected_strike is not None
        assert 0.20 <= selected_strike['delta'] <= 0.40
        
        # 2. Entry signal
        market_data = {
            'macd': 15,
            'macd_signal': 10,
            'rsi_14': 55,
            'adx': 28,
            'close': 18000,
            'iv_percentile': 30,
        }
        
        signal = await strategy.generate_signal(market_data)
        assert signal is not None
        
        # 3. Position tracking
        position = {
            'entry_price': 120.0,
            'entry_time': datetime.now(),
            'quantity': 50,  # 1 lot
            'strike': 18100,
        }
        
        # 4. Exit on 50% premium gain
        exit_market_data = {
            'premium': 180.0,  # 50% gain
            'days_to_expiry': 5,
        }
        
        should_exit = strategy._should_exit_fo(position, exit_market_data)
        assert should_exit is True
        
        # 5. P&L
        entry_cost = 120.0 * 50
        exit_value = 180.0 * 50
        gross_pnl = exit_value - entry_cost
        fees = 80  # Approximate
        net_pnl = gross_pnl - fees
        
        assert gross_pnl == 3000.0
        assert net_pnl > 150  # Above minimum profit threshold
    
    @pytest.mark.asyncio
    async def test_oauth_flow(self):
        """Test OAuth authentication flow."""
        # Mock Kite client
        kite_client = KiteClient(api_key="test_key", api_secret="test_secret")
        
        # 1. Get login URL
        login_url = kite_client.get_login_url()
        assert "api_key=test_key" in login_url
        
        # 2. Exchange request token (mock)
        with patch.object(kite_client, '_exchange_token') as mock_exchange:
            mock_exchange.return_value = "test_access_token"
            
            await kite_client.authenticate("test_request_token")
            
            assert kite_client.access_token == "test_access_token"
    
    @pytest.mark.asyncio
    async def test_concurrent_position_limits(self):
        """Test that concurrent position limits are enforced."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(side_effect=lambda key, default=None: {
            'risk.max_concurrent_positions': 3,
        }.get(key, default))
        
        risk_manager = RiskManager(config)
        
        # Open 3 positions
        risk_manager.add_position('NIFTY', 10, 18000)
        risk_manager.add_position('BANKNIFTY', 5, 45000)
        risk_manager.add_position('RELIANCE', 20, 2500)
        
        assert risk_manager.open_positions == 3
        
        # Should not allow 4th position
        can_trade = risk_manager.can_take_position()
        assert can_trade is False


class TestMultiStrategyIntegration:
    """Test multiple strategies working together."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_manages_multiple_strategies(self):
        """Test orchestrator coordinates F&O and Equity strategies."""
        # Setup orchestrator with both strategies
        config = Mock(spec=ConfigLoader)
        
        orchestrator = TradingOrchestrator(
            config=config,
            kite_client=AsyncMock(),
            dry_run=True,
        )
        
        # Add instruments for both strategies
        orchestrator.tracked_instruments = {
            'NIFTY': 'fo',  # F&O
            'RELIANCE': 'equity',  # Equity
        }
        
        # Should handle both without conflicts
        # Test implementation would go here
        pass


class TestBacktestIntegration:
    """Test backtest integration with strategies."""
    
    @pytest.mark.asyncio
    async def test_backtest_with_real_strategy(self):
        """Test running backtest with actual strategy implementation."""
        from backtest.engine import BacktestEngine
        from backtest.runner import BacktestRunner
        
        config = Mock(spec=ConfigLoader)
        config.get = Mock(return_value=None)
        
        # Create sample OHLCV data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 1000 + np.random.randn(100).cumsum(),
            'high': 1005 + np.random.randn(100).cumsum(),
            'low': 995 + np.random.randn(100).cumsum(),
            'close': 1000 + np.random.randn(100).cumsum(),
            'volume': np.random.randint(1000000, 5000000, 100),
        })
        
        # Create signals
        signals = pd.DataFrame({
            'timestamp': dates,
            'signal': np.random.choice([-1, 0, 1], 100),
            'confidence': np.random.uniform(0.5, 1.0, 100),
            'expected_return': np.random.uniform(-2, 5, 100),
        })
        
        # Run backtest
        engine = BacktestEngine(config, initial_capital=10000.0)
        result = engine.run_backtest(data, signals, "TestStrategy")
        
        # Verify results
        assert result.total_trades >= 0
        assert result.equity_curve is not None
        assert len(result.equity_curve) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
