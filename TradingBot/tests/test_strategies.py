"""Unit tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from strategies.fo_strategy import FOStrategy
from strategies.equity_strategy import EquityStrategy
from config.config_loader import ConfigLoader


class TestFOStrategy:
    """Test F&O options trading strategy."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(side_effect=lambda key, default=None: {
            'fo_strategy.strike_selection.min_delta': 0.20,
            'fo_strategy.strike_selection.max_delta': 0.40,
            'fo_strategy.strike_selection.min_open_interest': 1000,
            'fo_strategy.strike_selection.min_volume': 500,
            'fo_strategy.strike_selection.max_spread_pct': 2.0,
            'fo_strategy.entry.macd_bullish': True,
            'fo_strategy.entry.rsi_min': 40,
            'fo_strategy.entry.rsi_max': 70,
            'fo_strategy.entry.min_expected_profit_inr': 150,
            'fo_strategy.exit.take_profit_pct': 50.0,
            'fo_strategy.exit.stop_loss_pct': 60.0,
            'fo_strategy.position_sizing.risk_pct': 5.0,
            'fo_strategy.position_sizing.kelly_fraction': 0.25,
            'ml_models.classifier.min_confidence': 0.80,
        }.get(key, default))
        return config
    
    @pytest.fixture
    def strategy(self, config):
        """Create FOStrategy instance."""
        return FOStrategy(config)
    
    def test_strike_selection_delta_filter(self, strategy):
        """Test that strike selection filters by delta range."""
        strikes = pd.DataFrame([
            {'strike': 18000, 'delta': 0.15, 'open_interest': 5000, 'volume': 1000, 'bid': 100, 'ask': 101},
            {'strike': 18100, 'delta': 0.25, 'open_interest': 5000, 'volume': 1000, 'bid': 100, 'ask': 101},  # Valid
            {'strike': 18200, 'delta': 0.35, 'open_interest': 5000, 'volume': 1000, 'bid': 100, 'ask': 101},  # Valid
            {'strike': 18300, 'delta': 0.45, 'open_interest': 5000, 'volume': 1000, 'bid': 100, 'ask': 101},
        ])
        
        filtered = strategy._filter_strikes_by_delta(strikes, min_delta=0.20, max_delta=0.40)
        
        assert len(filtered) == 2
        assert filtered['delta'].min() >= 0.20
        assert filtered['delta'].max() <= 0.40
    
    def test_strike_selection_oi_filter(self, strategy):
        """Test that strike selection filters by open interest."""
        strikes = pd.DataFrame([
            {'strike': 18000, 'delta': 0.30, 'open_interest': 500, 'volume': 1000, 'bid': 100, 'ask': 101},   # Low OI
            {'strike': 18100, 'delta': 0.30, 'open_interest': 2000, 'volume': 1000, 'bid': 100, 'ask': 101},  # Valid
            {'strike': 18200, 'delta': 0.30, 'open_interest': 3000, 'volume': 1000, 'bid': 100, 'ask': 101},  # Valid
        ])
        
        filtered = strategy._filter_strikes_by_liquidity(strikes, min_oi=1000, min_volume=500)
        
        assert len(filtered) == 2
        assert filtered['open_interest'].min() >= 1000
    
    def test_strike_selection_spread_filter(self, strategy):
        """Test that strike selection filters by bid-ask spread."""
        strikes = pd.DataFrame([
            {'strike': 18000, 'delta': 0.30, 'open_interest': 2000, 'volume': 1000, 'bid': 100, 'ask': 103},  # 3% spread
            {'strike': 18100, 'delta': 0.30, 'open_interest': 2000, 'volume': 1000, 'bid': 100, 'ask': 101},  # 1% spread - Valid
        ])
        
        filtered = strategy._filter_strikes_by_spread(strikes, max_spread_pct=2.0)
        
        assert len(filtered) == 1
        assert filtered.iloc[0]['strike'] == 18100
    
    def test_position_sizing_kelly(self, strategy):
        """Test Kelly criterion position sizing."""
        # Win probability = 0.70, Win/Loss ratio = 2.0
        # Kelly = (0.70 * 2.0 - 0.30) / 2.0 = 0.55
        # With fraction 0.25: 0.55 * 0.25 = 0.1375
        
        kelly_fraction = strategy._calculate_kelly_fraction(
            win_prob=0.70,
            avg_win=200,
            avg_loss=100,
        )
        
        # Full Kelly
        expected_full = (0.70 * 2.0 - 0.30) / 2.0
        assert abs(kelly_fraction - expected_full) < 0.01
        
        # Apply safety fraction
        safe_kelly = kelly_fraction * 0.25
        assert safe_kelly < 0.15
    
    def test_position_sizing_risk_based(self, strategy):
        """Test risk-based position sizing."""
        available_capital = 10000.0
        premium_price = 100.0
        max_loss_pct = 60.0  # Stop loss at 60% loss
        risk_pct = 5.0  # Risk 5% of capital
        
        quantity = strategy._calculate_position_size(
            available_capital=available_capital,
            premium_price=premium_price,
            max_loss_pct=max_loss_pct,
            risk_pct=risk_pct,
        )
        
        # Max risk = 10000 * 0.05 = 500 INR
        # Risk per contract = 100 * 0.60 = 60 INR
        # Quantity = 500 / 60 = 8.33 -> 8
        
        assert quantity == 8
        
        # Verify total investment doesn't exceed capital
        total_investment = quantity * premium_price
        assert total_investment <= available_capital
    
    @pytest.mark.asyncio
    async def test_entry_signal_requires_macd_cross(self, strategy):
        """Test that entry signal requires MACD bullish crossover."""
        market_data = {
            'macd': 10,
            'macd_signal': 5,  # MACD > signal = bullish
            'rsi_14': 50,
            'adx': 25,
            'close': 18000,
        }
        
        # Mock ML model
        strategy.model = Mock()
        strategy.model.predict = Mock(return_value={
            'p_win': 0.85,
            'expected_return': 5.0,
            'confidence': 0.85,
            'should_trade': True,
        })
        
        signal = await strategy.generate_signal(market_data)
        
        # Should generate signal when MACD is bullish
        assert signal is not None
        assert signal.direction == 'BUY'
    
    @pytest.mark.asyncio
    async def test_entry_signal_rejects_low_confidence(self, strategy):
        """Test that entry signal is rejected if ML confidence is low."""
        market_data = {
            'macd': 10,
            'macd_signal': 5,
            'rsi_14': 50,
            'adx': 25,
            'close': 18000,
        }
        
        # Mock ML model with low confidence
        strategy.model = Mock()
        strategy.model.predict = Mock(return_value={
            'p_win': 0.60,  # Below 0.80 threshold
            'expected_return': 3.0,
            'confidence': 0.60,
            'should_trade': False,
        })
        
        signal = await strategy.generate_signal(market_data)
        
        # Should NOT generate signal
        assert signal is None
    
    def test_expected_profit_calculation(self, strategy):
        """Test expected profit calculation after fees."""
        entry_price = 100.0
        quantity = 10
        expected_return_pct = 50.0  # 50% gain expected
        
        # Expected profit = 100 * 10 * 0.50 = 500 INR
        # Fees ~= 55 (brokerage) + taxes ~= 70 INR
        # Net profit should be around 430 INR
        
        expected_profit = strategy._calculate_expected_profit(
            entry_price=entry_price,
            quantity=quantity,
            expected_return_pct=expected_return_pct,
        )
        
        assert expected_profit > 150  # Above minimum threshold
        assert expected_profit < 500  # Less than gross profit due to fees


class TestEquityStrategy:
    """Test equity swing and intraday strategy."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(side_effect=lambda key, default=None: {
            'equity_strategy.swing.enabled': True,
            'equity_strategy.swing.ema_short': 9,
            'equity_strategy.swing.ema_long': 21,
            'equity_strategy.swing.holding_days': 5,
            'equity_strategy.swing.take_profit_pct': 5.0,
            'equity_strategy.swing.stop_loss_pct': 3.0,
            'equity_strategy.swing.max_position_size_pct': 20.0,
            'equity_strategy.intraday.enabled': False,
            'ml_models.classifier.min_confidence': 0.80,
        }.get(key, default))
        return config
    
    @pytest.fixture
    def strategy(self, config):
        """Create EquityStrategy instance."""
        return EquityStrategy(config)
    
    def test_swing_ema_crossover_bullish(self, strategy):
        """Test detection of bullish EMA crossover."""
        # EMA(9) crosses above EMA(21)
        market_data = {
            'ema_9': 105,
            'ema_21': 100,
            'ema_9_prev': 99,  # Was below
            'ema_21_prev': 100,
            'close': 105,
            'volume': 1000000,
        }
        
        is_bullish = strategy._is_ema_crossover_bullish(market_data)
        assert is_bullish is True
    
    def test_swing_ema_crossover_not_bullish(self, strategy):
        """Test no signal when no crossover."""
        # EMA(9) already above EMA(21), no cross
        market_data = {
            'ema_9': 105,
            'ema_21': 100,
            'ema_9_prev': 104,  # Was already above
            'ema_21_prev': 100,
            'close': 105,
        }
        
        is_bullish = strategy._is_ema_crossover_bullish(market_data)
        assert is_bullish is False
    
    def test_swing_position_sizing(self, strategy):
        """Test swing position sizing based on % of capital."""
        available_capital = 10000.0
        price = 500.0
        max_position_pct = 20.0  # 20% of capital
        
        # Max investment = 10000 * 0.20 = 2000 INR
        # Quantity = 2000 / 500 = 4 shares
        
        quantity = strategy._calculate_swing_position_size(
            available_capital=available_capital,
            price=price,
            max_position_pct=max_position_pct,
        )
        
        assert quantity == 4
        assert quantity * price <= available_capital * (max_position_pct / 100.0)
    
    def test_swing_stop_loss_calculation(self, strategy):
        """Test stop loss price calculation."""
        entry_price = 1000.0
        stop_loss_pct = 3.0
        
        stop_loss_price = strategy._calculate_stop_loss(entry_price, stop_loss_pct)
        
        # Stop loss = 1000 * (1 - 0.03) = 970
        assert stop_loss_price == 970.0
    
    def test_swing_take_profit_calculation(self, strategy):
        """Test take profit price calculation."""
        entry_price = 1000.0
        take_profit_pct = 5.0
        
        take_profit_price = strategy._calculate_take_profit(entry_price, take_profit_pct)
        
        # Take profit = 1000 * (1 + 0.05) = 1050
        assert take_profit_price == 1050.0
    
    @pytest.mark.asyncio
    async def test_swing_exit_on_take_profit(self, strategy):
        """Test exit signal when take profit is hit."""
        # Entry at 1000, take profit at 1050 (5%)
        position = {
            'entry_price': 1000.0,
            'entry_time': datetime.now() - timedelta(days=2),
            'quantity': 10,
        }
        
        market_data = {
            'close': 1055.0,  # Above take profit
            'ema_9': 1055,
            'ema_21': 1040,
        }
        
        should_exit = strategy._should_exit_swing(position, market_data)
        
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_swing_exit_on_stop_loss(self, strategy):
        """Test exit signal when stop loss is hit."""
        # Entry at 1000, stop loss at 970 (3%)
        position = {
            'entry_price': 1000.0,
            'entry_time': datetime.now() - timedelta(days=1),
            'quantity': 10,
        }
        
        market_data = {
            'close': 965.0,  # Below stop loss
            'ema_9': 970,
            'ema_21': 980,
        }
        
        should_exit = strategy._should_exit_swing(position, market_data)
        
        assert should_exit is True
    
    @pytest.mark.asyncio
    async def test_swing_exit_on_holding_period(self, strategy):
        """Test exit signal when holding period expires."""
        # Entry 6 days ago, max hold = 5 days
        position = {
            'entry_price': 1000.0,
            'entry_time': datetime.now() - timedelta(days=6),
            'quantity': 10,
        }
        
        market_data = {
            'close': 1020.0,  # Small profit
            'ema_9': 1020,
            'ema_21': 1010,
        }
        
        should_exit = strategy._should_exit_swing(position, market_data)
        
        assert should_exit is True


class TestStrategyIntegration:
    """Integration tests for strategy workflows."""
    
    @pytest.mark.asyncio
    async def test_full_fo_trade_cycle(self):
        """Test complete F&O trade cycle from signal to exit."""
        # This would test:
        # 1. Signal generation
        # 2. Strike selection
        # 3. Position sizing
        # 4. Entry execution
        # 5. Exit conditions
        # 6. P&L calculation
        pass  # Implement in integration tests
    
    @pytest.mark.asyncio
    async def test_full_equity_trade_cycle(self):
        """Test complete equity trade cycle."""
        pass  # Implement in integration tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
