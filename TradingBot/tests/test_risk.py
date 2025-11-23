"""Unit tests for risk management."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from risk.risk_manager import RiskManager
from config.config_loader import ConfigLoader


class TestRiskManager:
    """Test risk management logic."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(side_effect=lambda key, default=None: {
            'capital.starting_capital': 10000.0,
            'capital.max_daily_loss_pct': 7.0,
            'capital.max_weekly_loss_pct': 15.0,
            'risk.max_concurrent_positions': 3,
            'risk.circuit_breaker.max_daily_losses': 3,
            'risk.circuit_breaker.max_consecutive_losses': 2,
            'risk.circuit_breaker.high_latency_ms': 500,
            'risk.circuit_breaker.min_margin_buffer_pct': 15.0,
            'risk.cooldown.fo_hours': 48,
            'risk.cooldown.equity_hours': 24,
        }.get(key, default))
        return config
    
    @pytest.fixture
    def risk_manager(self, config):
        """Create RiskManager instance."""
        return RiskManager(config)
    
    def test_initial_state(self, risk_manager):
        """Test risk manager initializes correctly."""
        assert risk_manager.current_capital == 10000.0
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.weekly_pnl == 0.0
        assert risk_manager.open_positions == 0
        assert risk_manager.circuit_breaker_active is False
    
    def test_daily_loss_limit_not_exceeded(self, risk_manager):
        """Test trading allowed when within daily loss limit."""
        # 5% loss, limit is 7%
        risk_manager.daily_pnl = -500.0
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is True
    
    def test_daily_loss_limit_exceeded(self, risk_manager):
        """Test trading blocked when daily loss limit exceeded."""
        # 8% loss, limit is 7%
        risk_manager.daily_pnl = -800.0
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is False
    
    def test_weekly_loss_limit_not_exceeded(self, risk_manager):
        """Test trading allowed when within weekly loss limit."""
        # 10% loss, limit is 15%
        risk_manager.weekly_pnl = -1000.0
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is True
    
    def test_weekly_loss_limit_exceeded(self, risk_manager):
        """Test trading blocked when weekly loss limit exceeded."""
        # 16% loss, limit is 15%
        risk_manager.weekly_pnl = -1600.0
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is False
    
    def test_max_positions_not_exceeded(self, risk_manager):
        """Test trading allowed when below max positions."""
        risk_manager.open_positions = 2  # Max is 3
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is True
    
    def test_max_positions_exceeded(self, risk_manager):
        """Test trading blocked when max positions reached."""
        risk_manager.open_positions = 3  # At max
        
        can_trade = risk_manager.can_take_position()
        
        assert can_trade is False
    
    def test_circuit_breaker_max_daily_losses(self, risk_manager):
        """Test circuit breaker triggers on max daily losses."""
        # Record 3 losing trades (max is 3)
        for i in range(3):
            risk_manager.record_trade_result(pnl=-100.0, strategy='test')
        
        # Should trigger circuit breaker
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.can_take_position() is False
    
    def test_circuit_breaker_consecutive_losses(self, risk_manager):
        """Test circuit breaker triggers on consecutive losses."""
        # Record 2 consecutive losses (max is 2)
        risk_manager.record_trade_result(pnl=-100.0, strategy='test')
        risk_manager.record_trade_result(pnl=-100.0, strategy='test')
        
        # Should trigger circuit breaker
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.can_take_position() is False
    
    def test_circuit_breaker_resets_on_win(self, risk_manager):
        """Test consecutive loss counter resets on winning trade."""
        # Record 1 loss
        risk_manager.record_trade_result(pnl=-100.0, strategy='test')
        assert risk_manager.consecutive_losses == 1
        
        # Record win
        risk_manager.record_trade_result(pnl=200.0, strategy='test')
        
        # Should reset consecutive losses
        assert risk_manager.consecutive_losses == 0
    
    def test_circuit_breaker_high_latency(self, risk_manager):
        """Test circuit breaker triggers on high latency."""
        # Simulate high latency (>500ms)
        risk_manager.record_latency(600)
        
        # Should trigger circuit breaker
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.can_take_position() is False
    
    def test_circuit_breaker_low_margin(self, risk_manager):
        """Test circuit breaker triggers on low margin buffer."""
        # Available margin = 1000, used margin = 900
        # Buffer = 1000 - 900 = 100 = 10% (below 15% threshold)
        
        risk_manager.check_margin_buffer(
            available_margin=1000.0,
            used_margin=900.0,
        )
        
        # Should trigger circuit breaker
        assert risk_manager.circuit_breaker_active is True
        assert risk_manager.can_take_position() is False
    
    def test_cooldown_fo_strategy(self, risk_manager):
        """Test cooldown period for F&O strategy."""
        # Record F&O trade
        now = datetime.now()
        risk_manager.record_trade_result(pnl=100.0, strategy='FOStrategy', timestamp=now)
        
        # Check immediately - should be in cooldown
        in_cooldown = risk_manager.is_in_cooldown('FOStrategy', instrument='NIFTY')
        assert in_cooldown is True
        
        # Check after 47 hours - still in cooldown (48h required)
        future_time = now + timedelta(hours=47)
        in_cooldown = risk_manager.is_in_cooldown('FOStrategy', instrument='NIFTY', current_time=future_time)
        assert in_cooldown is True
        
        # Check after 49 hours - cooldown expired
        future_time = now + timedelta(hours=49)
        in_cooldown = risk_manager.is_in_cooldown('FOStrategy', instrument='NIFTY', current_time=future_time)
        assert in_cooldown is False
    
    def test_cooldown_equity_strategy(self, risk_manager):
        """Test cooldown period for equity strategy."""
        # Record equity trade
        now = datetime.now()
        risk_manager.record_trade_result(pnl=100.0, strategy='EquityStrategy', timestamp=now)
        
        # Check after 23 hours - still in cooldown (24h required)
        future_time = now + timedelta(hours=23)
        in_cooldown = risk_manager.is_in_cooldown('EquityStrategy', instrument='RELIANCE', current_time=future_time)
        assert in_cooldown is True
        
        # Check after 25 hours - cooldown expired
        future_time = now + timedelta(hours=25)
        in_cooldown = risk_manager.is_in_cooldown('EquityStrategy', instrument='RELIANCE', current_time=future_time)
        assert in_cooldown is False
    
    def test_position_tracking(self, risk_manager):
        """Test position tracking."""
        assert risk_manager.open_positions == 0
        
        # Open position
        risk_manager.add_position('NIFTY', quantity=10, entry_price=18000)
        assert risk_manager.open_positions == 1
        
        # Open another
        risk_manager.add_position('RELIANCE', quantity=5, entry_price=2500)
        assert risk_manager.open_positions == 2
        
        # Close position
        risk_manager.close_position('NIFTY')
        assert risk_manager.open_positions == 1
        
        # Close last position
        risk_manager.close_position('RELIANCE')
        assert risk_manager.open_positions == 0
    
    def test_pnl_tracking(self, risk_manager):
        """Test P&L tracking updates correctly."""
        initial_capital = risk_manager.current_capital
        
        # Record winning trade
        risk_manager.record_trade_result(pnl=300.0, strategy='test')
        
        assert risk_manager.daily_pnl == 300.0
        assert risk_manager.weekly_pnl == 300.0
        assert risk_manager.current_capital == initial_capital + 300.0
        
        # Record losing trade
        risk_manager.record_trade_result(pnl=-150.0, strategy='test')
        
        assert risk_manager.daily_pnl == 150.0  # Net
        assert risk_manager.weekly_pnl == 150.0
        assert risk_manager.current_capital == initial_capital + 150.0
    
    def test_reset_daily_tracking(self, risk_manager):
        """Test daily metrics reset."""
        # Set some daily metrics
        risk_manager.daily_pnl = -500.0
        risk_manager.daily_losses = 2
        risk_manager.consecutive_losses = 1
        
        # Reset
        risk_manager.reset_daily_tracking()
        
        assert risk_manager.daily_pnl == 0.0
        assert risk_manager.daily_losses == 0
        assert risk_manager.consecutive_losses == 0
    
    def test_reset_circuit_breaker(self, risk_manager):
        """Test circuit breaker can be manually reset."""
        # Trigger circuit breaker
        risk_manager.circuit_breaker_active = True
        risk_manager.circuit_breaker_reason = "Max daily losses"
        
        # Reset
        risk_manager.reset_circuit_breaker()
        
        assert risk_manager.circuit_breaker_active is False
        assert risk_manager.circuit_breaker_reason == ""
    
    def test_get_risk_metrics(self, risk_manager):
        """Test risk metrics reporting."""
        risk_manager.daily_pnl = -300.0
        risk_manager.weekly_pnl = -800.0
        risk_manager.open_positions = 2
        risk_manager.daily_losses = 1
        
        metrics = risk_manager.get_risk_metrics()
        
        assert metrics['daily_pnl'] == -300.0
        assert metrics['daily_loss_pct'] == 3.0  # -300 / 10000
        assert metrics['weekly_pnl'] == -800.0
        assert metrics['weekly_loss_pct'] == 8.0
        assert metrics['open_positions'] == 2
        assert metrics['circuit_breaker_active'] is False
        assert 'daily_loss_limit' in metrics
        assert 'weekly_loss_limit' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
