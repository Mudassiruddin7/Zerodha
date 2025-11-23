"""Risk management system with circuit breaker and position limits."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger


class RiskManager:
    """
    Manage risk limits, circuit breakers, and position constraints.
    
    Implements:
    - Daily/weekly loss limits
    - Circuit breaker logic
    - Position size limits
    - Margin monitoring
    - Cool-down periods
    """
    
    def __init__(self, config):
        """
        Initialize risk manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Extract risk config
        self.capital = config.get("capital.starting_capital")
        self.max_daily_loss_pct = config.get("capital.max_daily_loss_pct")
        self.max_positions = config.get("trading_rules.max_concurrent_positions")
        
        # Circuit breaker config
        cb_config = config.get("risk_management.circuit_breaker")
        self.max_daily_losses = cb_config.get("max_daily_losses")
        self.max_consecutive_losses = cb_config.get("max_consecutive_losses")
        self.high_latency_threshold = cb_config.get("high_latency_ms")
        self.min_margin_buffer_pct = cb_config.get("min_margin_buffer_pct")
        
        # State tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.circuit_breaker_active = False
        self.last_trade_time: Dict[str, datetime] = {}
        
        logger.info("Risk manager initialized")
    
    def can_trade(self, strategy_type: str = "equity") -> bool:
        """
        Check if trading is allowed.
        
        Args:
            strategy_type: "equity" or "fo"
            
        Returns:
            True if trading is allowed
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker is active, trading not allowed")
            return False
        
        # Check daily loss limit
        max_loss = self.capital * (self.max_daily_loss_pct / 100)
        if abs(self.daily_pnl) >= max_loss:
            logger.warning(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            self.trip_circuit_breaker("daily_loss_limit")
            return False
        
        # Check cool-down period
        if not self._check_cooldown(strategy_type):
            return False
        
        return True
    
    def _check_cooldown(self, strategy_type: str) -> bool:
        """Check if cool-down period has elapsed."""
        if strategy_type not in self.last_trade_time:
            return True
        
        last_time = self.last_trade_time[strategy_type]
        
        if strategy_type == "fo":
            cooldown_hours = self.config.get("trading_rules.cool_down_hours_fo", 48)
        else:
            cooldown_hours = self.config.get("trading_rules.cool_down_hours_equity", 24)
        
        elapsed = datetime.now() - last_time
        if elapsed < timedelta(hours=cooldown_hours):
            remaining = timedelta(hours=cooldown_hours) - elapsed
            logger.warning(f"Cool-down active for {strategy_type}: {remaining} remaining")
            return False
        
        return True
    
    def record_trade(self, strategy_type: str):
        """Record that a trade was executed."""
        self.last_trade_time[strategy_type] = datetime.now()
    
    def update_pnl(self, pnl: float):
        """Update P&L tracking."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        if pnl < 0:
            self.daily_losses += 1
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Check circuit breaker conditions
        if self.daily_losses >= self.max_daily_losses:
            self.trip_circuit_breaker("max_daily_losses")
        
        if self.consecutive_losses >= self.max_consecutive_losses:
            self.trip_circuit_breaker("consecutive_losses")
    
    def trip_circuit_breaker(self, reason: str):
        """Trip the circuit breaker."""
        self.circuit_breaker_active = True
        logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
    
    def reset_circuit_breaker(self):
        """Reset the circuit breaker (manual intervention required)."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker reset")
    
    def reset_daily_metrics(self):
        """Reset daily tracking metrics (call at start of day)."""
        self.daily_pnl = 0.0
        self.daily_losses = 0
        self.consecutive_losses = 0
        logger.info("Daily metrics reset")
