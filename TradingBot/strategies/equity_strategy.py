"""Equity trading strategy for swing and intraday trades."""

import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, time
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal


class EquityStrategy(BaseStrategy):
    """
    Equity trading strategy with two sub-strategies:
    1. Swing trading: EMA crossover, 5-day hold
    2. Intraday trading: MACD on 5-min, max 4-hour hold
    
    Long-only positions with strict risk management.
    """
    
    def __init__(self, config):
        """Initialize equity strategy."""
        super().__init__(config, "Equity Strategy")
        
        # Strategy configuration
        self.equity_config = config.get("equity_strategy")
        self.enabled = self.equity_config.get("enabled", True)
        
        # Swing trading config
        self.swing_config = self.equity_config.get("swing_trading")
        self.swing_enabled = self.swing_config.get("enabled", True)
        self.swing_ema_fast = self.swing_config.get("ema_fast", 9)
        self.swing_ema_slow = self.swing_config.get("ema_slow", 21)
        self.swing_min_confidence = self.swing_config.get("min_model_confidence", 0.75)
        self.swing_holding_days = self.swing_config.get("holding_days", 5)
        self.swing_target_pct = self.swing_config.get("target_profit_pct", 5.0)
        self.swing_sl_pct = self.swing_config.get("stop_loss_pct", 3.0)
        self.swing_max_position_pct = self.swing_config.get("max_position_size_pct", 20.0)
        
        # Intraday trading config
        self.intraday_config = self.equity_config.get("intraday_trading")
        self.intraday_enabled = self.intraday_config.get("enabled", False)  # Disabled for MVP
        self.intraday_timeframe = self.intraday_config.get("timeframe", "5m")
        self.intraday_max_hold_hours = self.intraday_config.get("max_hold_hours", 4)
        self.intraday_target_pct = self.intraday_config.get("target_profit_pct", 2.0)
        self.intraday_sl_pct = self.intraday_config.get("stop_loss_pct", 1.5)
        self.intraday_exit_time = self.intraday_config.get("exit_time", "15:15")
        
        logger.info(f"Equity Strategy initialized (Swing: {self.swing_enabled}, Intraday: {self.intraday_enabled})")
    
    async def generate_signals(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate equity trading signals.
        
        Args:
            market_data: Current market data
            features: Technical indicators
            ml_prediction: ML model predictions
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.enabled:
            logger.debug("Equity strategy is disabled")
            return signals
        
        try:
            # Generate swing signals
            if self.swing_enabled:
                swing_signals = self._generate_swing_signals(
                    market_data, features, ml_prediction
                )
                signals.extend(swing_signals)
            
            # Generate intraday signals
            if self.intraday_enabled:
                intraday_signals = self._generate_intraday_signals(
                    market_data, features, ml_prediction
                )
                signals.extend(intraday_signals)
            
        except Exception as e:
            logger.error(f"Failed to generate equity signals: {e}")
        
        return signals
    
    def _generate_swing_signals(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]]
    ) -> List[Signal]:
        """Generate swing trading signals based on EMA crossover."""
        signals = []
        
        try:
            # Check ML confidence
            if ml_prediction and ml_prediction.get("p_win", 0) < self.swing_min_confidence:
                logger.debug(
                    f"ML confidence {ml_prediction.get('p_win', 0):.2%} "
                    f"below swing threshold {self.swing_min_confidence:.2%}"
                )
                return signals
            
            # EMA crossover check
            ema_fast = features.get(f"ema_{self.swing_ema_fast}")
            ema_slow = features.get(f"ema_{self.swing_ema_slow}")
            
            if ema_fast is None or ema_slow is None:
                logger.debug("EMA features not available")
                return signals
            
            # Bullish crossover: Fast EMA crosses above Slow EMA
            if ema_fast > ema_slow:
                current_price = market_data.get("last_price", 0)
                
                if current_price <= 0:
                    return signals
                
                # Calculate position size and profit
                signal = Signal(
                    instrument=market_data.get("tradingsymbol", ""),
                    instrument_token=market_data.get("instrument_token", 0),
                    action="BUY",
                    strategy_type="equity_swing",
                    entry_price=current_price,
                    quantity=1,  # Will be calculated by position sizer
                    confidence=ml_prediction.get("p_win", 0.75) if ml_prediction else 0.75,
                    expected_return=ml_prediction.get("expected_return", self.swing_target_pct) if ml_prediction else self.swing_target_pct,
                    stop_loss=current_price * (1 - self.swing_sl_pct / 100),
                    take_profit=current_price * (1 + self.swing_target_pct / 100),
                    metadata={
                        "holding_period_days": self.swing_holding_days,
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "rsi": features.get("rsi_14"),
                        "macd": features.get("macd"),
                        "ml_prediction": ml_prediction
                    }
                )
                
                signals.append(signal)
                logger.info(f"Swing signal generated: {signal}")
        
        except Exception as e:
            logger.error(f"Failed to generate swing signals: {e}")
        
        return signals
    
    def _generate_intraday_signals(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]]
    ) -> List[Signal]:
        """Generate intraday trading signals based on MACD."""
        signals = []
        
        try:
            # MACD crossover check
            macd = features.get("macd")
            macd_signal = features.get("macd_signal")
            
            if macd is None or macd_signal is None:
                return signals
            
            # Bullish crossover: MACD crosses above signal line
            if macd > macd_signal:
                current_price = market_data.get("last_price", 0)
                
                if current_price <= 0:
                    return signals
                
                signal = Signal(
                    instrument=market_data.get("tradingsymbol", ""),
                    instrument_token=market_data.get("instrument_token", 0),
                    action="BUY",
                    strategy_type="equity_intraday",
                    entry_price=current_price,
                    quantity=1,
                    confidence=ml_prediction.get("p_win", 0.70) if ml_prediction else 0.70,
                    expected_return=ml_prediction.get("expected_return", self.intraday_target_pct) if ml_prediction else self.intraday_target_pct,
                    stop_loss=current_price * (1 - self.intraday_sl_pct / 100),
                    take_profit=current_price * (1 + self.intraday_target_pct / 100),
                    metadata={
                        "max_hold_hours": self.intraday_max_hold_hours,
                        "exit_time": self.intraday_exit_time,
                        "macd": macd,
                        "macd_signal": macd_signal,
                        "rsi": features.get("rsi_14"),
                        "ml_prediction": ml_prediction
                    }
                )
                
                signals.append(signal)
                logger.info(f"Intraday signal generated: {signal}")
        
        except Exception as e:
            logger.error(f"Failed to generate intraday signals: {e}")
        
        return signals
    
    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        current_positions: int
    ) -> int:
        """
        Calculate equity position size.
        
        Args:
            signal: Trading signal
            available_capital: Available capital
            current_positions: Current number of positions
            
        Returns:
            Position size (quantity of shares)
        """
        try:
            # For swing trades, use percentage of capital
            if signal.strategy_type == "equity_swing":
                max_position_value = available_capital * (self.swing_max_position_pct / 100)
            else:
                # Intraday: more conservative sizing
                max_position_value = available_capital * 0.10  # 10% max
            
            # Calculate quantity
            quantity = int(max_position_value / signal.entry_price)
            
            # Ensure at least 1 share
            quantity = max(1, quantity)
            
            logger.info(
                f"Equity position size: {quantity} shares "
                f"(value: {quantity * signal.entry_price:.2f} INR)"
            )
            
            return quantity
            
        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            return 1
    
    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Check if equity position should be exited.
        
        Exits on:
        - Take profit hit
        - Stop loss hit
        - Holding period expired (swing)
        - Max hold time expired (intraday)
        - End of day (intraday)
        """
        try:
            entry_price = position["entry_price"]
            entry_time = position.get("entry_time", datetime.now())
            strategy_type = position.get("strategy_type", "equity_swing")
            
            # Calculate return
            return_pct = (current_price - entry_price) / entry_price * 100
            
            # Take profit
            target = self.swing_target_pct if strategy_type == "equity_swing" else self.intraday_target_pct
            if return_pct >= target:
                logger.info(f"Take profit hit: {return_pct:.2f}% >= {target}%")
                return True
            
            # Stop loss
            sl = self.swing_sl_pct if strategy_type == "equity_swing" else self.intraday_sl_pct
            if return_pct <= -sl:
                logger.info(f"Stop loss hit: {return_pct:.2f}% <= -{sl}%")
                return True
            
            # Time-based exits
            if strategy_type == "equity_swing":
                # Swing: Exit after holding period
                days_held = (datetime.now() - entry_time).days
                if days_held >= self.swing_holding_days:
                    logger.info(f"Holding period expired: {days_held} days")
                    return True
            
            else:  # Intraday
                # Exit after max hold hours
                hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                if hours_held >= self.intraday_max_hold_hours:
                    logger.info(f"Max hold time expired: {hours_held:.1f} hours")
                    return True
                
                # Exit at specified time
                exit_hour, exit_min = map(int, self.intraday_exit_time.split(":"))
                current_time = datetime.now().time()
                
                if current_time >= time(exit_hour, exit_min):
                    logger.info("Intraday exit time reached")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Exit check failed: {e}")
            return False
