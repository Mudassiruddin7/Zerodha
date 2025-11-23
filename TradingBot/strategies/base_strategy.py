"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger


class Signal:
    """Trading signal data structure."""
    
    def __init__(
        self,
        instrument: str,
        instrument_token: int,
        action: str,  # BUY or SELL
        strategy_type: str,  # fo, equity_swing, equity_intraday
        entry_price: float,
        quantity: int,
        confidence: float,
        expected_return: float,
        stop_loss: float,
        take_profit: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.instrument = instrument
        self.instrument_token = instrument_token
        self.action = action
        self.strategy_type = strategy_type
        self.entry_price = entry_price
        self.quantity = quantity
        self.confidence = confidence
        self.expected_return = expected_return
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.signal_id = f"{strategy_type}_{instrument}_{int(self.timestamp.timestamp())}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "instrument_token": self.instrument_token,
            "action": self.action,
            "strategy_type": self.strategy_type,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "expected_return": self.expected_return,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        return (
            f"Signal({self.action} {self.instrument} @ {self.entry_price:.2f}, "
            f"qty={self.quantity}, conf={self.confidence:.2%}, "
            f"exp_ret={self.expected_return:.2%})"
        )


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement:
    - generate_signals(): Generate trading signals
    - calculate_position_size(): Determine position sizing
    - should_exit(): Check exit conditions
    """
    
    def __init__(self, config, name: str):
        """
        Initialize base strategy.
        
        Args:
            config: Configuration object
            name: Strategy name
        """
        self.config = config
        self.name = name
        self.enabled = True
        
        # Strategy state
        self.active_signals: List[Signal] = []
        self.historical_signals: List[Signal] = []
        
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    async def generate_signals(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate trading signals based on market data and features.
        
        Args:
            market_data: Current market data (quotes, OHLC, etc.)
            features: Computed technical features
            ml_prediction: ML model predictions (p_win, expected_return)
            
        Returns:
            List of Signal objects
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        current_positions: int
    ) -> int:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal
            available_capital: Available capital in INR
            current_positions: Number of current open positions
            
        Returns:
            Position size (quantity)
        """
        pass
    
    @abstractmethod
    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Check if position should be exited.
        
        Args:
            position: Current position details
            current_price: Current market price
            market_data: Current market data
            
        Returns:
            True if should exit, False otherwise
        """
        pass
    
    def add_signal(self, signal: Signal):
        """Add signal to active signals."""
        self.active_signals.append(signal)
        logger.info(f"Signal added: {signal}")
    
    def archive_signal(self, signal: Signal):
        """Move signal from active to historical."""
        if signal in self.active_signals:
            self.active_signals.remove(signal)
        self.historical_signals.append(signal)
    
    def get_active_signals(self) -> List[Signal]:
        """Get all active signals."""
        return self.active_signals.copy()
    
    def enable(self):
        """Enable strategy."""
        self.enabled = True
        logger.info(f"Strategy '{self.name}' enabled")
    
    def disable(self):
        """Disable strategy."""
        self.enabled = False
        logger.info(f"Strategy '{self.name}' disabled")
    
    def calculate_expected_profit(
        self,
        entry_price: float,
        quantity: int,
        expected_return: float,
        fees: float
    ) -> float:
        """
        Calculate expected profit after fees.
        
        Args:
            entry_price: Entry price
            quantity: Position size
            expected_return: Expected return percentage
            fees: Total fees in INR
            
        Returns:
            Expected net profit in INR
        """
        gross_profit = entry_price * quantity * (expected_return / 100)
        net_profit = gross_profit - fees
        return net_profit
