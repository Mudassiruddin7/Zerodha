"""
Vectorized backtesting engine.

This module provides a fast, vectorized backtesting framework using pandas.
It simulates realistic trading conditions including:
- Transaction costs (55 INR brokerage + taxes)
- Slippage (configurable, default 0.01%)
- Position sizing
- Risk management rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from config.config_loader import ConfigLoader
from strategies.base_strategy import Signal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade execution."""
    entry_time: datetime
    exit_time: Optional[datetime]
    instrument: str
    quantity: int
    entry_price: float
    exit_price: Optional[float]
    direction: str  # 'BUY' or 'SELL'
    strategy: str
    entry_signal_confidence: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    exit_reason: str = ""
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE
    
    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_time is None


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    
    # Performance metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    total_pnl: float = 0.0
    total_fees: float = 0.0
    net_pnl: float = 0.0
    
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    avg_trade_duration: timedelta = field(default_factory=lambda: timedelta(0))
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    duration_days: int = 0
    
    def __post_init__(self):
        """Calculate metrics from trades."""
        if not self.trades:
            return
            
        self.total_trades = len(self.trades)
        closed_trades = [t for t in self.trades if not t.is_open()]
        
        if not closed_trades:
            return
            
        # Win/loss statistics
        winning = [t for t in closed_trades if t.pnl > 0]
        losing = [t for t in closed_trades if t.pnl <= 0]
        
        self.winning_trades = len(winning)
        self.losing_trades = len(losing)
        self.win_rate = self.winning_trades / len(closed_trades) if closed_trades else 0.0
        
        # PnL statistics
        self.total_pnl = sum(t.pnl for t in closed_trades)
        self.total_fees = sum(t.fees for t in closed_trades)
        self.net_pnl = self.total_pnl - self.total_fees
        
        if winning:
            self.avg_win = np.mean([t.pnl for t in winning])
            self.largest_win = max(t.pnl for t in winning)
        
        if losing:
            self.avg_loss = np.mean([t.pnl for t in losing])
            self.largest_loss = min(t.pnl for t in losing)
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning)
        gross_loss = abs(sum(t.pnl for t in losing))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        if len(self.equity_curve) > 0:
            cummax = self.equity_curve.cummax()
            drawdown = (self.equity_curve - cummax) / cummax
            self.max_drawdown = abs(drawdown.min())
            
            # Drawdown duration
            dd_starts = (drawdown == 0) & (drawdown.shift(1) < 0)
            dd_ends = (drawdown == 0) & (drawdown.shift(-1) < 0)
            if dd_starts.any() and dd_ends.any():
                durations = []
                for start_idx in drawdown[dd_starts].index:
                    end_indices = drawdown[dd_ends & (drawdown.index > start_idx)].index
                    if len(end_indices) > 0:
                        durations.append((end_indices[0] - start_idx).days)
                self.max_drawdown_duration = max(durations) if durations else 0
        
        # Risk-adjusted returns
        if len(self.daily_returns) > 1:
            mean_return = self.daily_returns.mean()
            std_return = self.daily_returns.std()
            
            if std_return > 0:
                self.sharpe_ratio = (mean_return / std_return) * np.sqrt(252)  # Annualized
                
                # Sortino (downside deviation)
                downside_returns = self.daily_returns[self.daily_returns < 0]
                if len(downside_returns) > 0:
                    downside_std = downside_returns.std()
                    if downside_std > 0:
                        self.sortino_ratio = (mean_return / downside_std) * np.sqrt(252)
            
            # Calmar ratio
            if self.max_drawdown > 0:
                annual_return = (1 + mean_return) ** 252 - 1
                self.calmar_ratio = annual_return / self.max_drawdown
        
        # Trade duration
        durations = [t.exit_time - t.entry_time for t in closed_trades if t.exit_time]
        if durations:
            self.avg_trade_duration = sum(durations, timedelta(0)) / len(durations)
        
        # Consecutive wins/losses
        streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in closed_trades:
            if trade.pnl > 0:
                if streak > 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            else:
                if streak < 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))
        
        self.max_consecutive_wins = max_win_streak
        self.max_consecutive_losses = max_loss_streak
        
        # Dates
        self.start_date = min(t.entry_time for t in self.trades)
        self.end_date = max(t.exit_time for t in closed_trades if t.exit_time)
        if self.start_date and self.end_date:
            self.duration_days = (self.end_date - self.start_date).days


class BacktestEngine:
    """
    Vectorized backtesting engine for trading strategies.
    
    Features:
    - Vectorized operations using pandas for speed
    - Realistic fee model (55 INR brokerage + taxes)
    - Configurable slippage
    - Position sizing
    - Walk-forward validation support
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        initial_capital: float = 10000.0,
        slippage_pct: float = 0.01,  # 0.01% default
        commission_per_trade: float = 55.0,  # Zerodha brokerage
    ):
        """
        Initialize backtesting engine.
        
        Args:
            config: Configuration loader
            initial_capital: Starting capital in INR
            slippage_pct: Slippage as percentage (0.01 = 0.01%)
            commission_per_trade: Flat commission per trade
        """
        self.config = config
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.commission_per_trade = commission_per_trade
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.equity_dates: List[datetime] = []
        
        logger.info(f"Initialized backtest engine: capital={initial_capital}, slippage={slippage_pct}%")
    
    def calculate_fees(self, price: float, quantity: int, is_equity: bool = True) -> float:
        """
        Calculate total trading fees.
        
        Zerodha fee structure:
        - Brokerage: 55 INR or 0.03% (whichever is lower)
        - STT: 0.025% on sell (equity), 0.05% on sell (F&O)
        - Transaction charges: ~0.00325%
        - GST: 18% on brokerage
        - SEBI charges: 0.0001%
        - Stamp duty: 0.003% on buy
        
        Args:
            price: Trade price
            quantity: Number of shares/contracts
            is_equity: True for equity, False for F&O
            
        Returns:
            Total fees in INR
        """
        turnover = price * quantity
        
        # Brokerage (55 INR or 0.03%, whichever is lower)
        brokerage_pct = min(self.commission_per_trade, turnover * 0.0003)
        brokerage = brokerage_pct * 2  # Buy + Sell
        
        # STT (only on sell)
        stt_rate = 0.00025 if is_equity else 0.0005
        stt = turnover * stt_rate
        
        # Transaction charges
        txn_charges = turnover * 0.0000325 * 2  # Buy + Sell
        
        # GST on brokerage
        gst = brokerage * 0.18
        
        # SEBI charges
        sebi = turnover * 0.000001 * 2
        
        # Stamp duty (only on buy)
        stamp_duty = turnover * 0.00003
        
        total_fees = brokerage + stt + txn_charges + gst + sebi + stamp_duty
        
        return round(total_fees, 2)
    
    def apply_slippage(self, price: float, direction: str) -> float:
        """
        Apply slippage to execution price.
        
        Args:
            price: Original price
            direction: 'BUY' or 'SELL'
            
        Returns:
            Price after slippage
        """
        slippage_factor = self.slippage_pct / 100.0
        
        if direction == 'BUY':
            # Pay more when buying
            return price * (1 + slippage_factor)
        else:
            # Receive less when selling
            return price * (1 - slippage_factor)
    
    def calculate_position_size(
        self,
        price: float,
        available_capital: float,
        risk_pct: float = 0.05,
        stop_loss_pct: float = 0.03,
    ) -> int:
        """
        Calculate position size based on risk management.
        
        Args:
            price: Entry price
            available_capital: Available capital
            risk_pct: Max risk per trade (0.05 = 5%)
            stop_loss_pct: Stop loss percentage
            
        Returns:
            Number of shares/contracts
        """
        max_risk_amount = available_capital * risk_pct
        risk_per_share = price * stop_loss_pct
        
        if risk_per_share <= 0:
            return 0
        
        quantity = int(max_risk_amount / risk_per_share)
        
        # Ensure we don't exceed available capital
        max_quantity = int(available_capital / price)
        
        return min(quantity, max_quantity)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame,
        strategy_name: str = "Unknown",
    ) -> BacktestResult:
        """
        Run vectorized backtest on historical data.
        
        Args:
            data: OHLCV data with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            signals: Signal data with columns ['timestamp', 'signal', 'confidence', 'expected_return']
                     signal: 1 (BUY), -1 (SELL), 0 (HOLD)
            strategy_name: Name of strategy being tested
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Running backtest for {strategy_name} on {len(data)} bars")
        
        # Reset state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.equity_dates = []
        
        # Merge data and signals
        df = data.merge(signals, on='timestamp', how='left')
        df['signal'] = df['signal'].fillna(0)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        current_capital = self.initial_capital
        current_position: Optional[Trade] = None
        
        for idx, row in df.iterrows():
            timestamp = row['timestamp']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            signal = row['signal']
            confidence = row.get('confidence', 0.0)
            
            # Update MAE/MFE for open position
            if current_position and not current_position.is_open():
                current_position = None
                
            if current_position:
                # Update unrealized P&L
                unrealized_pnl = (close_price - current_position.entry_price) * current_position.quantity
                
                # Update MAE (worst move against position)
                if current_position.direction == 'BUY':
                    adverse = (low_price - current_position.entry_price) * current_position.quantity
                    favorable = (high_price - current_position.entry_price) * current_position.quantity
                else:
                    adverse = (high_price - current_position.entry_price) * current_position.quantity
                    favorable = (low_price - current_position.entry_price) * current_position.quantity
                
                current_position.max_adverse_excursion = min(
                    current_position.max_adverse_excursion, adverse
                )
                current_position.max_favorable_excursion = max(
                    current_position.max_favorable_excursion, favorable
                )
            
            # Entry signals
            if current_position is None and signal != 0:
                direction = 'BUY' if signal > 0 else 'SELL'
                
                # Calculate position size
                quantity = self.calculate_position_size(
                    price=close_price,
                    available_capital=current_capital,
                    risk_pct=0.05,
                    stop_loss_pct=0.03,
                )
                
                if quantity > 0:
                    # Apply slippage
                    entry_price = self.apply_slippage(close_price, direction)
                    
                    # Calculate fees
                    fees = self.calculate_fees(entry_price, quantity, is_equity=True)
                    
                    # Create trade
                    current_position = Trade(
                        entry_time=timestamp,
                        exit_time=None,
                        instrument=row.get('instrument', 'UNKNOWN'),
                        quantity=quantity,
                        entry_price=entry_price,
                        exit_price=None,
                        direction=direction,
                        strategy=strategy_name,
                        entry_signal_confidence=confidence,
                        fees=fees,
                        slippage=abs(entry_price - close_price) * quantity,
                    )
                    
                    self.trades.append(current_position)
                    current_capital -= (entry_price * quantity + fees)
                    
                    logger.debug(f"Entry: {direction} {quantity} @ {entry_price:.2f}, capital: {current_capital:.2f}")
            
            # Exit signals (opposite direction or close signal)
            elif current_position and (
                (signal < 0 and current_position.direction == 'BUY') or
                (signal > 0 and current_position.direction == 'SELL')
            ):
                # Exit position
                exit_price = self.apply_slippage(close_price, 'SELL' if current_position.direction == 'BUY' else 'BUY')
                
                current_position.exit_time = timestamp
                current_position.exit_price = exit_price
                current_position.exit_reason = "Signal"
                
                # Calculate P&L
                if current_position.direction == 'BUY':
                    pnl = (exit_price - current_position.entry_price) * current_position.quantity
                else:
                    pnl = (current_position.entry_price - exit_price) * current_position.quantity
                
                current_position.pnl = pnl - current_position.fees
                current_position.pnl_pct = (pnl / (current_position.entry_price * current_position.quantity)) * 100
                
                current_capital += (exit_price * current_position.quantity - current_position.fees)
                
                logger.debug(f"Exit: P&L={current_position.pnl:.2f}, capital: {current_capital:.2f}")
                
                current_position = None
            
            # Update equity curve
            equity = current_capital
            if current_position:
                # Add unrealized P&L
                unrealized = (close_price - current_position.entry_price) * current_position.quantity
                equity += unrealized
            
            self.equity_curve.append(equity)
            self.equity_dates.append(timestamp)
        
        # Close any remaining open positions
        if current_position:
            last_row = df.iloc[-1]
            exit_price = self.apply_slippage(last_row['close'], 'SELL' if current_position.direction == 'BUY' else 'BUY')
            
            current_position.exit_time = last_row['timestamp']
            current_position.exit_price = exit_price
            current_position.exit_reason = "End of backtest"
            
            if current_position.direction == 'BUY':
                pnl = (exit_price - current_position.entry_price) * current_position.quantity
            else:
                pnl = (current_position.entry_price - exit_price) * current_position.quantity
            
            current_position.pnl = pnl - current_position.fees
            current_position.pnl_pct = (pnl / (current_position.entry_price * current_position.quantity)) * 100
        
        # Create equity curve series
        equity_series = pd.Series(self.equity_curve[1:], index=self.equity_dates)
        
        # Calculate daily returns
        daily_equity = equity_series.resample('D').last().fillna(method='ffill')
        daily_returns = daily_equity.pct_change().dropna()
        
        # Create result
        result = BacktestResult(
            trades=self.trades,
            equity_curve=equity_series,
            daily_returns=daily_returns,
        )
        
        logger.info(f"Backtest complete: {result.total_trades} trades, Win rate: {result.win_rate:.2%}, Net P&L: {result.net_pnl:.2f}")
        
        return result
    
    def walk_forward_validation(
        self,
        data: pd.DataFrame,
        signal_generator,
        train_window_days: int = 180,
        test_window_days: int = 30,
        min_passing_pct: float = 0.80,
    ) -> Tuple[List[BacktestResult], bool]:
        """
        Perform walk-forward validation.
        
        Args:
            data: Historical OHLCV data
            signal_generator: Function that takes data and returns signals DataFrame
            train_window_days: Days for training window
            test_window_days: Days for testing window
            min_passing_pct: Minimum % of windows that must pass (0.80 = 80%)
            
        Returns:
            (list of results for each window, whether validation passed)
        """
        logger.info(f"Starting walk-forward validation: train={train_window_days}d, test={test_window_days}d")
        
        results = []
        total_windows = 0
        passing_windows = 0
        
        data = data.sort_values('timestamp').reset_index(drop=True)
        start_idx = 0
        
        while start_idx < len(data):
            # Define train and test periods
            train_end_idx = start_idx + train_window_days
            test_end_idx = train_end_idx + test_window_days
            
            if test_end_idx > len(data):
                break
            
            train_data = data.iloc[start_idx:train_end_idx]
            test_data = data.iloc[train_end_idx:test_end_idx]
            
            # Generate signals for test period
            signals = signal_generator(train_data, test_data)
            
            # Run backtest on test period
            result = self.run_backtest(test_data, signals, f"WF_Window_{total_windows}")
            results.append(result)
            
            # Check if window passed
            window_passed = (
                result.win_rate >= 0.50 and
                result.sharpe_ratio >= 0.80 and
                result.max_drawdown <= 0.15 and
                result.net_pnl > 200
            )
            
            if window_passed:
                passing_windows += 1
            
            total_windows += 1
            logger.info(f"Window {total_windows}: Win rate={result.win_rate:.2%}, Sharpe={result.sharpe_ratio:.2f}, Passed={window_passed}")
            
            # Move to next window
            start_idx = train_end_idx
        
        # Check if enough windows passed
        passing_pct = passing_windows / total_windows if total_windows > 0 else 0
        validation_passed = passing_pct >= min_passing_pct
        
        logger.info(f"Walk-forward complete: {passing_windows}/{total_windows} passed ({passing_pct:.1%}), Required: {min_passing_pct:.1%}, Status: {'PASS' if validation_passed else 'FAIL'}")
        
        return results, validation_passed
