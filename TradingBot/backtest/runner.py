"""Backtest runner and configuration."""

import asyncio
import pandas as pd
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
import logging

from config.config_loader import ConfigLoader
from backtest.engine import BacktestEngine, BacktestResult
from backtest.metrics import PerformanceMetrics
from data.loader import DataLoader
from data.feature_computer import FeatureComputer
from models.signal_model import SignalModel

logger = logging.getLogger(__name__)


class BacktestRunner:
    """
    High-level runner for backtesting strategies.
    
    Handles:
    - Data loading
    - Feature computation
    - Signal generation
    - Backtest execution
    - Results analysis
    """
    
    def __init__(
        self,
        config: ConfigLoader,
        data_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize backtest runner.
        
        Args:
            config: Configuration loader
            data_loader: Optional DataLoader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.feature_computer = FeatureComputer()
        
        logger.info("Initialized backtest runner")
    
    async def run_strategy_backtest(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        strategy,
        model: Optional[SignalModel] = None,
        initial_capital: float = 10000.0,
    ) -> BacktestResult:
        """
        Run backtest for a strategy on an instrument.
        
        Args:
            instrument: Instrument symbol (e.g., 'NIFTY', 'RELIANCE')
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy: Strategy instance (FOStrategy, EquityStrategy, etc.)
            model: Optional ML model for signal generation
            initial_capital: Starting capital
            
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Running backtest for {instrument} from {start_date.date()} to {end_date.date()}")
        
        # Load historical data
        if self.data_loader is None:
            raise ValueError("DataLoader not initialized")
        
        data = await self.data_loader.get_historical_data(
            instrument=instrument,
            from_date=start_date,
            to_date=end_date,
            interval='day',
        )
        
        if data.empty:
            raise ValueError(f"No data available for {instrument}")
        
        logger.info(f"Loaded {len(data)} bars for {instrument}")
        
        # Compute features
        features_df = self.feature_computer.compute_all_features(data)
        logger.info(f"Computed {len(features_df.columns)} features")
        
        # Generate signals
        signals = await self._generate_signals(features_df, strategy, model)
        
        # Run backtest
        engine = BacktestEngine(
            config=self.config,
            initial_capital=initial_capital,
            slippage_pct=self.config.get('execution.slippage_pct', 0.01),
            commission_per_trade=self.config.get('execution.fees.brokerage_flat', 55.0),
        )
        
        result = engine.run_backtest(
            data=data,
            signals=signals,
            strategy_name=strategy.__class__.__name__,
        )
        
        return result
    
    async def _generate_signals(
        self,
        features: pd.DataFrame,
        strategy,
        model: Optional[SignalModel] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals from features.
        
        Args:
            features: DataFrame with computed features
            strategy: Strategy instance
            model: Optional ML model
            
        Returns:
            DataFrame with columns ['timestamp', 'signal', 'confidence', 'expected_return']
        """
        signals_list = []
        
        for idx, row in features.iterrows():
            # Convert row to dict
            market_data = row.to_dict()
            
            # Generate strategy signal
            signal = await strategy.generate_signal(market_data)
            
            if signal and model:
                # Get ML prediction
                prediction = model.predict(features.iloc[[idx]])
                
                # Override signal based on ML confidence
                min_confidence = self.config.get('ml_models.classifier.min_confidence', 0.80)
                if prediction['confidence'] < min_confidence:
                    signal = None
            
            if signal:
                signals_list.append({
                    'timestamp': row.get('timestamp', idx),
                    'signal': 1 if signal.direction == 'BUY' else -1,
                    'confidence': signal.confidence,
                    'expected_return': signal.expected_profit_pct,
                })
            else:
                signals_list.append({
                    'timestamp': row.get('timestamp', idx),
                    'signal': 0,
                    'confidence': 0.0,
                    'expected_return': 0.0,
                })
        
        return pd.DataFrame(signals_list)
    
    async def run_walk_forward_validation(
        self,
        instrument: str,
        start_date: datetime,
        end_date: datetime,
        strategy,
        model_trainer: Callable,
        train_window_days: int = 180,
        test_window_days: int = 30,
    ) -> tuple:
        """
        Run walk-forward validation.
        
        Args:
            instrument: Instrument symbol
            start_date: Start date
            end_date: End date
            strategy: Strategy instance
            model_trainer: Function to train model on data
            train_window_days: Training window size
            test_window_days: Testing window size
            
        Returns:
            (list of results, validation passed)
        """
        logger.info(f"Running walk-forward validation for {instrument}")
        
        # Load all data
        data = await self.data_loader.get_historical_data(
            instrument=instrument,
            from_date=start_date,
            to_date=end_date,
            interval='day',
        )
        
        # Compute features
        features = self.feature_computer.compute_all_features(data)
        
        # Define signal generator
        def signal_generator(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
            """Generate signals for test period using model trained on train period."""
            # Train model on train data
            model = model_trainer(train_data)
            
            # Generate signals for test data
            return asyncio.run(self._generate_signals(test_data, strategy, model))
        
        # Run walk-forward
        engine = BacktestEngine(
            config=self.config,
            initial_capital=10000.0,
        )
        
        results, passed = engine.walk_forward_validation(
            data=data,
            signal_generator=signal_generator,
            train_window_days=train_window_days,
            test_window_days=test_window_days,
            min_passing_pct=0.80,
        )
        
        return results, passed
    
    def analyze_results(
        self,
        result: BacktestResult,
        verbose: bool = True,
        save_plot: str = None,
    ):
        """
        Analyze and print backtest results.
        
        Args:
            result: BacktestResult to analyze
            verbose: Print detailed summary
            save_plot: Optional path to save equity curve plot
        """
        # Print summary
        PerformanceMetrics.print_summary(result, verbose=verbose)
        
        # Plot equity curve
        if save_plot:
            PerformanceMetrics.plot_equity_curve(result, save_path=save_plot)
        
        # Trade distribution
        distribution = PerformanceMetrics.analyze_trade_distribution(result.trades)
        
        if distribution:
            print("\nTRADE DISTRIBUTION:")
            print("-" * 60)
            print("P&L Distribution (INR):")
            for key, value in distribution['pnl_distribution'].items():
                print(f"  {key:10}: ₹{value:10.2f}")
            
            print("\nDuration Distribution (Hours):")
            for key, value in distribution['duration_distribution_hours'].items():
                print(f"  {key:10}: {value:10.2f}")
            print("-" * 60)
