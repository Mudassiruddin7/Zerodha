"""pytest configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    base_price = 1000.0
    
    # Generate realistic price data
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.uniform(-5, 5, n_days),
        'high': prices + np.random.uniform(5, 15, n_days),
        'low': prices - np.random.uniform(5, 15, n_days),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_days),
    })
    
    return data


@pytest.fixture
def sample_features():
    """Create sample feature data for ML testing."""
    np.random.seed(42)
    n_samples = 1000
    
    features = pd.DataFrame({
        'rsi_14': np.random.uniform(30, 70, n_samples),
        'macd': np.random.uniform(-10, 10, n_samples),
        'ema_9': np.random.uniform(1000, 1100, n_samples),
        'ema_21': np.random.uniform(1000, 1100, n_samples),
        'atr_14': np.random.uniform(10, 30, n_samples),
        'volume': np.random.uniform(1000000, 5000000, n_samples),
        'adx': np.random.uniform(15, 40, n_samples),
        'bb_upper': np.random.uniform(1050, 1100, n_samples),
        'bb_lower': np.random.uniform(900, 950, n_samples),
        'vwap': np.random.uniform(1000, 1100, n_samples),
    })
    
    return features


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.get = Mock(side_effect=lambda key, default=None: {
        'capital.starting_capital': 10000.0,
        'capital.max_daily_loss_pct': 7.0,
        'capital.max_weekly_loss_pct': 15.0,
        'risk.max_concurrent_positions': 3,
        'execution.fees.brokerage_flat': 55.0,
        'execution.slippage_pct': 0.01,
        'ml_models.classifier.min_confidence': 0.80,
    }.get(key, default))
    return config


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "edge_case: marks tests as edge case tests"
    )
