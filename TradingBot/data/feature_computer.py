"""Feature computation for technical indicators and ML model inputs."""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from loguru import logger

try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using manual implementations")
    TALIB_AVAILABLE = False


class FeatureComputer:
    """Compute technical indicators and features for ML models."""
    
    def __init__(self, config):
        """
        Initialize feature computer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.feature_config = config.get("features", {})
        
    def compute_all_features(
        self,
        df: pd.DataFrame,
        instrument_type: str = "equity"
    ) -> pd.DataFrame:
        """
        Compute all configured features for a DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            instrument_type: "equity" or "fo" (futures & options)
            
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            logger.warning("Cannot compute features on empty DataFrame")
            return df
        
        df = df.copy()
        
        logger.info(f"Computing features for {instrument_type}...")
        
        # Momentum features
        df = self._compute_momentum_features(df)
        
        # Mean reversion features
        df = self._compute_mean_reversion_features(df)
        
        # Volatility features
        df = self._compute_volatility_features(df)
        
        # Microstructure features (if available)
        if 'bid' in df.columns and 'ask' in df.columns:
            df = self._compute_microstructure_features(df)
        
        # Options greeks (F&O only)
        if instrument_type == "fo" and self._has_options_data(df):
            df = self._compute_options_features(df)
        
        # Volume features
        df = self._compute_volume_features(df)
        
        # Price action features
        df = self._compute_price_action_features(df)
        
        logger.success(f"Computed {len([c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features")
        
        return df
    
    def _compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute momentum-based technical indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Exponential Moving Averages
        for period in [5, 9, 12, 21, 50, 200]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        if TALIB_AVAILABLE:
            df['rsi_14'] = ta.RSI(close, timeperiod=14)
            df['rsi_7'] = ta.RSI(close, timeperiod=7)
        else:
            df['rsi_14'] = self._compute_rsi(close, 14)
            df['rsi_7'] = self._compute_rsi(close, 7)
        
        # ADX (Average Directional Index)
        if TALIB_AVAILABLE:
            df['adx_14'] = ta.ADX(high, low, close, timeperiod=14)
        else:
            df['adx_14'] = self._compute_adx(df, 14)
        
        # CCI (Commodity Channel Index)
        if TALIB_AVAILABLE:
            df['cci_20'] = ta.CCI(high, low, close, timeperiod=20)
        else:
            df['cci_20'] = self._compute_cci(df, 20)
        
        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = close.pct_change(periods=period) * 100
        
        # Momentum
        df['momentum_10'] = close - close.shift(10)
        
        # Williams %R
        if TALIB_AVAILABLE:
            df['willr_14'] = ta.WILLR(high, low, close, timeperiod=14)
        else:
            df['willr_14'] = self._compute_williams_r(df, 14)
        
        # Stochastic Oscillator
        if TALIB_AVAILABLE:
            df['stoch_k'], df['stoch_d'] = ta.STOCH(high, low, close)
        else:
            df['stoch_k'], df['stoch_d'] = self._compute_stochastic(df)
        
        return df
    
    def _compute_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute mean reversion indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # Bollinger Bands
        for period in [20, 50]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'])
        
        # Z-Score
        for period in [20, 50]:
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            df[f'zscore_{period}'] = (close - mean) / std
        
        # Distance from Moving Averages
        for period in [9, 21, 50]:
            sma = close.rolling(window=period).mean()
            df[f'dist_sma_{period}'] = (close - sma) / sma * 100
        
        return df
    
    def _compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based indicators."""
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # ATR (Average True Range)
        if TALIB_AVAILABLE:
            df['atr_14'] = ta.ATR(high, low, close, timeperiod=14)
        else:
            df['atr_14'] = self._compute_atr(df, 14)
        
        # ATR Ratio (normalized ATR)
        df['atr_ratio'] = df['atr_14'] / close
        
        # Historical Volatility
        for period in [10, 20, 30]:
            returns = close.pct_change()
            df[f'hv_{period}'] = returns.rolling(window=period).std() * np.sqrt(252) * 100
        
        # Keltner Channels
        ema_20 = close.ewm(span=20, adjust=False).mean()
        df['keltner_upper'] = ema_20 + (2 * df['atr_14'])
        df['keltner_lower'] = ema_20 - (2 * df['atr_14'])
        
        # True Range
        df['tr'] = self._compute_true_range(df)
        
        return df
    
    def _compute_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute market microstructure features."""
        
        # Bid-Ask Spread
        if 'bid' in df.columns and 'ask' in df.columns:
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df['bid_ask_spread_pct'] = (df['ask'] - df['bid']) / df['Close'] * 100
            
            # Mid Price
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            
            # Order Imbalance (if bid/ask volumes available)
            if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
                df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        return df
    
    def _compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based indicators."""
        volume = df['Volume']
        close = df['Close']
        
        # Volume Moving Averages
        df['volume_sma_20'] = volume.rolling(window=20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (close * volume).cumsum() / volume.cumsum()
        df['vwap_distance'] = (close - df['vwap']) / df['vwap'] * 100
        
        # OBV (On Balance Volume)
        if TALIB_AVAILABLE:
            df['obv'] = ta.OBV(close, volume)
        else:
            df['obv'] = self._compute_obv(close, volume)
        
        # Money Flow Index
        if TALIB_AVAILABLE:
            df['mfi_14'] = ta.MFI(df['High'], df['Low'], close, volume, timeperiod=14)
        
        # Chaikin Money Flow
        if 'High' in df.columns and 'Low' in df.columns:
            df['cmf_20'] = self._compute_cmf(df, 20)
        
        return df
    
    def _compute_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price action patterns."""
        open_price = df['Open']
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # Candle body and wicks
        df['body_size'] = abs(close - open_price)
        df['upper_wick'] = high - np.maximum(open_price, close)
        df['lower_wick'] = np.minimum(open_price, close) - low
        df['total_range'] = high - low
        
        # Body ratio
        df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, np.nan)
        
        # Gap detection
        df['gap'] = open_price - close.shift(1)
        df['gap_pct'] = (open_price - close.shift(1)) / close.shift(1) * 100
        
        # Higher highs, lower lows
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)
        
        # Doji detection (small body relative to range)
        df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        return df
    
    def _compute_options_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute options-specific features (Greeks, IV, etc.)."""
        
        # These would typically come from the Kite API for F&O instruments
        # Placeholder implementation - actual values should come from API
        
        if 'delta' in df.columns:
            df['delta_normalized'] = df['delta']
        
        if 'gamma' in df.columns:
            df['gamma_normalized'] = df['gamma']
        
        if 'theta' in df.columns:
            df['theta_decay'] = df['theta']
        
        if 'vega' in df.columns:
            df['vega_sensitivity'] = df['vega']
        
        if 'iv' in df.columns:
            # IV Rank (requires historical IV data)
            iv_52w_high = df['iv'].rolling(window=252).max()
            iv_52w_low = df['iv'].rolling(window=252).min()
            df['iv_rank'] = (df['iv'] - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100
            
            # IV Percentile
            df['iv_percentile'] = df['iv'].rolling(window=252).apply(
                lambda x: (x[-1] <= x).sum() / len(x) * 100
            )
        
        # Time value percentage (if premium data available)
        if 'premium' in df.columns and 'intrinsic_value' in df.columns:
            df['time_value'] = df['premium'] - df['intrinsic_value']
            df['time_value_pct'] = df['time_value'] / df['premium'] * 100
        
        return df
    
    def _has_options_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has options-related columns."""
        options_cols = ['delta', 'gamma', 'theta', 'vega', 'iv', 'strike']
        return any(col in df.columns for col in options_cols)
    
    # Helper methods for manual indicator computation (when TA-Lib not available)
    
    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI manually."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR manually."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def _compute_true_range(df: pd.DataFrame) -> pd.Series:
        """Compute True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ADX manually."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = FeatureComputer._compute_true_range(df)
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def _compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute CCI manually."""
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
        
        return (tp - sma) / (0.015 * mad)
    
    @staticmethod
    def _compute_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Williams %R manually."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def _compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Compute Stochastic Oscillator manually."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        
        return k, d
    
    @staticmethod
    def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Compute OBV manually."""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def _compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Compute Chaikin Money Flow."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']
        
        mf_multiplier = ((close - low) - (high - close)) / (high - low)
        mf_volume = mf_multiplier * volume
        
        cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum()
        
        return cmf
    
    def get_feature_names(self, instrument_type: str = "equity") -> List[str]:
        """
        Get list of all computed feature names.
        
        Args:
            instrument_type: "equity" or "fo"
            
        Returns:
            List of feature names
        """
        # Create dummy DataFrame and compute features to get column names
        dummy_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [102, 103, 104],
            'Low': [99, 100, 101],
            'Close': [101, 102, 103],
            'Volume': [1000, 1100, 1200]
        })
        
        result_df = self.compute_all_features(dummy_df, instrument_type)
        
        # Return all columns except OHLCV
        base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return [col for col in result_df.columns if col not in base_cols]
