"""Data storage for historical market data using Parquet format."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


class DataStorage:
    """Handle storage and retrieval of historical market data."""
    
    def __init__(self, config):
        """
        Initialize data storage.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.storage_type = config.get("data.sources.historical_storage", "parquet")
        self.base_path = Path("data/historical")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data storage initialized at {self.base_path}")
    
    def _get_file_path(
        self,
        instrument_token: int,
        interval: str = "day"
    ) -> Path:
        """
        Get file path for an instrument's data.
        
        Args:
            instrument_token: Instrument token
            interval: Candle interval
            
        Returns:
            Path to data file
        """
        filename = f"{instrument_token}_{interval}.parquet"
        return self.base_path / filename
    
    def save_historical(
        self,
        df: pd.DataFrame,
        instrument_token: int,
        interval: str = "day"
    ):
        """
        Save historical data to storage.
        
        Args:
            df: DataFrame with OHLCV data
            instrument_token: Instrument token
            interval: Candle interval
        """
        try:
            if df.empty:
                logger.warning("Cannot save empty DataFrame")
                return
            
            file_path = self._get_file_path(instrument_token, interval)
            
            # Convert index to column if it's a DatetimeIndex
            if isinstance(df.index, pd.DatetimeIndex):
                df_to_save = df.reset_index()
            else:
                df_to_save = df.copy()
            
            # Save to parquet
            df_to_save.to_parquet(
                file_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )
            
            logger.debug(f"Saved {len(df)} rows to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")
    
    def load_historical(
        self,
        instrument_token: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        interval: str = "day"
    ) -> Optional[pd.DataFrame]:
        """
        Load historical data from storage.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date (optional)
            to_date: End date (optional)
            interval: Candle interval
            
        Returns:
            DataFrame with OHLCV data or None if not found
        """
        try:
            file_path = self._get_file_path(instrument_token, interval)
            
            if not file_path.exists():
                return None
            
            # Load from parquet
            df = pd.read_parquet(file_path, engine='pyarrow')
            
            # Set date as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            
            # Filter by date range if specified
            if from_date is not None:
                df = df[df.index >= from_date]
            if to_date is not None:
                df = df[df.index <= to_date]
            
            logger.debug(f"Loaded {len(df)} rows from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return None
    
    def append_historical(
        self,
        df: pd.DataFrame,
        instrument_token: int,
        interval: str = "day"
    ):
        """
        Append new data to existing historical data.
        
        Args:
            df: DataFrame with new OHLCV data
            instrument_token: Instrument token
            interval: Candle interval
        """
        try:
            # Load existing data
            existing_df = self.load_historical(instrument_token, interval=interval)
            
            if existing_df is None or existing_df.empty:
                # No existing data, just save new data
                self.save_historical(df, instrument_token, interval)
                return
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # Save combined data
            self.save_historical(combined_df, instrument_token, interval)
            
            logger.debug(f"Appended {len(df)} rows, total now {len(combined_df)} rows")
            
        except Exception as e:
            logger.error(f"Failed to append historical data: {e}")
    
    def delete_historical(self, instrument_token: int, interval: str = "day"):
        """
        Delete historical data file.
        
        Args:
            instrument_token: Instrument token
            interval: Candle interval
        """
        try:
            file_path = self._get_file_path(instrument_token, interval)
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted historical data: {file_path}")
            else:
                logger.warning(f"File not found: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to delete historical data: {e}")
    
    def list_stored_instruments(self, interval: Optional[str] = None) -> list[int]:
        """
        List all instruments with stored data.
        
        Args:
            interval: Filter by interval (optional)
            
        Returns:
            List of instrument tokens
        """
        try:
            instruments = []
            
            for file in self.base_path.glob("*.parquet"):
                parts = file.stem.split("_")
                if len(parts) >= 2:
                    token = int(parts[0])
                    file_interval = parts[1]
                    
                    if interval is None or file_interval == interval:
                        instruments.append(token)
            
            return sorted(set(instruments))
            
        except Exception as e:
            logger.error(f"Failed to list stored instruments: {e}")
            return []
    
    def get_storage_size(self) -> float:
        """
        Get total size of stored data in MB.
        
        Returns:
            Total size in megabytes
        """
        try:
            total_size = sum(
                f.stat().st_size for f in self.base_path.glob("*.parquet")
            )
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Failed to get storage size: {e}")
            return 0.0
