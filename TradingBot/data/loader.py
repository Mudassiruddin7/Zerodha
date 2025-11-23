"""Data loader for historical and real-time market data."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

from data.storage import DataStorage


class DataLoader:
    """Load historical and real-time market data from Kite Connect API."""
    
    def __init__(self, config, kite_client):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object
            kite_client: KiteMCPClient instance
        """
        self.config = config
        self.kite = kite_client
        self.storage = DataStorage(config)
        
        # Data configuration
        self.lookback_days = config.get("data.historical.lookback_days", 90)
        self.candle_interval = config.get("data.historical.candle_interval", "day")
        self.intraday_interval = config.get("data.historical.intraday_interval", "5minute")
        
    async def load_historical_data(
        self,
        instrument_token: int,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        interval: Optional[str] = None,
        continuous: bool = False
    ) -> pd.DataFrame:
        """
        Load historical candle data for an instrument.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date (defaults to lookback_days ago)
            to_date: End date (defaults to today)
            interval: Candle interval (defaults to config value)
            continuous: Whether to get continuous data for F&O
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Set default dates
            if to_date is None:
                to_date = datetime.now()
            if from_date is None:
                from_date = to_date - timedelta(days=self.lookback_days)
            
            if interval is None:
                interval = self.candle_interval
            
            logger.info(
                f"Loading historical data for token {instrument_token} "
                f"from {from_date.date()} to {to_date.date()}, interval={interval}"
            )
            
            # Check if data exists in storage
            cached_data = self.storage.load_historical(
                instrument_token, from_date, to_date, interval
            )
            
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Loaded {len(cached_data)} candles from cache")
                return cached_data
            
            # Fetch from API
            data = await self.kite.get_historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No historical data available for token {instrument_token}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Rename columns to standard names
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Add OI if available (F&O only)
            if 'oi' in df.columns:
                df = df.rename(columns={'oi': 'OI'})
            
            logger.success(f"Loaded {len(df)} candles from API")
            
            # Save to storage
            self.storage.save_historical(df, instrument_token, interval)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            return pd.DataFrame()
    
    async def load_multiple_instruments(
        self,
        instrument_tokens: List[int],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        interval: Optional[str] = None
    ) -> Dict[int, pd.DataFrame]:
        """
        Load historical data for multiple instruments.
        
        Args:
            instrument_tokens: List of instrument tokens
            from_date: Start date
            to_date: End date
            interval: Candle interval
            
        Returns:
            Dictionary mapping instrument token to DataFrame
        """
        results = {}
        
        for token in instrument_tokens:
            df = await self.load_historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            results[token] = df
        
        return results
    
    async def get_quote(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Get real-time quotes for instruments.
        
        Args:
            instruments: List of instruments in format "EXCHANGE:SYMBOL"
            
        Returns:
            Dictionary of quotes
        """
        try:
            quotes = await self.kite.get_quotes(instruments)
            return quotes
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}
    
    async def get_ltp(self, instruments: List[str]) -> Dict[str, float]:
        """
        Get last traded price for instruments.
        
        Args:
            instruments: List of instruments in format "EXCHANGE:SYMBOL"
            
        Returns:
            Dictionary mapping instrument to LTP
        """
        try:
            ltp_data = await self.kite.get_ltp(instruments)
            return {k: v['last_price'] for k, v in ltp_data.items()}
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
            return {}
    
    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for instruments.
        
        Args:
            query: Search query
            exchange: Filter by exchange (NSE, BSE, NFO, etc.)
            
        Returns:
            List of matching instruments
        """
        try:
            instruments = await self.kite.search_instruments(query)
            
            if exchange:
                instruments = [i for i in instruments if i.get('exchange') == exchange]
            
            return instruments
        except Exception as e:
            logger.error(f"Failed to search instruments: {e}")
            return []
    
    async def get_option_chain(
        self,
        symbol: str,
        expiry: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get option chain for an underlying.
        
        Args:
            symbol: Underlying symbol (e.g., "NIFTY", "BANKNIFTY")
            expiry: Expiry date (defaults to nearest expiry)
            
        Returns:
            DataFrame with option chain data
        """
        try:
            # Search for options
            instruments = await self.search_instruments(symbol, exchange="NFO")
            
            # Filter options
            options = [
                i for i in instruments
                if i.get('instrument_type') in ['CE', 'PE']
            ]
            
            if expiry:
                options = [
                    i for i in options
                    if i.get('expiry') == expiry
                ]
            
            # Convert to DataFrame
            df = pd.DataFrame(options)
            
            if df.empty:
                logger.warning(f"No options found for {symbol}")
                return df
            
            # Sort by strike
            df = df.sort_values('strike')
            
            logger.info(f"Loaded option chain for {symbol}: {len(df)} strikes")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get option chain: {e}")
            return pd.DataFrame()
    
    def get_instrument_info(self, instrument_token: int) -> Optional[Dict[str, Any]]:
        """
        Get instrument information.
        
        Args:
            instrument_token: Instrument token
            
        Returns:
            Instrument info dictionary
        """
        try:
            return self.kite.get_instrument_info(instrument_token)
        except Exception as e:
            logger.error(f"Failed to get instrument info: {e}")
            return None
    
    async def subscribe_live_data(
        self,
        instrument_tokens: List[int],
        mode: str = "quote"
    ):
        """
        Subscribe to live market data via WebSocket.
        
        Args:
            instrument_tokens: List of instrument tokens
            mode: Subscription mode (ltp, quote, full)
        """
        try:
            await self.kite.subscribe(instrument_tokens, mode)
            logger.info(f"Subscribed to {len(instrument_tokens)} instruments, mode={mode}")
        except Exception as e:
            logger.error(f"Failed to subscribe to live data: {e}")
    
    async def unsubscribe_live_data(self, instrument_tokens: List[int]):
        """
        Unsubscribe from live market data.
        
        Args:
            instrument_tokens: List of instrument tokens
        """
        try:
            await self.kite.unsubscribe(instrument_tokens)
            logger.info(f"Unsubscribed from {len(instrument_tokens)} instruments")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from live data: {e}")
