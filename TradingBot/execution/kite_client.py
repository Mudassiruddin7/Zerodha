"""Kite Connect MCP client with OAuth authentication and WebSocket support."""

import os
import asyncio
import json
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

from kiteconnect import KiteConnect, KiteTicker


class KiteMCPClient:
    """
    Wrapper for Zerodha Kite Connect API with OAuth flow and WebSocket support.
    
    Handles authentication, token management, market data streaming,
    and REST API operations.
    """
    
    def __init__(self, config):
        """
        Initialize Kite MCP client.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Get API credentials from environment
        self.api_key = os.getenv("KITE_API_KEY")
        self.api_secret = os.getenv("KITE_API_SECRET")
        self.user_id = os.getenv("KITE_USER_ID")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("KITE_API_KEY and KITE_API_SECRET must be set in environment")
        
        # Initialize Kite Connect
        self.kite = KiteConnect(api_key=self.api_key)
        self.ticker: Optional[KiteTicker] = None
        
        # Token management
        self.access_token: Optional[str] = None
        self.token_file = Path("config/.kite_token.json")  # Use same file as authenticator
        
        # WebSocket callbacks
        self.on_tick_callbacks: List[Callable] = []
        self.on_connect_callbacks: List[Callable] = []
        self.on_close_callbacks: List[Callable] = []
        
        # Instruments cache
        self.instruments_cache: Optional[List[Dict]] = None
        self.instruments_by_token: Dict[int, Dict] = {}
        
        logger.info("Kite MCP client initialized")
    
    async def initialize(self):
        """Initialize client with authentication."""
        try:
            # Try to load saved token
            if await self._load_access_token():
                logger.info("Loaded access token from cache")
                self.kite.set_access_token(self.access_token)
                
                # Verify token is valid
                if await self._verify_token():
                    logger.success("Access token is valid")
                    await self._load_instruments()
                    return
                else:
                    logger.warning("Cached token is invalid, need re-authentication")
            
            # Need to authenticate
            await self.authenticate()
            
        except Exception as e:
            logger.error(f"Failed to initialize Kite client: {e}")
            raise
    
    async def authenticate(self):
        """
        Perform OAuth authentication flow.
        
        This requires manual intervention to get the request token
        from the login redirect URL.
        """
        try:
            # Generate login URL
            login_url = self.kite.login_url()
            
            logger.info("=" * 60)
            logger.info("KITE CONNECT AUTHENTICATION REQUIRED")
            logger.info("=" * 60)
            logger.info(f"Please visit this URL to login:\n{login_url}")
            logger.info("After login, you will be redirected to a URL with 'request_token' parameter")
            logger.info("Copy the request_token from the redirect URL")
            logger.info("=" * 60)
            
            # In production, this could be automated with a callback server
            # For now, require manual input
            request_token = input("Enter request_token: ").strip()
            
            if not request_token:
                raise ValueError("Request token is required")
            
            # Generate session
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save token
            await self._save_access_token()
            
            logger.success("Authentication successful!")
            logger.info(f"User ID: {data.get('user_id')}")
            logger.info(f"User Name: {data.get('user_name')}")
            
            # Load instruments
            await self._load_instruments()
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    async def _verify_token(self) -> bool:
        """Verify if the current access token is valid."""
        try:
            profile = self.kite.profile()
            return profile is not None
        except Exception as e:
            logger.debug(f"Token verification failed: {e}")
            return False
    
    async def _save_access_token(self):
        """Save access token to file."""
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use same format as kite_authenticator.py
            token_data = {
                "access_token": self.access_token,
                "created_at": datetime.now().isoformat(),
                "expires_at": datetime.now().replace(hour=15, minute=30, second=0).isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f, indent=2)
            
            logger.debug(f"Access token saved to {self.token_file}")
            
        except Exception as e:
            logger.error(f"Failed to save access token: {e}")
    
    async def _load_access_token(self) -> bool:
        """Load access token from file."""
        try:
            if not self.token_file.exists():
                logger.debug("No cached token found")
                return False
            
            with open(self.token_file, 'r') as f:
                token_data = json.load(f)
            
            self.access_token = token_data.get("access_token")
            
            # Check if token is too old (Kite tokens expire daily at 3:30 PM)
            created_at = datetime.fromisoformat(token_data.get("created_at"))
            expires_at = datetime.fromisoformat(token_data.get("expires_at"))
            
            if datetime.now() > expires_at:
                logger.warning("Access token has expired (daily expiry)")
                return False
            
            logger.debug(f"Loaded token created at {created_at}")
            return self.access_token is not None
            
        except Exception as e:
            logger.debug(f"Failed to load access token: {e}")
            return False
    
    async def _load_instruments(self):
        """Load and cache all instruments."""
        try:
            logger.info("Loading instruments list...")
            
            # Load all instruments
            instruments = self.kite.instruments()
            self.instruments_cache = instruments
            
            # Build token index
            self.instruments_by_token = {
                inst['instrument_token']: inst
                for inst in instruments
            }
            
            logger.success(f"Loaded {len(instruments)} instruments")
            
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            raise
    
    def get_instrument_info(self, instrument_token: int) -> Optional[Dict[str, Any]]:
        """Get instrument information by token."""
        return self.instruments_by_token.get(instrument_token)
    
    async def search_instruments(
        self,
        query: str,
        exchange: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for instruments by name or symbol.
        
        Args:
            query: Search query
            exchange: Filter by exchange
            
        Returns:
            List of matching instruments
        """
        if not self.instruments_cache:
            await self._load_instruments()
        
        query = query.upper()
        results = []
        
        for inst in self.instruments_cache:
            if query in inst['tradingsymbol'].upper() or query in inst.get('name', '').upper():
                if exchange is None or inst['exchange'] == exchange:
                    results.append(inst)
        
        return results
    
    async def get_historical_data(
        self,
        instrument_token: int,
        from_date: datetime,
        to_date: datetime,
        interval: str = "day",
        continuous: bool = False,
        oi: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get historical candle data.
        
        Args:
            instrument_token: Instrument token
            from_date: Start date
            to_date: End date
            interval: Candle interval (minute, day, 3minute, 5minute, etc.)
            continuous: Get continuous data for F&O
            oi: Include open interest
            
        Returns:
            List of candle dictionaries
        """
        try:
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval,
                continuous=continuous,
                oi=oi
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return []
    
    async def get_quotes(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Get market quotes for instruments.
        
        Args:
            instruments: List of instruments (format: "EXCHANGE:SYMBOL")
            
        Returns:
            Dictionary of quotes
        """
        try:
            return self.kite.quote(instruments)
        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}
    
    async def get_ltp(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Get last traded price for instruments.
        
        Args:
            instruments: List of instruments (format: "EXCHANGE:SYMBOL")
            
        Returns:
            Dictionary with LTP data
        """
        try:
            return self.kite.ltp(instruments)
        except Exception as e:
            logger.error(f"Failed to get LTP: {e}")
            return {}
    
    async def get_ohlc(self, instruments: List[str]) -> Dict[str, Any]:
        """
        Get OHLC data for instruments.
        
        Args:
            instruments: List of instruments (format: "EXCHANGE:SYMBOL")
            
        Returns:
            Dictionary with OHLC data
        """
        try:
            return self.kite.ohlc(instruments)
        except Exception as e:
            logger.error(f"Failed to get OHLC: {e}")
            return {}
    
    async def place_order(
        self,
        variety: str,
        exchange: str,
        tradingsymbol: str,
        transaction_type: str,
        quantity: int,
        product: str,
        order_type: str,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        **kwargs
    ) -> Optional[str]:
        """
        Place an order.
        
        Args:
            variety: Order variety (regular, co, amo, etc.)
            exchange: Exchange (NSE, BSE, NFO, etc.)
            tradingsymbol: Trading symbol
            transaction_type: BUY or SELL
            quantity: Order quantity
            product: Product type (CNC, NRML, MIS)
            order_type: Order type (MARKET, LIMIT, SL, SL-M)
            price: Limit price (required for LIMIT orders)
            trigger_price: Trigger price (required for SL orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            order_id = self.kite.place_order(
                variety=variety,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=product,
                order_type=order_type,
                price=price,
                trigger_price=trigger_price,
                **kwargs
            )
            
            logger.info(f"Order placed: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None
    
    async def modify_order(
        self,
        variety: str,
        order_id: str,
        **kwargs
    ) -> Optional[str]:
        """Modify an existing order."""
        try:
            result = self.kite.modify_order(
                variety=variety,
                order_id=order_id,
                **kwargs
            )
            
            logger.info(f"Order modified: {order_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            return None
    
    async def cancel_order(self, variety: str, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self.kite.cancel_order(variety=variety, order_id=order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get all orders for the day."""
        try:
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        try:
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {"net": [], "day": []}
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """Get holdings."""
        try:
            return self.kite.holdings()
        except Exception as e:
            logger.error(f"Failed to get holdings: {e}")
            return []
    
    async def get_margins(self) -> Dict[str, Any]:
        """Get account margins."""
        try:
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Failed to get margins: {e}")
            return {}
    
    def profile(self) -> Dict[str, Any]:
        """Get user profile."""
        try:
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Failed to get profile: {e}")
            return {}
    
    # WebSocket Methods
    
    def init_ticker(self):
        """Initialize WebSocket ticker."""
        if self.ticker is None:
            self.ticker = KiteTicker(self.api_key, self.access_token)
            
            # Assign callbacks
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_close = self._on_close
            self.ticker.on_error = self._on_error
            self.ticker.on_reconnect = self._on_reconnect
            self.ticker.on_noreconnect = self._on_noreconnect
    
    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks."""
        for callback in self.on_tick_callbacks:
            try:
                callback(ticks)
            except Exception as e:
                logger.error(f"Error in tick callback: {e}")
    
    def _on_connect(self, ws, response):
        """Handle WebSocket connection."""
        logger.info("WebSocket connected")
        for callback in self.on_connect_callbacks:
            try:
                callback(response)
            except Exception as e:
                logger.error(f"Error in connect callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Handle WebSocket close."""
        logger.warning(f"WebSocket closed: {code} - {reason}")
        for callback in self.on_close_callbacks:
            try:
                callback(code, reason)
            except Exception as e:
                logger.error(f"Error in close callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {code} - {reason}")
    
    def _on_reconnect(self, ws, attempts_count):
        """Handle WebSocket reconnection."""
        logger.info(f"WebSocket reconnecting (attempt {attempts_count})...")
    
    def _on_noreconnect(self, ws):
        """Handle WebSocket no reconnect."""
        logger.error("WebSocket reconnection failed")
    
    async def subscribe(self, instrument_tokens: List[int], mode: str = "quote"):
        """
        Subscribe to live market data.
        
        Args:
            instrument_tokens: List of instrument tokens
            mode: Subscription mode (ltp, quote, full)
        """
        if self.ticker is None:
            self.init_ticker()
        
        # Map mode to ticker constant
        mode_map = {
            "ltp": self.ticker.MODE_LTP,
            "quote": self.ticker.MODE_QUOTE,
            "full": self.ticker.MODE_FULL
        }
        
        self.ticker.subscribe(instrument_tokens)
        self.ticker.set_mode(mode_map.get(mode, self.ticker.MODE_QUOTE), instrument_tokens)
    
    async def unsubscribe(self, instrument_tokens: List[int]):
        """Unsubscribe from live market data."""
        if self.ticker:
            self.ticker.unsubscribe(instrument_tokens)
    
    def add_tick_callback(self, callback: Callable):
        """Add callback for tick data."""
        self.on_tick_callbacks.append(callback)
    
    def add_connect_callback(self, callback: Callable):
        """Add callback for WebSocket connection."""
        self.on_connect_callbacks.append(callback)
    
    def add_close_callback(self, callback: Callable):
        """Add callback for WebSocket close."""
        self.on_close_callbacks.append(callback)
    
    def start_ticker(self):
        """Start WebSocket ticker (blocking)."""
        if self.ticker:
            logger.info("Starting WebSocket ticker...")
            self.ticker.connect(threaded=False)
    
    def start_ticker_async(self):
        """Start WebSocket ticker in background thread."""
        if self.ticker:
            logger.info("Starting WebSocket ticker (async)...")
            self.ticker.connect(threaded=True)
    
    def stop_ticker(self):
        """Stop WebSocket ticker."""
        if self.ticker:
            logger.info("Stopping WebSocket ticker...")
            self.ticker.close()
    
    async def close(self):
        """Close all connections."""
        self.stop_ticker()
        logger.info("Kite MCP client closed")
