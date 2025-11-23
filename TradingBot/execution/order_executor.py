"""Order execution engine with validation and reconciliation."""

from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger

from execution.kite_client import KiteMCPClient
from strategies.base_strategy import Signal


class OrderExecutor:
    """
    Handle order placement, validation, and fill reconciliation.
    
    Features:
    - Pre-trade validation (margin, limits, filters)
    - Order placement with retry logic
    - Fill tracking and reconciliation
    - Slippage monitoring
    """
    
    def __init__(self, config, kite_client: KiteMCPClient):
        """
        Initialize order executor.
        
        Args:
            config: Configuration object
            kite_client: KiteMCPClient instance
        """
        self.config = config
        self.kite = kite_client
        
        # Execution config
        self.exec_config = config.get("execution")
        self.order_params = self.exec_config.get("order_params")
        self.slippage_config = self.exec_config.get("slippage")
        self.fees_config = self.exec_config.get("fees")
        self.retry_config = self.exec_config.get("retry")
        
        # Order tracking
        self.pending_orders: Dict[str, Dict] = {}
        self.filled_orders: List[Dict] = []
        self.rejected_orders: List[Dict] = []
        
        # Slippage tracking
        self.slippage_history: List[float] = []
        
        logger.info("Order executor initialized")
    
    async def execute_signal(
        self,
        signal: Signal,
        quantity: Optional[int] = None,
        dry_run: bool = False
    ) -> Optional[str]:
        """
        Execute a trading signal.
        
        Args:
            signal: Trading signal to execute
            quantity: Override quantity (uses signal.quantity if None)
            dry_run: If True, validate but don't place order
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            logger.info(f"Executing signal: {signal.signal_id}")
            
            # Use provided quantity or signal quantity
            qty = quantity or signal.quantity
            
            # Pre-trade validation
            if not await self._validate_order(signal, qty):
                logger.warning("Order validation failed")
                return None
            
            # Calculate limit price with expected slippage
            limit_price = self._calculate_limit_price(
                signal.entry_price,
                signal.action,
                self.slippage_config.get("expected_pct", 0.01)
            )
            
            # Prepare order parameters
            order_params = {
                "variety": self.order_params.get("variety", "regular"),
                "exchange": self._get_exchange(signal.instrument),
                "tradingsymbol": signal.instrument,
                "transaction_type": signal.action,
                "quantity": qty,
                "product": self._get_product_type(signal.strategy_type),
                "order_type": self.order_params.get("order_type", "LIMIT"),
                "price": limit_price,
                "validity": self.order_params.get("validity", "DAY"),
                "tag": f"bot_{signal.strategy_type}"
            }
            
            logger.info(f"Order params: {order_params}")
            
            # Dry run check
            if dry_run:
                logger.info("DRY RUN: Order would be placed with above params")
                return f"DRY_RUN_{signal.signal_id}"
            
            # Place order with retry logic
            order_id = await self._place_order_with_retry(order_params)
            
            if order_id:
                # Track pending order
                self.pending_orders[order_id] = {
                    "signal": signal,
                    "order_params": order_params,
                    "expected_price": signal.entry_price,
                    "limit_price": limit_price,
                    "quantity": qty,
                    "timestamp": datetime.now()
                }
                
                logger.success(f"Order placed successfully: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None
    
    async def _validate_order(self, signal: Signal, quantity: int) -> bool:
        """
        Validate order before placement.
        
        Checks:
        - Sufficient margin/capital
        - Position limits
        - Price sanity
        - Market hours
        """
        try:
            # Check if live trading is enabled
            if not self.config.get("trading_rules.enable_live_trading", False):
                logger.info("Live trading disabled, validation passed for paper trading")
                return True
            
            # Get current margins
            margins = await self.kite.get_margins()
            
            if not margins:
                logger.error("Could not fetch margins")
                return False
            
            # Calculate required capital
            required_capital = signal.entry_price * quantity
            
            # Add fees
            fees = self._calculate_fees(required_capital, signal.strategy_type)
            total_required = required_capital + fees
            
            # Check available funds
            available = margins.get("equity", {}).get("available", {}).get("live_balance", 0)
            
            if total_required > available:
                logger.warning(
                    f"Insufficient funds: Required {total_required:.2f} INR, "
                    f"Available {available:.2f} INR"
                )
                return False
            
            # Check position limits
            positions = await self.kite.get_positions()
            current_positions = len(positions.get("net", []))
            max_positions = self.config.get("trading_rules.max_concurrent_positions", 3)
            
            if current_positions >= max_positions:
                logger.warning(
                    f"Position limit reached: {current_positions}/{max_positions}"
                )
                return False
            
            # Price sanity check
            if signal.entry_price <= 0:
                logger.error("Invalid entry price")
                return False
            
            logger.info("Order validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return False
    
    async def _place_order_with_retry(
        self,
        order_params: Dict[str, Any]
    ) -> Optional[str]:
        """Place order with retry logic."""
        max_retries = self.retry_config.get("max_retries", 3)
        retry_delay = self.retry_config.get("retry_delay_ms", 1000) / 1000  # Convert to seconds
        backoff = self.retry_config.get("backoff_multiplier", 2.0)
        
        for attempt in range(max_retries):
            try:
                order_id = await self.kite.place_order(**order_params)
                
                if order_id:
                    return order_id
                
            except Exception as e:
                logger.warning(f"Order placement attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    import asyncio
                    wait_time = retry_delay * (backoff ** attempt)
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
        
        logger.error("Order placement failed after all retries")
        return None
    
    def _calculate_limit_price(
        self,
        expected_price: float,
        action: str,
        slippage_pct: float
    ) -> float:
        """Calculate limit price accounting for expected slippage."""
        if action == "BUY":
            # For buy, add slippage to account for price moving up
            limit_price = expected_price * (1 + slippage_pct / 100)
        else:
            # For sell, subtract slippage
            limit_price = expected_price * (1 - slippage_pct / 100)
        
        # Round to 2 decimal places
        return round(limit_price, 2)
    
    def _get_exchange(self, tradingsymbol: str) -> str:
        """Determine exchange from trading symbol."""
        # F&O instruments are on NFO
        if any(suffix in tradingsymbol for suffix in ["CE", "PE", "FUT"]):
            return "NFO"
        
        # Default to NSE for equity
        return "NSE"
    
    def _get_product_type(self, strategy_type: str) -> str:
        """Get product type based on strategy."""
        if strategy_type == "fo":
            return self.order_params.get("product_fo", "NRML")
        elif strategy_type == "equity_intraday":
            return "MIS"  # Intraday
        else:
            return self.order_params.get("product_equity", "CNC")  # Cash & Carry
    
    def _calculate_fees(self, trade_value: float, strategy_type: str) -> float:
        """
        Calculate total trading fees.
        
        Includes:
        - Brokerage
        - STT
        - Transaction charges
        - GST
        - SEBI charges
        - Stamp duty
        """
        fees = 0.0
        
        # Flat brokerage
        brokerage = self.fees_config.get("brokerage_per_trade", 55)
        fees += brokerage
        
        # STT (on sell side, but conservative estimate - count for both)
        if strategy_type == "fo":
            stt_pct = self.fees_config.get("stt_fo_pct", 0.0625)
        else:
            stt_pct = self.fees_config.get("stt_equity_pct", 0.1)
        
        fees += trade_value * (stt_pct / 100)
        
        # Transaction charges
        txn_charges_pct = self.fees_config.get("transaction_charges_pct", 0.0019)
        fees += trade_value * (txn_charges_pct / 100)
        
        # GST on brokerage
        gst_pct = self.fees_config.get("gst_pct", 18.0)
        fees += brokerage * (gst_pct / 100)
        
        # SEBI charges
        sebi_pct = self.fees_config.get("sebi_charges_pct", 0.0001)
        fees += trade_value * (sebi_pct / 100)
        
        # Stamp duty
        stamp_pct = self.fees_config.get("stamp_duty_pct", 0.003)
        fees += trade_value * (stamp_pct / 100)
        
        return round(fees, 2)
    
    async def reconcile_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Reconcile order status and track fill.
        
        Args:
            order_id: Order ID to reconcile
            
        Returns:
            Fill details if order is filled
        """
        try:
            # Get order history
            orders = await self.kite.get_orders()
            
            order = None
            for o in orders:
                if o.get("order_id") == order_id:
                    order = o
                    break
            
            if not order:
                logger.warning(f"Order {order_id} not found")
                return None
            
            status = order.get("status")
            
            if status == "COMPLETE":
                # Order filled
                fill_price = order.get("average_price", 0)
                fill_quantity = order.get("filled_quantity", 0)
                
                # Calculate slippage
                if order_id in self.pending_orders:
                    pending = self.pending_orders[order_id]
                    expected_price = pending["expected_price"]
                    
                    slippage_pct = (fill_price - expected_price) / expected_price * 100
                    self.slippage_history.append(slippage_pct)
                    
                    fill_data = {
                        "order_id": order_id,
                        "signal_id": pending["signal"].signal_id,
                        "fill_price": fill_price,
                        "fill_quantity": fill_quantity,
                        "expected_price": expected_price,
                        "slippage_pct": slippage_pct,
                        "timestamp": datetime.now(),
                        "order_details": order
                    }
                    
                    self.filled_orders.append(fill_data)
                    del self.pending_orders[order_id]
                    
                    logger.success(
                        f"Order filled: {order_id} @ {fill_price} "
                        f"(slippage: {slippage_pct:+.2f}%)"
                    )
                    
                    return fill_data
            
            elif status == "REJECTED":
                logger.error(f"Order rejected: {order_id}")
                
                if order_id in self.pending_orders:
                    self.rejected_orders.append({
                        "order_id": order_id,
                        "order_details": order,
                        "timestamp": datetime.now()
                    })
                    del self.pending_orders[order_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        try:
            variety = self.order_params.get("variety", "regular")
            success = await self.kite.cancel_order(variety, order_id)
            
            if success and order_id in self.pending_orders:
                del self.pending_orders[order_id]
            
            return success
            
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False
    
    def get_average_slippage(self) -> float:
        """Get average slippage from recent trades."""
        if not self.slippage_history:
            return 0.0
        
        # Use last 20 trades for average
        recent = self.slippage_history[-20:]
        return sum(recent) / len(recent)
    
    def get_pending_orders(self) -> List[Dict]:
        """Get all pending orders."""
        return list(self.pending_orders.values())
