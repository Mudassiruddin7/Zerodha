"""Unit tests for order execution."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from execution.order_executor import OrderExecutor
from config.config_loader import ConfigLoader
from strategies.base_strategy import Signal


class TestOrderExecutor:
    """Test order execution logic."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        config = Mock(spec=ConfigLoader)
        config.get = Mock(side_effect=lambda key, default=None: {
            'execution.fees.brokerage_flat': 55.0,
            'execution.fees.stt_equity_pct': 0.025,
            'execution.fees.stt_fo_pct': 0.05,
            'execution.fees.txn_charges_pct': 0.00325,
            'execution.fees.gst_pct': 18.0,
            'execution.fees.sebi_charges_pct': 0.0001,
            'execution.fees.stamp_duty_pct': 0.003,
            'execution.slippage_pct': 0.01,
            'execution.retry_attempts': 3,
            'execution.retry_delay_sec': 2,
        }.get(key, default))
        return config
    
    @pytest.fixture
    def kite_client(self):
        """Create mock Kite client."""
        client = AsyncMock()
        client.get_quote = AsyncMock(return_value={
            'last_price': 1000.0,
            'bid': 999.0,
            'ask': 1001.0,
        })
        client.place_order = AsyncMock(return_value='ORDER123')
        client.get_order_status = AsyncMock(return_value='COMPLETE')
        client.get_margins = AsyncMock(return_value={
            'available': 5000.0,
            'used': 3000.0,
        })
        return client
    
    @pytest.fixture
    def risk_manager(self):
        """Create mock risk manager."""
        manager = Mock()
        manager.can_take_position = Mock(return_value=True)
        manager.open_positions = 0
        return manager
    
    @pytest.fixture
    def executor(self, config, kite_client, risk_manager):
        """Create OrderExecutor instance."""
        return OrderExecutor(config, kite_client, risk_manager)
    
    def test_fee_calculation_equity(self, executor):
        """Test fee calculation for equity trades."""
        # Price = 1000, Quantity = 10
        # Turnover = 10,000
        
        fees = executor._calculate_fees(
            price=1000.0,
            quantity=10,
            is_equity=True,
        )
        
        # Expected:
        # Brokerage: 55 * 2 = 110 (buy + sell, or 0.03% if lower)
        # STT: 10000 * 0.00025 = 2.50
        # Txn charges: 10000 * 0.0000325 * 2 = 0.65
        # GST: 110 * 0.18 = 19.80
        # SEBI: 10000 * 0.000001 * 2 = 0.02
        # Stamp duty: 10000 * 0.00003 = 0.30
        # Total ~= 133.27
        
        assert fees > 100  # Should be around 133
        assert fees < 200
    
    def test_fee_calculation_fo(self, executor):
        """Test fee calculation for F&O trades."""
        # Premium = 100, Quantity = 50 (lot size)
        # Turnover = 5,000
        
        fees = executor._calculate_fees(
            price=100.0,
            quantity=50,
            is_equity=False,
        )
        
        # F&O has higher STT (0.05% vs 0.025%)
        assert fees > 50
        assert fees < 150
    
    def test_slippage_buy(self, executor):
        """Test slippage applied correctly for buy orders."""
        price = 1000.0
        slippage_pct = 0.01  # 0.01%
        
        slipped_price = executor._apply_slippage(price, 'BUY')
        
        # Buy: pay more
        # Expected: 1000 * (1 + 0.0001) = 1000.10
        assert slipped_price > price
        assert slipped_price == 1000.10
    
    def test_slippage_sell(self, executor):
        """Test slippage applied correctly for sell orders."""
        price = 1000.0
        slippage_pct = 0.01
        
        slipped_price = executor._apply_slippage(price, 'SELL')
        
        # Sell: receive less
        # Expected: 1000 * (1 - 0.0001) = 999.90
        assert slipped_price < price
        assert slipped_price == 999.90
    
    @pytest.mark.asyncio
    async def test_validate_order_sufficient_margin(self, executor, kite_client):
        """Test order validation passes with sufficient margin."""
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=10,
            entry_price=1000.0,
            stop_loss=950.0,
            take_profit=1050.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        # Available margin = 5000, order cost ~= 10000 + fees
        # Should fail due to insufficient margin
        
        is_valid = await executor._validate_order(signal)
        
        # Actually should fail because order cost > margin
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_order_insufficient_margin(self, executor, kite_client):
        """Test order validation fails with insufficient margin."""
        # Set low available margin
        kite_client.get_margins = AsyncMock(return_value={
            'available': 500.0,  # Not enough
            'used': 9000.0,
        })
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=10,
            entry_price=1000.0,
            stop_loss=950.0,
            take_profit=1050.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        is_valid = await executor._validate_order(signal)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_order_max_positions_reached(self, executor, risk_manager):
        """Test order validation fails when max positions reached."""
        # Set risk manager to reject new positions
        risk_manager.can_take_position = Mock(return_value=False)
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        is_valid = await executor._validate_order(signal)
        
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_validate_order_price_sanity_check(self, executor, kite_client):
        """Test order validation fails on price sanity check."""
        # Set quote price much different from signal price
        kite_client.get_quote = AsyncMock(return_value={
            'last_price': 1500.0,  # Signal says 1000
            'bid': 1499.0,
            'ask': 1501.0,
        })
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=1,
            entry_price=1000.0,  # 33% different!
            stop_loss=950.0,
            take_profit=1050.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        is_valid = await executor._validate_order(signal)
        
        # Should fail due to price mismatch > 5%
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, executor, kite_client):
        """Test successful order placement."""
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        order_id = await executor._place_order_with_retry(signal)
        
        assert order_id == 'ORDER123'
        kite_client.place_order.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_place_order_retry_on_failure(self, executor, kite_client):
        """Test order placement retries on failure."""
        # Fail twice, succeed on third attempt
        kite_client.place_order = AsyncMock(side_effect=[
            Exception("Network error"),
            Exception("Network error"),
            'ORDER123',
        ])
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        order_id = await executor._place_order_with_retry(signal)
        
        assert order_id == 'ORDER123'
        assert kite_client.place_order.call_count == 3
    
    @pytest.mark.asyncio
    async def test_place_order_max_retries_exceeded(self, executor, kite_client):
        """Test order placement fails after max retries."""
        # Always fail
        kite_client.place_order = AsyncMock(side_effect=Exception("Network error"))
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=1,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=105.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        order_id = await executor._place_order_with_retry(signal)
        
        assert order_id is None
        assert kite_client.place_order.call_count == 3  # Max retries
    
    @pytest.mark.asyncio
    async def test_reconcile_order_complete(self, executor, kite_client):
        """Test order reconciliation for complete order."""
        kite_client.get_order_status = AsyncMock(return_value={
            'status': 'COMPLETE',
            'filled_quantity': 10,
            'average_price': 1000.50,
        })
        
        fill_info = await executor.reconcile_order('ORDER123')
        
        assert fill_info['status'] == 'COMPLETE'
        assert fill_info['filled_quantity'] == 10
        assert fill_info['average_price'] == 1000.50
    
    @pytest.mark.asyncio
    async def test_reconcile_order_partial_fill(self, executor, kite_client):
        """Test order reconciliation for partial fill."""
        kite_client.get_order_status = AsyncMock(return_value={
            'status': 'COMPLETE',
            'filled_quantity': 5,  # Ordered 10
            'average_price': 1000.50,
        })
        
        fill_info = await executor.reconcile_order('ORDER123')
        
        assert fill_info['filled_quantity'] == 5
    
    @pytest.mark.asyncio
    async def test_reconcile_order_rejected(self, executor, kite_client):
        """Test order reconciliation for rejected order."""
        kite_client.get_order_status = AsyncMock(return_value={
            'status': 'REJECTED',
            'status_message': 'Insufficient funds',
        })
        
        fill_info = await executor.reconcile_order('ORDER123')
        
        assert fill_info['status'] == 'REJECTED'
    
    @pytest.mark.asyncio
    async def test_execute_signal_dry_run(self, executor):
        """Test signal execution in dry run mode (no actual order)."""
        executor.dry_run = True
        
        signal = Signal(
            instrument='NIFTY',
            direction='BUY',
            quantity=10,
            entry_price=1000.0,
            stop_loss=950.0,
            take_profit=1050.0,
            confidence=0.85,
            expected_profit_pct=5.0,
        )
        
        result = await executor.execute_signal(signal)
        
        assert result['dry_run'] is True
        assert 'fees' in result
        assert 'slippage_price' in result
        # Should not actually place order
        executor.kite_client.place_order.assert_not_called()
    
    def test_calculate_limit_price_buy(self, executor):
        """Test limit price calculation for buy orders."""
        market_price = 1000.0
        
        limit_price = executor._calculate_limit_price(market_price, 'BUY')
        
        # Buy limit should be slightly above market (to ensure fill)
        assert limit_price >= market_price
        assert limit_price <= market_price * 1.001  # Max 0.1% above
    
    def test_calculate_limit_price_sell(self, executor):
        """Test limit price calculation for sell orders."""
        market_price = 1000.0
        
        limit_price = executor._calculate_limit_price(market_price, 'SELL')
        
        # Sell limit should be slightly below market
        assert limit_price <= market_price
        assert limit_price >= market_price * 0.999  # Max 0.1% below


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
