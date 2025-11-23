"""Main trading orchestrator to coordinate all components."""

import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger as log

from strategies.fo_strategy import FOStrategy
from strategies.equity_strategy import EquityStrategy
from models.signal_model import SignalModel
from execution.order_executor import OrderExecutor
from data.feature_computer import FeatureComputer


class TradingOrchestrator:
    """
    Orchestrate trading operations across all components.
    
    Coordinates:
    - Data ingestion
    - Signal generation
    - Risk checks
    - Order execution
    - Position monitoring
    """
    
    def __init__(self, config, kite_client, data_loader, risk_manager, logger):
        """Initialize trading orchestrator."""
        self.config = config
        self.kite = kite_client
        self.data_loader = data_loader
        self.risk_manager = risk_manager
        self.audit_logger = logger
        
        # Initialize strategies
        self.fo_strategy = FOStrategy(config)
        self.equity_strategy = EquityStrategy(config)
        
        # Initialize ML model
        self.signal_model = SignalModel(config)
        
        # Initialize order executor
        self.order_executor = OrderExecutor(config, kite_client)
        
        # Initialize feature computer
        self.feature_computer = FeatureComputer(config)
        
        # Tracked instruments
        self.tracked_instruments = config.get("fo_strategy.instruments", ["NIFTY"])
        
        # Position tracking
        self.active_positions: Dict[str, Any] = {}
        
        self.running = False
        
        log.info("Trading orchestrator initialized")
    
    async def start(self):
        """Start the trading bot."""
        self.running = True
        log.info("🚀 Trading bot started")
        
        # Check if models are loaded
        try:
            # Try to load latest models
            from pathlib import Path
            model_dir = Path("models/saved")
            
            if model_dir.exists():
                # Find latest models
                classifier_files = sorted(model_dir.glob("classifier_*.joblib"))
                regressor_files = sorted(model_dir.glob("regressor_*.joblib"))
                
                if classifier_files and regressor_files:
                    self.signal_model.load_models(
                        str(classifier_files[-1]),
                        str(regressor_files[-1])
                    )
                    log.success("ML models loaded")
                else:
                    log.warning("No trained models found. Run train_models.py first.")
            else:
                log.warning("Model directory not found. ML predictions disabled.")
                
        except Exception as e:
            log.warning(f"Could not load models: {e}")
        
        try:
            while self.running:
                # Main trading loop
                await self._trading_cycle()
                
                # Sleep between cycles
                await asyncio.sleep(60)  # 1 minute
                
        except KeyboardInterrupt:
            log.info("Received keyboard interrupt")
            self.running = False
        except Exception as e:
            log.exception(f"Error in trading cycle: {e}")
            self.running = False
    
    async def _trading_cycle(self):
        """Execute one trading cycle."""
        try:
            log.debug("Starting trading cycle...")
            
            # Monitor existing positions
            await self._monitor_positions()
            
            # Check if can generate new signals
            if not self.risk_manager.can_trade():
                log.debug("Trading not allowed by risk manager")
                return
            
            # Generate signals for tracked instruments
            for instrument in self.tracked_instruments:
                await self._process_instrument(instrument)
            
        except Exception as e:
            log.error(f"Trading cycle error: {e}")
    
    async def _process_instrument(self, instrument: str):
        """Process a single instrument for signals."""
        try:
            # Search for instrument
            instruments = await self.data_loader.search_instruments(
                instrument, exchange="NSE"
            )
            
            if not instruments:
                log.debug(f"Instrument {instrument} not found")
                return
            
            inst_data = instruments[0]
            inst_token = inst_data["instrument_token"]
            
            # Get current quote
            quotes = await self.data_loader.get_quote([f"NSE:{instrument}"])
            
            if not quotes:
                return
            
            quote = quotes.get(f"NSE:{instrument}", {})
            
            # Load recent historical data for features
            from datetime import timedelta
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            
            df = await self.data_loader.load_historical_data(
                instrument_token=inst_token,
                from_date=from_date,
                to_date=to_date
            )
            
            if df.empty:
                log.debug(f"No historical data for {instrument}")
                return
            
            # Compute features
            df = self.feature_computer.compute_all_features(df, "equity")
            
            # Get latest features (last row)
            latest_features = df.iloc[-1].to_dict()
            
            # Get ML prediction if model is available
            ml_prediction = None
            if self.signal_model.classifier is not None:
                try:
                    # Prepare features for prediction
                    X = df[self.signal_model.feature_names].iloc[[-1]]
                    prediction = self.signal_model.predict(X)
                    ml_prediction = prediction if isinstance(prediction, dict) else prediction[0]
                except Exception as e:
                    log.debug(f"ML prediction failed: {e}")
            
            # Generate signals from strategies
            market_data = {
                "tradingsymbol": instrument,
                "instrument_token": inst_token,
                "last_price": quote.get("last_price", 0),
                "quote": quote
            }
            
            # Equity signals
            equity_signals = await self.equity_strategy.generate_signals(
                market_data, latest_features, ml_prediction
            )
            
            # Process signals
            for signal in equity_signals:
                await self._process_signal(signal)
            
        except Exception as e:
            log.error(f"Failed to process {instrument}: {e}")
    
    async def _process_signal(self, signal):
        """Process a trading signal."""
        try:
            log.info(f"Processing signal: {signal}")
            
            # Log signal
            self.audit_logger.log_signal(signal.to_dict())
            
            # Manual confirmation check
            if self.config.get("trading_rules.require_manual_confirmation", True):
                log.warning("⚠️  MANUAL CONFIRMATION REQUIRED")
                log.info(f"Signal: {signal}")
                
                # In production, this would trigger UI notification
                # For now, we skip execution
                log.info("Skipping execution (manual confirmation not implemented)")
                return
            
            # Calculate position size
            available_capital = self.config.get("capital.starting_capital", 10000)
            current_positions = len(self.active_positions)
            
            quantity = self.equity_strategy.calculate_position_size(
                signal, available_capital, current_positions
            )
            
            # Execute order (dry run for safety)
            dry_run = not self.config.get("trading_rules.enable_live_trading", False)
            
            order_id = await self.order_executor.execute_signal(
                signal, quantity, dry_run=dry_run
            )
            
            if order_id:
                log.success(f"Order executed: {order_id}")
                
                # Track position
                self.active_positions[signal.signal_id] = {
                    "signal": signal,
                    "order_id": order_id,
                    "quantity": quantity,
                    "entry_price": signal.entry_price,
                    "entry_time": datetime.now(),
                    "strategy_type": signal.strategy_type
                }
                
                # Record trade with risk manager
                self.risk_manager.record_trade(signal.strategy_type)
            
        except Exception as e:
            log.error(f"Failed to process signal: {e}")
    
    async def _monitor_positions(self):
        """Monitor and manage active positions."""
        try:
            if not self.active_positions:
                return
            
            log.debug(f"Monitoring {len(self.active_positions)} positions")
            
            for signal_id, position in list(self.active_positions.items()):
                # Get current price
                instrument = position["signal"].instrument
                quotes = await self.data_loader.get_ltp([f"NSE:{instrument}"])
                
                if not quotes:
                    continue
                
                current_price = quotes.get(f"NSE:{instrument}", 0)
                
                # Check exit conditions
                strategy_type = position["strategy_type"]
                
                if strategy_type.startswith("equity"):
                    should_exit = self.equity_strategy.should_exit(
                        position, current_price, {}
                    )
                else:
                    should_exit = self.fo_strategy.should_exit(
                        position, current_price, {}
                    )
                
                if should_exit:
                    await self._exit_position(signal_id, position, current_price)
            
        except Exception as e:
            log.error(f"Position monitoring error: {e}")
    
    async def _exit_position(self, signal_id: str, position: Dict, exit_price: float):
        """Exit a position."""
        try:
            log.info(f"Exiting position: {signal_id}")
            
            # Calculate P&L
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            
            pnl = (exit_price - entry_price) * quantity
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            
            log.info(f"P&L: {pnl:.2f} INR ({pnl_pct:+.2f}%)")
            
            # Update risk manager
            self.risk_manager.update_pnl(pnl)
            
            # Log exit
            self.audit_logger.log_event("position_exit", {
                "signal_id": signal_id,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "pnl": pnl,
                "pnl_pct": pnl_pct
            })
            
            # Remove from active positions
            del self.active_positions[signal_id]
            
        except Exception as e:
            log.error(f"Failed to exit position: {e}")
    
    async def stop(self):
        """Stop the trading bot."""
        self.running = False
        log.info("Trading bot stopped")
