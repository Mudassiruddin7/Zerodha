"""F&O (Futures & Options) trading strategy - Long options only."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal


class FOStrategy(BaseStrategy):
    """
    F&O strategy for trading long call and put options.
    
    Features:
    - Strike selection based on delta, OI, volume, bid-ask spread
    - Entry rules: MACD cross, RSI filters, ML model approval
    - Exit rules: Take profit (50% gain), stop loss (60% loss), time-based
    - Position sizing: Risk-based with Kelly fraction
    """
    
    def __init__(self, config):
        """Initialize F&O strategy."""
        super().__init__(config, "F&O Strategy")
        
        # Strategy configuration
        self.fo_config = config.get("fo_strategy")
        self.enabled = self.fo_config.get("enabled", True)
        self.instruments = self.fo_config.get("instruments", ["NIFTY"])
        
        # Strike selection criteria
        self.strike_config = self.fo_config.get("strike_selection")
        self.min_delta = self.strike_config.get("min_delta", 0.20)
        self.max_delta = self.strike_config.get("max_delta", 0.40)
        self.min_oi = self.strike_config.get("min_open_interest", 1000)
        self.min_volume = self.strike_config.get("min_volume", 500)
        self.max_spread_pct = self.strike_config.get("max_bid_ask_spread_pct", 2.0)
        
        # Entry rules
        self.entry_config = self.fo_config.get("entry_rules")
        self.min_confidence = self.entry_config.get("min_model_confidence", 0.80)
        self.min_expected_profit = self.entry_config.get("min_expected_profit_inr", 150)
        
        # Exit rules
        self.exit_config = self.fo_config.get("exit_rules")
        self.take_profit_pct = self.exit_config.get("take_profit_pct", 50.0)
        self.stop_loss_pct = self.exit_config.get("stop_loss_pct", 60.0)
        self.exit_days_before_expiry = self.exit_config.get("exit_days_before_expiry", 1)
        
        # Position sizing
        self.sizing_config = self.fo_config.get("position_sizing")
        self.risk_per_trade_pct = self.sizing_config.get("risk_per_trade_pct", 5.0)
        self.kelly_fraction = self.sizing_config.get("kelly_fraction", 0.25)
        
        logger.info(f"F&O Strategy configured for instruments: {self.instruments}")
    
    async def generate_signals(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> List[Signal]:
        """
        Generate F&O trading signals.
        
        Args:
            market_data: Current market data including option chain
            features: Technical indicators
            ml_prediction: ML model predictions
            
        Returns:
            List of trading signals
        """
        signals = []
        
        if not self.enabled:
            logger.debug("F&O strategy is disabled")
            return signals
        
        try:
            # Check ML model approval
            if ml_prediction is None:
                logger.debug("No ML prediction available")
                return signals
            
            if ml_prediction.get("p_win", 0) < self.min_confidence:
                logger.debug(
                    f"ML confidence {ml_prediction.get('p_win', 0):.2%} "
                    f"below threshold {self.min_confidence:.2%}"
                )
                return signals
            
            # Check technical indicators
            if not self._check_entry_conditions(features):
                logger.debug("Entry conditions not met")
                return signals
            
            # Determine trade direction (CALL or PUT)
            direction = self._determine_direction(features, ml_prediction)
            
            if direction is None:
                logger.debug("No clear directional bias")
                return signals
            
            # Get option chain
            option_chain = market_data.get("option_chain")
            if option_chain is None or option_chain.empty:
                logger.warning("Option chain not available")
                return signals
            
            # Select best strike
            selected_strike = self._select_strike(
                option_chain,
                direction,
                market_data.get("spot_price")
            )
            
            if selected_strike is None:
                logger.debug("No suitable strike found")
                return signals
            
            # Calculate expected profit
            entry_price = selected_strike["last_price"]
            lot_size = selected_strike.get("lot_size", 1)
            fees = self.config.get("execution.fees.brokerage_per_trade", 55)
            
            expected_profit = self.calculate_expected_profit(
                entry_price=entry_price,
                quantity=lot_size,
                expected_return=ml_prediction.get("expected_return", 0),
                fees=fees
            )
            
            if expected_profit < self.min_expected_profit:
                logger.debug(
                    f"Expected profit {expected_profit:.2f} INR below minimum "
                    f"{self.min_expected_profit} INR"
                )
                return signals
            
            # Create signal
            signal = Signal(
                instrument=selected_strike["tradingsymbol"],
                instrument_token=selected_strike["instrument_token"],
                action="BUY",
                strategy_type="fo",
                entry_price=entry_price,
                quantity=lot_size,
                confidence=ml_prediction.get("p_win", 0),
                expected_return=ml_prediction.get("expected_return", 0),
                stop_loss=entry_price * (1 - self.stop_loss_pct / 100),
                take_profit=entry_price * (1 + self.take_profit_pct / 100),
                metadata={
                    "strike": selected_strike["strike"],
                    "option_type": direction,
                    "expiry": selected_strike["expiry"],
                    "delta": selected_strike.get("delta"),
                    "iv": selected_strike.get("iv"),
                    "oi": selected_strike.get("oi"),
                    "volume": selected_strike.get("volume"),
                    "expected_profit_inr": expected_profit,
                    "ml_prediction": ml_prediction,
                    "technical_features": {
                        "macd": features.get("macd"),
                        "rsi": features.get("rsi_14"),
                        "adx": features.get("adx_14")
                    }
                }
            )
            
            signals.append(signal)
            logger.info(f"F&O signal generated: {signal}")
            
        except Exception as e:
            logger.error(f"Failed to generate F&O signals: {e}")
        
        return signals
    
    def _check_entry_conditions(self, features: Dict[str, Any]) -> bool:
        """Check if technical entry conditions are met."""
        # MACD signal
        macd = features.get("macd", 0)
        macd_signal = features.get("macd_signal", 0)
        macd_threshold = self.entry_config.get("macd_signal_threshold", 0)
        
        if macd <= macd_signal:
            return False
        
        # RSI filters
        rsi = features.get("rsi_14", 50)
        rsi_oversold = self.entry_config.get("rsi_oversold", 30)
        rsi_overbought = self.entry_config.get("rsi_overbought", 70)
        
        # ADX for trend strength
        adx = features.get("adx_14", 0)
        min_adx = self.entry_config.get("adx_min", 25)
        
        if adx < min_adx:
            logger.debug(f"ADX {adx:.2f} below minimum {min_adx}")
            return False
        
        return True
    
    def _determine_direction(
        self,
        features: Dict[str, Any],
        ml_prediction: Dict[str, Any]
    ) -> Optional[str]:
        """
        Determine trade direction (CE for calls, PE for puts).
        
        Returns:
            "CE" for calls, "PE" for puts, None if no clear direction
        """
        # Use MACD and RSI for direction
        macd_hist = features.get("macd_hist", 0)
        rsi = features.get("rsi_14", 50)
        expected_return = ml_prediction.get("expected_return", 0)
        
        # Bullish signals -> CALL
        if macd_hist > 0 and rsi > 50 and expected_return > 0:
            return "CE"
        
        # Bearish signals -> PUT
        elif macd_hist < 0 and rsi < 50 and expected_return > 0:
            return "PE"
        
        return None
    
    def _select_strike(
        self,
        option_chain: pd.DataFrame,
        option_type: str,
        spot_price: float
    ) -> Optional[Dict[str, Any]]:
        """
        Select best strike based on delta, OI, volume, and spread.
        
        Args:
            option_chain: DataFrame with option chain data
            option_type: "CE" or "PE"
            spot_price: Current spot price
            
        Returns:
            Dictionary with selected strike details or None
        """
        try:
            # Filter by option type
            options = option_chain[option_chain["instrument_type"] == option_type].copy()
            
            if options.empty:
                return None
            
            # Filter by delta range (if available)
            if "delta" in options.columns:
                options = options[
                    (options["delta"].abs() >= self.min_delta) &
                    (options["delta"].abs() <= self.max_delta)
                ]
            
            # Filter by OI and volume
            if "oi" in options.columns:
                options = options[options["oi"] >= self.min_oi]
            
            if "volume" in options.columns:
                options = options[options["volume"] >= self.min_volume]
            
            # Filter by bid-ask spread
            if "bid" in options.columns and "ask" in options.columns:
                options["spread_pct"] = (
                    (options["ask"] - options["bid"]) / options["last_price"] * 100
                )
                options = options[options["spread_pct"] <= self.max_spread_pct]
            
            if options.empty:
                logger.debug("No strikes passed filters")
                return None
            
            # Sort by proximity to desired delta and OI
            if "delta" in options.columns:
                target_delta = (self.min_delta + self.max_delta) / 2
                options["delta_score"] = abs(options["delta"].abs() - target_delta)
                options = options.sort_values(["delta_score", "oi"], ascending=[True, False])
            else:
                # If no delta, sort by OI
                options = options.sort_values("oi", ascending=False)
            
            # Select best strike
            best_strike = options.iloc[0].to_dict()
            
            logger.info(
                f"Selected {option_type} strike {best_strike.get('strike')} "
                f"@ {best_strike.get('last_price')} "
                f"(OI: {best_strike.get('oi')}, Vol: {best_strike.get('volume')})"
            )
            
            return best_strike
            
        except Exception as e:
            logger.error(f"Strike selection failed: {e}")
            return None
    
    def calculate_position_size(
        self,
        signal: Signal,
        available_capital: float,
        current_positions: int
    ) -> int:
        """
        Calculate position size using risk-based approach.
        
        Args:
            signal: Trading signal
            available_capital: Available capital
            current_positions: Number of current positions
            
        Returns:
            Number of lots
        """
        try:
            # Maximum risk per trade
            max_risk = available_capital * (self.risk_per_trade_pct / 100)
            
            # Risk per lot (premium paid)
            risk_per_lot = signal.entry_price * signal.quantity
            
            # Maximum lots based on risk
            max_lots = int(max_risk / risk_per_lot)
            
            # Apply Kelly fraction (conservative sizing)
            if signal.confidence > 0 and signal.expected_return > 0:
                kelly_lots = int(
                    (signal.confidence - 0.5) * self.kelly_fraction *
                    available_capital / risk_per_lot
                )
                max_lots = min(max_lots, kelly_lots)
            
            # Ensure at least 1 lot
            lots = max(1, max_lots)
            
            logger.info(
                f"Position size: {lots} lots (risk: {lots * risk_per_lot:.2f} INR)"
            )
            
            return lots * signal.quantity  # Convert lots to quantity
            
        except Exception as e:
            logger.error(f"Position sizing failed: {e}")
            return signal.quantity  # Return single lot as fallback
    
    def should_exit(
        self,
        position: Dict[str, Any],
        current_price: float,
        market_data: Dict[str, Any]
    ) -> bool:
        """
        Check if F&O position should be exited.
        
        Exits on:
        - Take profit hit (50% premium gain)
        - Stop loss hit (60% premium loss)
        - N days before expiry
        - End of day for intraday
        """
        try:
            entry_price = position["entry_price"]
            entry_time = position.get("entry_time", datetime.now())
            
            # Calculate return
            return_pct = (current_price - entry_price) / entry_price * 100
            
            # Take profit
            if return_pct >= self.take_profit_pct:
                logger.info(
                    f"Take profit hit: {return_pct:.2f}% >= {self.take_profit_pct}%"
                )
                return True
            
            # Stop loss
            if return_pct <= -self.stop_loss_pct:
                logger.info(
                    f"Stop loss hit: {return_pct:.2f}% <= -{self.stop_loss_pct}%"
                )
                return True
            
            # Time-based exit (expiry proximity)
            expiry = position.get("expiry")
            if expiry:
                expiry_date = pd.to_datetime(expiry)
                days_to_expiry = (expiry_date - datetime.now()).days
                
                if days_to_expiry <= self.exit_days_before_expiry:
                    logger.info(
                        f"Expiry proximity exit: {days_to_expiry} days to expiry"
                    )
                    return True
            
            # Intraday exit (3:15 PM)
            exit_time = self.exit_config.get("intraday_exit_time", "15:15")
            current_time = datetime.now().time()
            exit_hour, exit_min = map(int, exit_time.split(":"))
            
            if current_time.hour >= exit_hour and current_time.minute >= exit_min:
                # Check if it's an intraday position (entered today)
                if entry_time.date() == datetime.now().date():
                    logger.info("Intraday exit time reached")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Exit check failed: {e}")
            return False
