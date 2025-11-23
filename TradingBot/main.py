#!/usr/bin/env python3
"""
Zerodha Kite MCP AI Trading Bot - Main Entry Point

Semi-automated trading bot with ML-driven signals, strict risk management,
and manual confirmation for all trades.

Author: Trading Bot Team
Date: November 7, 2025
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import ConfigLoader
from execution.kite_client import KiteMCPClient
from data.loader import DataLoader
from risk.risk_manager import RiskManager
from monitoring.logger import AuditLogger
from orchestration.trading_orchestrator import TradingOrchestrator


class TradingBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trading bot with configuration."""
        self.config = ConfigLoader(config_path)
        self.logger = AuditLogger()
        
        # Initialize components
        self.kite_client = None
        self.data_loader = None
        self.risk_manager = None
        self.orchestrator = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging system."""
        log_config = self.config.get("monitoring.logging")
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sys.stderr,
            level=log_config.get("level", "INFO"),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        )
        
        # File logging
        if log_config.get("file"):
            logger.add(
                log_config["file"],
                rotation=log_config.get("rotation", "10 MB"),
                retention=log_config.get("retention", "30 days"),
                level=log_config.get("level", "INFO"),
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
            )
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Zerodha Kite MCP AI Trading Bot...")
        
        # Load environment variables
        load_dotenv()
        
        # Initialize Kite client
        logger.info("Initializing Kite MCP client...")
        self.kite_client = KiteMCPClient(self.config)
        await self.kite_client.initialize()
        
        # Initialize data loader
        logger.info("Initializing data loader...")
        self.data_loader = DataLoader(self.config, self.kite_client)
        
        # Initialize risk manager
        logger.info("Initializing risk manager...")
        self.risk_manager = RiskManager(self.config)
        
        # Initialize orchestrator
        logger.info("Initializing trading orchestrator...")
        self.orchestrator = TradingOrchestrator(
            config=self.config,
            kite_client=self.kite_client,
            data_loader=self.data_loader,
            risk_manager=self.risk_manager,
            logger=self.logger
        )
        
        logger.success("All components initialized successfully!")
        
    async def run(self):
        """Run the trading bot."""
        try:
            await self.initialize()
            
            # Check if live trading is enabled
            if self.config.get("trading_rules.enable_live_trading"):
                logger.warning("⚠️  LIVE TRADING MODE ENABLED ⚠️")
                logger.warning("Real money will be at risk!")
                
                # Require manual confirmation
                if self.config.get("trading_rules.require_manual_confirmation"):
                    logger.info("Manual confirmation required for all trades")
            else:
                logger.info("📊 Running in PAPER TRADING mode")
            
            # Start the orchestrator
            await self.orchestrator.start()
            
        except KeyboardInterrupt:
            logger.warning("Received keyboard interrupt, shutting down gracefully...")
            await self.shutdown()
        except Exception as e:
            logger.exception(f"Fatal error in trading bot: {e}")
            await self.shutdown()
            sys.exit(1)
    
    async def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("Shutting down trading bot...")
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        if self.kite_client:
            await self.kite_client.close()
        
        logger.info("Shutdown complete")


def main():
    """Main entry point."""
    # ASCII Art Banner
    print(r"""
    ╔══════════════════════════════════════════════════════════╗
    ║     Zerodha Kite MCP AI Trading Bot v1.0.0             ║
    ║     Semi-Automated Trading with ML-Driven Signals       ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Zerodha Kite MCP AI Trading Bot")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run backtest mode instead of live trading"
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode"
    )
    args = parser.parse_args()
    
    # Create and run bot
    bot = TradingBot(config_path=args.config)
    
    # Override config for command line flags
    if args.paper:
        bot.config.set("trading_rules.enable_live_trading", False)
        logger.info("Paper trading mode enabled via command line")
    
    # Run the bot
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
