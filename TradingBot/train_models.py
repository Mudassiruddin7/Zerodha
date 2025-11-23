"""
Sample script to train ML models on historical data.

Usage:
    python train_models.py --instrument NIFTY --days 180
"""

import asyncio
import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import ConfigLoader
from execution.kite_client import KiteMCPClient
from data.loader import DataLoader
from models.training_pipeline import ModelTrainer
from loguru import logger


async def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train ML models for trading bot")
    parser.add_argument(
        "--instrument",
        default="NIFTY",
        help="Instrument symbol to train on"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Historical data lookback days"
    )
    parser.add_argument(
        "--type",
        choices=["equity", "fo"],
        default="equity",
        help="Instrument type"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("ML MODEL TRAINING SCRIPT")
    logger.info("=" * 70)
    logger.info(f"Instrument: {args.instrument}")
    logger.info(f"Lookback: {args.days} days")
    logger.info(f"Type: {args.type}")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config = ConfigLoader("config/config.yaml")
        
        # Initialize Kite client
        kite_client = KiteMCPClient(config)
        await kite_client.initialize()
        
        # Search for instrument
        instruments = await kite_client.search_instruments(
            args.instrument,
            exchange="NSE" if args.type == "equity" else "NFO"
        )
        
        if not instruments:
            logger.error(f"Instrument {args.instrument} not found")
            return
        
        instrument = instruments[0]
        instrument_token = instrument["instrument_token"]
        
        logger.info(f"Found instrument: {instrument['tradingsymbol']} (token: {instrument_token})")
        
        # Initialize data loader and trainer
        data_loader = DataLoader(config, kite_client)
        trainer = ModelTrainer(config, data_loader)
        
        # Train models
        logger.info("\nStarting model training pipeline...")
        results = await trainer.train_models(
            instrument_token=instrument_token,
            lookback_days=args.days,
            instrument_type=args.type,
            save_models=True
        )
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Training samples: {results['training_samples']}")
        logger.info(f"Features: {results['features_count']}")
        logger.info("\nClassifier Metrics:")
        for k, v in results['classifier_metrics'].items():
            logger.info(f"  {k}: {v}")
        logger.info("\nRegressor Metrics:")
        for k, v in results['regressor_metrics'].items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 70)
        
        logger.success("✅ Training completed successfully!")
        
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
