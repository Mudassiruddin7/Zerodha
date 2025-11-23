"""Model training pipeline for preparing data and training models."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from loguru import logger

from data.loader import DataLoader
from data.feature_computer import FeatureComputer
from models.signal_model import SignalModel


class ModelTrainer:
    """
    End-to-end pipeline for training ML models.
    
    Handles:
    - Data loading and preprocessing
    - Feature engineering
    - Label generation
    - Model training
    - Validation and performance tracking
    """
    
    def __init__(self, config, data_loader: DataLoader):
        """
        Initialize model trainer.
        
        Args:
            config: Configuration object
            data_loader: DataLoader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.feature_computer = FeatureComputer(config)
        self.signal_model = SignalModel(config)
        
        logger.info("Model trainer initialized")
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        forward_window: int = 5,
        profit_threshold: float = 2.0,
        loss_threshold: float = -1.5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate training labels from historical data.
        
        Args:
            df: DataFrame with OHLCV data
            forward_window: Number of periods to look forward
            profit_threshold: Percentage gain to consider "winning" trade
            loss_threshold: Percentage loss threshold
            
        Returns:
            Tuple of (binary_labels, return_labels)
        """
        try:
            logger.info(f"Generating labels with forward window={forward_window}")
            
            close = df['Close']
            
            # Calculate forward returns
            forward_returns = []
            for i in range(forward_window):
                ret = close.shift(-i-1) / close - 1
                forward_returns.append(ret)
            
            # Maximum return in forward window
            max_forward_return = pd.concat(forward_returns, axis=1).max(axis=1) * 100
            
            # Minimum return in forward window (for stop-loss simulation)
            min_forward_return = pd.concat(forward_returns, axis=1).min(axis=1) * 100
            
            # Binary labels: 1 if profitable trade, 0 otherwise
            binary_labels = (
                (max_forward_return >= profit_threshold) &
                (min_forward_return > loss_threshold)
            ).astype(int)
            
            # Continuous labels: actual return achieved
            return_labels = max_forward_return.copy()
            
            # Remove last few rows (no forward data)
            binary_labels = binary_labels[:-forward_window]
            return_labels = return_labels[:-forward_window]
            
            logger.info(f"Generated {len(binary_labels)} labels")
            logger.info(f"Win rate: {binary_labels.mean():.2%}")
            logger.info(f"Avg winning return: {return_labels[binary_labels == 1].mean():.2f}%")
            logger.info(f"Avg losing return: {return_labels[binary_labels == 0].mean():.2f}%")
            
            return binary_labels, return_labels
            
        except Exception as e:
            logger.error(f"Failed to generate labels: {e}")
            raise
    
    async def prepare_training_data(
        self,
        instrument_token: int,
        lookback_days: int = 180,
        instrument_type: str = "equity"
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare complete training dataset.
        
        Args:
            instrument_token: Instrument to train on
            lookback_days: Historical data lookback
            instrument_type: "equity" or "fo"
            
        Returns:
            Tuple of (features_df, binary_labels, return_labels)
        """
        try:
            logger.info(f"Preparing training data for token {instrument_token}")
            
            # Load historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=lookback_days)
            
            df = await self.data_loader.load_historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date
            )
            
            if df.empty:
                raise ValueError("No historical data available")
            
            logger.info(f"Loaded {len(df)} historical candles")
            
            # Compute features
            df = self.feature_computer.compute_all_features(df, instrument_type)
            
            # Generate labels
            binary_labels, return_labels = self.generate_labels(df)
            
            # Get feature columns (exclude OHLCV)
            base_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            feature_cols = [c for c in df.columns if c not in base_cols]
            
            # Align features with labels
            X = df[feature_cols].iloc[:-5]  # Remove last 5 rows (forward window)
            
            # Handle missing values
            X = X.fillna(method='ffill').fillna(method='bfill')
            
            # Remove any remaining NaN rows
            valid_idx = ~(X.isna().any(axis=1) | binary_labels.isna() | return_labels.isna())
            X = X[valid_idx]
            binary_labels = binary_labels[valid_idx]
            return_labels = return_labels[valid_idx]
            
            logger.success(f"Prepared {len(X)} training samples with {len(feature_cols)} features")
            
            return X, binary_labels, return_labels
            
        except Exception as e:
            logger.error(f"Failed to prepare training data: {e}")
            raise
    
    async def train_models(
        self,
        instrument_token: int,
        lookback_days: int = 180,
        instrument_type: str = "equity",
        save_models: bool = True
    ) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            instrument_token: Instrument to train on
            lookback_days: Historical data lookback
            instrument_type: "equity" or "fo"
            save_models: Whether to save trained models
            
        Returns:
            Dictionary with training results
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("=" * 60)
            
            # Prepare data
            X, y_binary, y_return = await self.prepare_training_data(
                instrument_token, lookback_days, instrument_type
            )
            
            # Check minimum samples
            min_samples = self.config.get("ml_models.training.min_training_samples", 1000)
            if len(X) < min_samples:
                raise ValueError(f"Insufficient training samples: {len(X)} < {min_samples}")
            
            # Train classifier
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING CLASSIFIER")
            logger.info("=" * 60)
            classifier_metrics = self.signal_model.train_classifier(X, y_binary)
            
            # Train regressor
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING REGRESSOR")
            logger.info("=" * 60)
            regressor_metrics = self.signal_model.train_regressor(X, y_return)
            
            # Feature importance
            logger.info("\n" + "=" * 60)
            logger.info("FEATURE IMPORTANCE ANALYSIS")
            logger.info("=" * 60)
            importance_df = self.signal_model.get_feature_importance(top_n=20)
            logger.info("\nTop 20 Features:")
            logger.info(importance_df.to_string(index=False))
            
            # Save models
            if save_models:
                logger.info("\n" + "=" * 60)
                logger.info("SAVING MODELS")
                logger.info("=" * 60)
                suffix = f"_{instrument_type}_{instrument_token}"
                self.signal_model.save_models(suffix=suffix)
            
            results = {
                "classifier_metrics": classifier_metrics,
                "regressor_metrics": regressor_metrics,
                "feature_importance": importance_df.to_dict('records'),
                "training_samples": len(X),
                "features_count": len(X.columns),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING COMPLETE")
            logger.info("=" * 60)
            logger.success(f"Classifier Accuracy: {classifier_metrics['accuracy']:.2%}")
            logger.success(f"Regressor R²: {regressor_metrics['r2_score']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    async def retrain_models(
        self,
        instrument_tokens: list[int],
        instrument_type: str = "equity"
    ):
        """
        Retrain models for multiple instruments.
        
        Args:
            instrument_tokens: List of instruments to train on
            instrument_type: "equity" or "fo"
        """
        results = {}
        
        for token in instrument_tokens:
            try:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Training models for instrument {token}")
                logger.info(f"{'=' * 60}\n")
                
                result = await self.train_models(token, instrument_type=instrument_type)
                results[token] = result
                
            except Exception as e:
                logger.error(f"Failed to train models for {token}: {e}")
                results[token] = {"error": str(e)}
        
        return results
