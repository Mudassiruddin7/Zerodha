"""XGBoost-based signal generation models for trade prediction."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from loguru import logger

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
import joblib


class SignalModel:
    """
    ML-based signal generation using XGBoost.
    
    Implements two models:
    1. Classifier: Predicts probability of winning trade (p_win)
    2. Regressor: Predicts expected return percentage
    
    Both models must agree for a signal to be generated.
    """
    
    def __init__(self, config):
        """
        Initialize signal model.
        
        Args:
            config: Configuration object
        """
        self.config = config
        
        # Model configuration
        self.classifier_config = config.get("ml_models.classifier")
        self.regressor_config = config.get("ml_models.regressor")
        self.training_config = config.get("ml_models.training")
        
        # Models
        self.classifier: Optional[xgb.XGBClassifier] = None
        self.regressor: Optional[xgb.XGBRegressor] = None
        
        # Feature importance tracking
        self.feature_names: Optional[list] = None
        self.classifier_importance: Optional[Dict[str, float]] = None
        self.regressor_importance: Optional[Dict[str, float]] = None
        
        # Model metadata
        self.classifier_metrics: Dict[str, float] = {}
        self.regressor_metrics: Dict[str, float] = {}
        
        # Model save paths
        self.model_dir = Path("models/saved")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Signal model initialized")
    
    def train_classifier(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Train the win probability classifier.
        
        Args:
            X: Feature matrix
            y: Binary labels (1 = winning trade, 0 = losing trade)
            validation_split: Validation split ratio (default from config)
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            logger.info("Training classifier model...")
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Split data
            test_size = validation_split or self.training_config.get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
            
            # Initialize classifier
            self.classifier = xgb.XGBClassifier(
                objective=self.classifier_config.get("objective", "binary:logistic"),
                max_depth=self.classifier_config.get("max_depth", 6),
                learning_rate=self.classifier_config.get("learning_rate", 0.1),
                n_estimators=self.classifier_config.get("n_estimators", 100),
                min_child_weight=self.classifier_config.get("min_child_weight", 3),
                subsample=self.classifier_config.get("subsample", 0.8),
                colsample_bytree=self.classifier_config.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric='logloss'
            )
            
            # Train with early stopping
            early_stopping = self.classifier_config.get("early_stopping_rounds", 10)
            
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred = self.classifier.predict(X_test)
            y_proba = self.classifier.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            self.classifier_metrics = metrics
            
            # Feature importance
            importance_dict = dict(zip(
                self.feature_names,
                self.classifier.feature_importances_
            ))
            self.classifier_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Log results
            logger.success(f"Classifier trained successfully!")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            
            # Check if meets target
            target_accuracy = self.classifier_config.get("target_accuracy", 0.70)
            if metrics['accuracy'] < target_accuracy:
                logger.warning(
                    f"Classifier accuracy {metrics['accuracy']:.4f} below target {target_accuracy}"
                )
            
            # Top features
            logger.info("Top 10 important features:")
            for i, (feat, imp) in enumerate(list(self.classifier_importance.items())[:10], 1):
                logger.info(f"  {i}. {feat}: {imp:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            raise
    
    def train_regressor(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Train the expected return regressor.
        
        Args:
            X: Feature matrix
            y: Continuous labels (percentage return)
            validation_split: Validation split ratio
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            logger.info("Training regressor model...")
            
            # Split data
            test_size = validation_split or self.training_config.get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Initialize regressor
            self.regressor = xgb.XGBRegressor(
                objective=self.regressor_config.get("objective", "reg:squarederror"),
                max_depth=self.regressor_config.get("max_depth", 6),
                learning_rate=self.regressor_config.get("learning_rate", 0.1),
                n_estimators=self.regressor_config.get("n_estimators", 100),
                subsample=self.regressor_config.get("subsample", 0.8),
                colsample_bytree=self.regressor_config.get("colsample_bytree", 0.8),
                random_state=42,
                eval_metric='rmse'
            )
            
            # Train with early stopping
            self.regressor.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predictions
            y_pred = self.regressor.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            metrics = {
                "r2_score": r2,
                "rmse": rmse,
                "mae": mae,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            }
            
            self.regressor_metrics = metrics
            
            # Feature importance
            importance_dict = dict(zip(
                self.feature_names,
                self.regressor.feature_importances_
            ))
            self.regressor_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
            
            # Log results
            logger.success(f"Regressor trained successfully!")
            logger.info(f"R² Score: {metrics['r2_score']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"MAE: {metrics['mae']:.4f}")
            
            # Check if meets target
            target_r2 = self.regressor_config.get("target_r2", 0.50)
            if metrics['r2_score'] < target_r2:
                logger.warning(
                    f"Regressor R² {metrics['r2_score']:.4f} below target {target_r2}"
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to train regressor: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate trading signal prediction.
        
        Args:
            X: Feature matrix (single row or multiple rows)
            
        Returns:
            Dictionary with predictions:
            - p_win: Probability of winning trade
            - expected_return: Expected return percentage
            - confidence: Overall confidence score
            - should_trade: Boolean recommendation
        """
        if self.classifier is None or self.regressor is None:
            raise ValueError("Models not trained. Call train_classifier() and train_regressor() first.")
        
        try:
            # Ensure features match training
            if list(X.columns) != self.feature_names:
                logger.warning("Feature mismatch, reordering...")
                X = X[self.feature_names]
            
            # Classifier prediction
            p_win = self.classifier.predict_proba(X)[:, 1]
            
            # Regressor prediction
            expected_return = self.regressor.predict(X)
            
            # Generate signals for each row
            results = []
            min_confidence = self.config.get("fo_strategy.entry_rules.min_model_confidence", 0.80)
            
            for i in range(len(X)):
                signal = {
                    "p_win": float(p_win[i]),
                    "expected_return": float(expected_return[i]),
                    "confidence": float(p_win[i]),  # Use p_win as confidence
                    "should_trade": p_win[i] >= min_confidence and expected_return[i] > 0,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(signal)
            
            return results if len(results) > 1 else results[0]
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def save_models(self, suffix: str = ""):
        """
        Save trained models to disk.
        
        Args:
            suffix: Optional suffix for model filenames
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.classifier is not None:
                classifier_path = self.model_dir / f"classifier_{timestamp}{suffix}.joblib"
                joblib.dump(self.classifier, classifier_path)
                logger.info(f"Classifier saved to {classifier_path}")
                
                # Save metadata
                metadata_path = self.model_dir / f"classifier_{timestamp}{suffix}_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "metrics": self.classifier_metrics,
                        "feature_importance": self.classifier_importance,
                        "feature_names": self.feature_names,
                        "config": self.classifier_config
                    }, f, indent=2)
            
            if self.regressor is not None:
                regressor_path = self.model_dir / f"regressor_{timestamp}{suffix}.joblib"
                joblib.dump(self.regressor, regressor_path)
                logger.info(f"Regressor saved to {regressor_path}")
                
                # Save metadata
                metadata_path = self.model_dir / f"regressor_{timestamp}{suffix}_metadata.json"
                import json
                with open(metadata_path, 'w') as f:
                    json.dump({
                        "metrics": self.regressor_metrics,
                        "feature_importance": self.regressor_importance,
                        "feature_names": self.feature_names,
                        "config": self.regressor_config
                    }, f, indent=2)
            
            logger.success("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self, classifier_path: str, regressor_path: str):
        """
        Load trained models from disk.
        
        Args:
            classifier_path: Path to classifier model
            regressor_path: Path to regressor model
        """
        try:
            self.classifier = joblib.load(classifier_path)
            self.regressor = joblib.load(regressor_path)
            
            logger.success(f"Models loaded from {classifier_path} and {regressor_path}")
            
            # Try to load metadata
            try:
                import json
                metadata_path = Path(classifier_path).with_name(
                    Path(classifier_path).stem + "_metadata.json"
                )
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.feature_names = metadata.get("feature_names")
                        self.classifier_importance = metadata.get("feature_importance")
                        self.classifier_metrics = metadata.get("metrics")
            except Exception as e:
                logger.warning(f"Could not load classifier metadata: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from both models.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.classifier_importance is None:
            logger.warning("Classifier not trained yet")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "classifier_importance": [
                self.classifier_importance.get(f, 0) for f in self.feature_names
            ],
            "regressor_importance": [
                self.regressor_importance.get(f, 0) if self.regressor_importance else 0
                for f in self.feature_names
            ]
        })
        
        df["avg_importance"] = (df["classifier_importance"] + df["regressor_importance"]) / 2
        df = df.sort_values("avg_importance", ascending=False).head(top_n)
        
        return df
