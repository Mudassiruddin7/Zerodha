"""Configuration loader with validation and type checking."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator
from loguru import logger


class CapitalConfig(BaseModel):
    """Capital and account settings."""
    starting_capital: float = Field(gt=0)
    min_reserve: float = Field(ge=0)
    max_daily_loss_pct: float = Field(gt=0, le=100)
    max_weekly_loss_pct: float = Field(gt=0, le=100)
    margin_buffer_pct: float = Field(gt=0, le=100)


class TradingRulesConfig(BaseModel):
    """Trading rules configuration."""
    max_concurrent_positions: int = Field(gt=0)
    enable_live_trading: bool = False
    require_manual_confirmation: bool = True
    cool_down_hours_fo: int = Field(ge=0)
    cool_down_hours_equity: int = Field(ge=0)


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker settings."""
    max_daily_losses: int = Field(gt=0)
    max_consecutive_losses: int = Field(gt=0)
    high_latency_ms: int = Field(gt=0)
    min_margin_buffer_pct: float = Field(gt=0, le=100)
    exception_count_threshold: int = Field(gt=0)
    recovery_wait_minutes: int = Field(gt=0)


class RiskManagementConfig(BaseModel):
    """Risk management configuration."""
    circuit_breaker: CircuitBreakerConfig
    position_limits: Dict[str, Any]


class FOStrategyConfig(BaseModel):
    """F&O strategy configuration."""
    enabled: bool = True
    instruments: list[str]
    strike_selection: Dict[str, Any]
    entry_rules: Dict[str, Any]
    exit_rules: Dict[str, Any]
    position_sizing: Dict[str, Any]


class EquityStrategyConfig(BaseModel):
    """Equity strategy configuration."""
    enabled: bool = True
    swing_trading: Dict[str, Any]
    intraday_trading: Dict[str, Any]


class MLModelsConfig(BaseModel):
    """ML models configuration."""
    classifier: Dict[str, Any]
    regressor: Dict[str, Any]
    training: Dict[str, Any]


class ExecutionConfig(BaseModel):
    """Execution settings."""
    order_params: Dict[str, Any]
    slippage: Dict[str, Any]
    fees: Dict[str, Any]
    retry: Dict[str, Any]


class ConfigLoader:
    """Load and validate configuration from YAML file."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration using Pydantic models."""
        try:
            # Validate each section
            CapitalConfig(**self._config.get("capital", {}))
            TradingRulesConfig(**self._config.get("trading_rules", {}))
            RiskManagementConfig(**self._config.get("risk_management", {}))
            FOStrategyConfig(**self._config.get("fo_strategy", {}))
            EquityStrategyConfig(**self._config.get("equity_strategy", {}))
            MLModelsConfig(**self._config.get("ml_models", {}))
            ExecutionConfig(**self._config.get("execution", {}))
            
            logger.success("Configuration validation passed")
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "capital.starting_capital")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split(".")
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Configuration updated: {key} = {value}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary."""
        return self._config.copy()
    
    def reload(self):
        """Reload configuration from file."""
        logger.info("Reloading configuration...")
        self._load_config()
