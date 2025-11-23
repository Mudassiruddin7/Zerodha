"""Audit logging system for trade tracking."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from loguru import logger


class AuditLogger:
    """
    Structured audit logging for trade decisions and executions.
    
    Logs all signals, trade decisions, order placements, and fills
    in JSON format for later analysis.
    """
    
    def __init__(self):
        """Initialize audit logger."""
        self.log_dir = Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"audit_{date_str}.jsonl"
        
        logger.info(f"Audit logger initialized: {self.log_file}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log an event to the audit log.
        
        Args:
            event_type: Type of event (signal, order, fill, etc.)
            data: Event data dictionary
        """
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                **data
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def log_signal(self, signal: Dict[str, Any]):
        """Log a trading signal."""
        self.log_event("signal", signal)
    
    def log_order(self, order: Dict[str, Any]):
        """Log an order placement."""
        self.log_event("order", order)
    
    def log_fill(self, fill: Dict[str, Any]):
        """Log an order fill."""
        self.log_event("fill", fill)
    
    def log_trade_decision(self, decision: Dict[str, Any]):
        """Log a trade decision (approved/rejected)."""
        self.log_event("trade_decision", decision)
