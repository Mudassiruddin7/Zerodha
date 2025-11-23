"""Performance metrics calculation for backtesting."""

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import logging

from backtest.engine import BacktestResult, Trade

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calculate and analyze performance metrics for backtests."""
    
    @staticmethod
    def calculate_metrics(result: BacktestResult) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary of all metrics
        """
        return {
            # Trade statistics
            "total_trades": result.total_trades,
            "winning_trades": result.winning_trades,
            "losing_trades": result.losing_trades,
            "win_rate": result.win_rate,
            
            # P&L
            "total_pnl": result.total_pnl,
            "total_fees": result.total_fees,
            "net_pnl": result.net_pnl,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "largest_win": result.largest_win,
            "largest_loss": result.largest_loss,
            "profit_factor": result.profit_factor,
            
            # Risk metrics
            "max_drawdown": result.max_drawdown,
            "max_drawdown_duration_days": result.max_drawdown_duration,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "calmar_ratio": result.calmar_ratio,
            
            # Other
            "avg_trade_duration_hours": result.avg_trade_duration.total_seconds() / 3600,
            "max_consecutive_wins": result.max_consecutive_wins,
            "max_consecutive_losses": result.max_consecutive_losses,
            "duration_days": result.duration_days,
        }
    
    @staticmethod
    def print_summary(result: BacktestResult, verbose: bool = True):
        """
        Print formatted backtest summary.
        
        Args:
            result: BacktestResult object
            verbose: If True, print detailed metrics
        """
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        
        if result.start_date and result.end_date:
            print(f"Period: {result.start_date.date()} to {result.end_date.date()} ({result.duration_days} days)")
        
        print(f"\nTrade Statistics:")
        print(f"  Total Trades:     {result.total_trades}")
        print(f"  Winning Trades:   {result.winning_trades} ({result.win_rate:.2%})")
        print(f"  Losing Trades:    {result.losing_trades}")
        
        print(f"\nP&L Analysis:")
        print(f"  Total P&L:        ₹{result.total_pnl:,.2f}")
        print(f"  Total Fees:       ₹{result.total_fees:,.2f}")
        print(f"  Net P&L:          ₹{result.net_pnl:,.2f}")
        print(f"  Avg Win:          ₹{result.avg_win:,.2f}")
        print(f"  Avg Loss:         ₹{result.avg_loss:,.2f}")
        print(f"  Largest Win:      ₹{result.largest_win:,.2f}")
        print(f"  Largest Loss:     ₹{result.largest_loss:,.2f}")
        print(f"  Profit Factor:    {result.profit_factor:.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:     {result.max_drawdown:.2%}")
        print(f"  DD Duration:      {result.max_drawdown_duration} days")
        print(f"  Sharpe Ratio:     {result.sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:    {result.sortino_ratio:.2f}")
        print(f"  Calmar Ratio:     {result.calmar_ratio:.2f}")
        
        print(f"\nOther Metrics:")
        print(f"  Avg Trade Duration: {result.avg_trade_duration}")
        print(f"  Max Consecutive Wins:   {result.max_consecutive_wins}")
        print(f"  Max Consecutive Losses: {result.max_consecutive_losses}")
        
        # Success gates
        print(f"\n{'='*80}")
        print("SUCCESS GATES")
        print("="*80)
        
        gates = [
            ("Win Rate >= 50%", result.win_rate >= 0.50, f"{result.win_rate:.2%}"),
            ("Sharpe Ratio >= 0.80", result.sharpe_ratio >= 0.80, f"{result.sharpe_ratio:.2f}"),
            ("Max Drawdown <= 15%", result.max_drawdown <= 0.15, f"{result.max_drawdown:.2%}"),
            ("Net P&L > ₹200", result.net_pnl > 200, f"₹{result.net_pnl:,.2f}"),
            ("Min 30 Trades", result.total_trades >= 30, f"{result.total_trades}"),
        ]
        
        all_passed = True
        for gate_name, passed, value in gates:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}  {gate_name:30} {value}")
            if not passed:
                all_passed = False
        
        print("\n" + "="*80)
        if all_passed:
            print("✓✓✓ ALL GATES PASSED - Strategy ready for paper trading ✓✓✓")
        else:
            print("✗✗✗ SOME GATES FAILED - Strategy needs improvement ✗✗✗")
        print("="*80 + "\n")
        
        # Detailed trade log if verbose
        if verbose and result.trades:
            print("\nTRADE LOG (First 10 and Last 10):")
            print("-" * 120)
            print(f"{'Entry Time':<20} {'Exit Time':<20} {'Direction':<8} {'Qty':<6} {'Entry':<10} {'Exit':<10} {'P&L':<12} {'Reason':<15}")
            print("-" * 120)
            
            trades_to_show = result.trades[:10] + result.trades[-10:] if len(result.trades) > 20 else result.trades
            
            for trade in trades_to_show:
                if trade.is_open():
                    continue
                    
                print(
                    f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<20} "
                    f"{trade.exit_time.strftime('%Y-%m-%d %H:%M') if trade.exit_time else 'OPEN':<20} "
                    f"{trade.direction:<8} "
                    f"{trade.quantity:<6} "
                    f"₹{trade.entry_price:<9.2f} "
                    f"₹{trade.exit_price if trade.exit_price else 0:<9.2f} "
                    f"₹{trade.pnl:<11.2f} "
                    f"{trade.exit_reason:<15}"
                )
            print("-" * 120)
    
    @staticmethod
    def plot_equity_curve(result: BacktestResult, save_path: str = None):
        """
        Plot equity curve and drawdown.
        
        Args:
            result: BacktestResult object
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Equity curve
            ax1.plot(result.equity_curve.index, result.equity_curve.values, label='Equity', linewidth=2)
            ax1.axhline(y=result.equity_curve.iloc[0], color='gray', linestyle='--', label='Initial Capital')
            ax1.set_ylabel('Equity (INR)')
            ax1.set_title('Equity Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            cummax = result.equity_curve.cummax()
            drawdown = (result.equity_curve - cummax) / cummax * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
            ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_xlabel('Date')
            ax2.set_title('Drawdown')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"Equity curve saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
    
    @staticmethod
    def analyze_trade_distribution(trades: List[Trade]) -> Dict:
        """
        Analyze distribution of trade characteristics.
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Dictionary with distribution statistics
        """
        closed_trades = [t for t in trades if not t.is_open()]
        
        if not closed_trades:
            return {}
        
        pnls = [t.pnl for t in closed_trades]
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in closed_trades if t.exit_time]
        
        return {
            "pnl_distribution": {
                "mean": np.mean(pnls),
                "median": np.median(pnls),
                "std": np.std(pnls),
                "min": np.min(pnls),
                "max": np.max(pnls),
                "p25": np.percentile(pnls, 25),
                "p75": np.percentile(pnls, 75),
            },
            "duration_distribution_hours": {
                "mean": np.mean(durations) if durations else 0,
                "median": np.median(durations) if durations else 0,
                "std": np.std(durations) if durations else 0,
                "min": np.min(durations) if durations else 0,
                "max": np.max(durations) if durations else 0,
            },
        }
    
    @staticmethod
    def compare_results(results: List[BacktestResult], labels: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple backtest results.
        
        Args:
            results: List of BacktestResult objects
            labels: Optional labels for each result
            
        Returns:
            DataFrame with comparison
        """
        if labels is None:
            labels = [f"Result {i+1}" for i in range(len(results))]
        
        metrics_list = []
        for result in results:
            metrics_list.append(PerformanceMetrics.calculate_metrics(result))
        
        df = pd.DataFrame(metrics_list, index=labels)
        
        return df
