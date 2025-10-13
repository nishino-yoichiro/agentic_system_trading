"""
Professional Report Generator
Data-driven plotting logic that adapts to any combination of symbols/strategies
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import itertools
from datetime import datetime
import logging
from pathlib import Path

from equity_curve_model import EquityCurve, RollingWindowManager, NormalizationPolicy

logger = logging.getLogger(__name__)

class ReportConfig:
    """Configuration for report generation"""
    
    def __init__(self, 
                 equity: bool = True,
                 rolling_sharpe: bool = True,
                 drawdown: bool = True,
                 segmented_performance: bool = True,
                 regime_performance: bool = True,
                 rolling_window_percentages: List[float] = None,
                 figsize: Tuple[int, int] = (15, 10),
                 style: str = 'seaborn-v0_8'):
        
        self.equity = equity
        self.rolling_sharpe = rolling_sharpe
        self.drawdown = drawdown
        self.segmented_performance = segmented_performance
        self.regime_performance = regime_performance
        self.rolling_window_percentages = rolling_window_percentages or [0.01, 0.03, 0.07]
        self.figsize = figsize
        self.style = style


class ReportGenerator:
    """
    Professional report generator with adaptive visualization logic
    """
    
    def __init__(self, curves: List[EquityCurve], config: ReportConfig = None):
        self.curves = curves
        self.config = config or ReportConfig()
        self.window_manager = RollingWindowManager()
        
        # Set matplotlib style
        plt.style.use(self.config.style)
        
        # Determine view type based on data
        self.view_type = self._determine_view_type()
        logger.info(f"Detected view type: {self.view_type}")
    
    def _determine_view_type(self) -> str:
        """Automatically determine the best visualization approach"""
        symbols = set(curve.symbol for curve in self.curves)
        strategies = set(curve.strategy for curve in self.curves)
        
        if len(symbols) == 1 and len(strategies) > 1:
            return "strategy_comparison"
        elif len(symbols) > 1 and len(strategies) == 1:
            return "symbol_comparison"
        elif len(symbols) > 1 and len(strategies) > 1:
            return "portfolio_overview"
        else:
            return "single_curve"
    
    def group_by(self, key: str) -> Dict[str, List[EquityCurve]]:
        """Group curves by specified attribute"""
        groups = {}
        for curve in self.curves:
            value = getattr(curve, key)
            if value not in groups:
                groups[value] = []
            groups[value].append(curve)
        return groups
    
    def generate_report(self, output_path: str = None) -> str:
        """Generate comprehensive report"""
        if not self.curves:
            logger.warning("No curves to plot")
            return ""
        
        # Normalize curves based on view type
        normalized_curves = self._normalize_curves()
        
        # Create figure with appropriate layout
        fig, axes = self._create_layout()
        
        # Plot each section
        plot_idx = 0
        
        if self.config.equity:
            ax = axes.flat[plot_idx] if hasattr(axes, 'flat') else axes
            self._plot_equity_curves(normalized_curves, ax)
            plot_idx += 1
        
        if self.config.rolling_sharpe:
            ax = axes.flat[plot_idx] if hasattr(axes, 'flat') else axes
            self._plot_rolling_sharpe(normalized_curves, ax)
            plot_idx += 1
        
        if self.config.drawdown:
            ax = axes.flat[plot_idx] if hasattr(axes, 'flat') else axes
            self._plot_drawdown(normalized_curves, ax)
            plot_idx += 1
        
        if self.config.segmented_performance:
            ax = axes.flat[plot_idx] if hasattr(axes, 'flat') else axes
            self._plot_segmented_performance(normalized_curves, ax)
            plot_idx += 1
        
        if self.config.regime_performance:
            ax = axes.flat[plot_idx] if hasattr(axes, 'flat') else axes
            self._plot_regime_performance(normalized_curves, ax)
            plot_idx += 1
        
        # Add summary table
        self._add_summary_table(fig, normalized_curves)
        
        # Save and return path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"backtests/results/professional_report_{timestamp}.png"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Professional report saved to: {output_path}")
        return output_path
    
    def _normalize_curves(self) -> List[EquityCurve]:
        """Apply appropriate normalization based on view type"""
        if self.view_type == "strategy_comparison":
            return NormalizationPolicy.normalize_multi_strategy(self.curves)
        elif self.view_type == "symbol_comparison":
            return NormalizationPolicy.normalize_multi_symbol(self.curves)
        else:
            return [curve.normalize_to_one() for curve in self.curves]
    
    def _create_layout(self) -> Tuple[plt.Figure, np.ndarray]:
        """Create appropriate subplot layout"""
        num_plots = sum([
            self.config.equity,
            self.config.rolling_sharpe,
            self.config.drawdown,
            self.config.segmented_performance,
            self.config.regime_performance
        ])
        
        if num_plots <= 2:
            rows, cols = 1, 2
        elif num_plots <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 3, 2
        
        fig, axes = plt.subplots(rows, cols, figsize=self.config.figsize)
        
        if num_plots == 1:
            axes = np.array([axes])
        elif axes.ndim == 1:
            axes = axes.reshape(-1, 1)
        
        return fig, axes
    
    def _plot_equity_curves(self, curves: List[EquityCurve], ax: plt.Axes):
        """Plot normalized equity curves"""
        ax.set_title(f"Equity Curves - {self.view_type.replace('_', ' ').title()}")
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(curves)))
        
        for i, curve in enumerate(curves):
            # Convert timestamps to datetime objects to avoid matplotlib warnings
            timestamps = pd.to_datetime(curve.timestamps)
            ax.plot(timestamps, curve.equity, 
                   label=f"{curve.symbol}_{curve.strategy}", 
                   color=colors[i], linewidth=2, alpha=0.8)
        
        ax.set_ylabel("Normalized Equity")
        ax.set_xlabel("Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    def _plot_rolling_sharpe(self, curves: List[EquityCurve], ax: plt.Axes):
        """Plot rolling Sharpe ratios with dynamic windows per strategy"""
        ax.set_title("Rolling Sharpe Ratios")
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(curves)))
        
        for i, curve in enumerate(curves):
            windows = self.window_manager.select_windows(curve, self.config.rolling_window_percentages)
            
            for j, window in enumerate(windows):
                rolling_sharpe = curve.get_rolling_sharpe(window)
                if len(rolling_sharpe) > 0:
                    alpha = 0.6 + 0.4 * (j / len(windows))  # Fade effect
                    # Rolling Sharpe starts from index (window-1) and has same length as returns
                    # But we need to align with timestamps which are one longer than returns
                    rolling_sharpe_length = len(rolling_sharpe)
                    timestamps_length = len(curve.timestamps)
                    
                    # Use the shorter of the two lengths to avoid dimension mismatch
                    min_length = min(rolling_sharpe_length, timestamps_length)
                    
                    # For rolling metrics, we typically start from window-1 index
                    start_idx = max(0, window - 1)
                    end_idx = start_idx + min_length
                    
                    # Ensure we don't go out of bounds
                    if end_idx > timestamps_length:
                        end_idx = timestamps_length
                        start_idx = end_idx - min_length
                    
                    timestamps_subset = curve.timestamps[start_idx:end_idx]
                    rolling_sharpe_subset = rolling_sharpe[:len(timestamps_subset)]
                    
                    # Convert timestamps to datetime objects to avoid matplotlib warnings
                    timestamps_dt = pd.to_datetime(timestamps_subset)
                    ax.plot(timestamps_dt, rolling_sharpe_subset,
                           label=f"{curve.symbol}_{curve.strategy} ({window})",
                           color=colors[i], alpha=alpha, linewidth=1.5)
        
        ax.set_ylabel("Rolling Sharpe Ratio")
        ax.set_xlabel("Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    def _plot_drawdown(self, curves: List[EquityCurve], ax: plt.Axes):
        """Plot drawdown curves"""
        ax.set_title("Drawdown Analysis")
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(curves)))
        
        for i, curve in enumerate(curves):
            peak = np.maximum.accumulate(curve.equity)
            drawdown = (curve.equity - peak) / peak
            drawdown_pct = drawdown * 100
            
            # Convert timestamps to datetime objects to avoid matplotlib warnings
            timestamps = pd.to_datetime(curve.timestamps)
            ax.fill_between(timestamps, drawdown_pct, 0, 
                           alpha=0.3, color=colors[i],
                           label=f"{curve.symbol}_{curve.strategy}")
            ax.plot(timestamps, drawdown_pct, 
                   color=colors[i], linewidth=1, alpha=0.8)
        
        ax.set_ylabel("Drawdown (%)")
        ax.set_xlabel("Time")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0] * 1.1))
    
    def _plot_segmented_performance(self, curves: List[EquityCurve], ax: plt.Axes):
        """Plot segmented performance heatmap"""
        ax.set_title("Segmented Performance Analysis")
        
        # Use the best performing curve for segmentation
        best_curve = max(curves, key=lambda c: c.total_return)
        
        # Try to get segmented performance data
        if hasattr(best_curve, '_segmented_performance') and best_curve._segmented_performance:
            # Find the best segment type to display
            segment_types = ['hourly', 'dayofweek', 'volatility']
            segment_data = None
            segment_type = None
            
            for seg_type in segment_types:
                if seg_type in best_curve._segmented_performance:
                    segment_data = best_curve._segmented_performance[seg_type]
                    segment_type = seg_type.replace('dayofweek', 'Day of Week').title()
                    break
            
            if segment_data:
                segments = list(segment_data.keys())
                sharpes = [segment_data[s].get('sharpe_ratio', 0) for s in segments]
                
                bars = ax.bar(range(len(segments)), sharpes,
                             color=['green' if s > 0 else 'red' for s in sharpes], 
                             alpha=0.7)
                
                ax.set_title(f'{best_curve.symbol} - Performance by {segment_type}')
                ax.set_xlabel(segment_type)
                ax.set_ylabel('Sharpe Ratio')
                ax.set_xticks(range(len(segments)))
                ax.set_xticklabels(segments, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels
                for bar, sharpe in zip(bars, sharpes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           height + (0.01 if height > 0 else -0.01),
                           f'{sharpe:.2f}', ha='center', 
                           va='bottom' if height > 0 else 'top', fontsize=8)
            else:
                ax.text(0.5, 0.5, 'No segmented data available', 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No segmented data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_regime_performance(self, curves: List[EquityCurve], ax: plt.Axes):
        """Plot market regime performance"""
        ax.set_title("Market Regime Performance")
        
        # Use the best performing curve for regime analysis
        best_curve = max(curves, key=lambda c: c.total_return)
        
        if hasattr(best_curve, '_regime_performance') and best_curve._regime_performance:
            regimes = list(best_curve._regime_performance.keys())
            sharpes = [best_curve._regime_performance[r].get('sharpe_ratio', 0) for r in regimes]
            counts = [best_curve._regime_performance[r].get('count', 0) for r in regimes]
            
            # Create bar chart with size proportional to count
            bars = ax.bar(range(len(regimes)), sharpes,
                         color=['green' if s > 0 else 'red' for s in sharpes], 
                         alpha=0.7)
            
            ax.set_xlabel('Market Regime')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_xticks(range(len(regimes)))
            ax.set_xticklabels(regimes, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add count labels
            for bar, sharpe, count in zip(bars, sharpes, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (0.01 if height > 0 else -0.01),
                       f'{sharpe:.2f}\n(n={count})', ha='center', 
                       va='bottom' if height > 0 else 'top', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No regime data available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _add_summary_table(self, fig: plt.Figure, curves: List[EquityCurve]):
        """Add summary statistics table"""
        # Create summary data
        summary_data = []
        for curve in curves:
            summary_data.append({
                'ID': curve.id,
                'Return': f"{curve.total_return:.2%}",
                'Sharpe': f"{curve.sharpe_ratio:.2f}",
                'Max DD': f"{curve.max_drawdown:.2%}",
                'Trades': curve.total_trades,
                'Win Rate': f"{curve.win_rate:.1%}"
            })
        
        # Add table as text
        fig.text(0.02, 0.02, "Summary Statistics:", fontsize=12, fontweight='bold')
        
        table_text = ""
        for data in summary_data:
            table_text += f"{data['ID']}: Return={data['Return']}, Sharpe={data['Sharpe']}, DD={data['Max DD']}, Trades={data['Trades']}\n"
        
        fig.text(0.02, 0.01, table_text, fontsize=10, verticalalignment='bottom')
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics for all curves"""
        if not self.curves:
            return {
                'view_type': 'empty',
                'total_curves': 0,
                'symbols': [],
                'strategies': [],
                'curves': [],
                'best_performer': None,
                'worst_performer': None
            }
        
        summary = {
            'view_type': self.view_type,
            'total_curves': len(self.curves),
            'symbols': list(set(curve.symbol for curve in self.curves)),
            'strategies': list(set(curve.strategy for curve in self.curves)),
            'curves': [curve.to_dict() for curve in self.curves],
            'best_performer': max(self.curves, key=lambda c: c.total_return).to_dict(),
            'worst_performer': min(self.curves, key=lambda c: c.total_return).to_dict()
        }
        
        return summary
