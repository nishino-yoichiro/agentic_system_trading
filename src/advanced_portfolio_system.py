#!/usr/bin/env python3
"""
Advanced Portfolio Management System
Complete integration of low-correlation strategy basket with Kelly sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from portfolio_manager import PortfolioManager, StrategyMetrics, RegimeState
from strategy_backtester import StrategyBacktester
from strategy_framework import StrategyFramework, BaseStrategy

logger = logging.getLogger(__name__)

class AdvancedPortfolioSystem:
    """Complete advanced portfolio management system"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 target_volatility: float = 0.10,
                 max_strategy_weight: float = 0.25,
                 correlation_threshold: float = 0.5,
                 fractional_kelly: float = 0.25):
        
        # Initialize components
        self.portfolio_manager = PortfolioManager(
            target_volatility=target_volatility,
            max_strategy_weight=max_strategy_weight,
            correlation_threshold=correlation_threshold,
            fractional_kelly=fractional_kelly
        )
        
        self.backtester = StrategyBacktester(
            initial_capital=initial_capital,
            rebalance_frequency="daily",
            walk_forward_periods=30
        )
        
        self.strategy_framework = StrategyFramework()
        
        # System state
        self.current_capital = initial_capital
        self.current_allocation = {}
        self.performance_history = []
        
        # Create default strategies
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self):
        """Initialize the 5-edge strategy basket"""
        
        # Create default strategies
        strategies = self.strategy_framework.create_default_strategies()
        
        # Add to framework
        for strategy in strategies.values():
            self.strategy_framework.add_strategy(strategy)
        
        logger.info("Initialized 5-edge strategy basket")
        
    def run_live_trading_simulation(self, 
                                  market_data: Dict[str, pd.DataFrame],
                                  start_date: datetime,
                                  end_date: datetime) -> Dict[str, Any]:
        """Run live trading simulation with portfolio management"""
        
        logger.info(f"Starting live trading simulation from {start_date} to {end_date}")
        
        # Initialize results
        results = {
            'equity_curve': [],
            'allocations': [],
            'signals': [],
            'regime_history': [],
            'performance_metrics': {},
            'strategy_performance': {}
        }
        
        current_date = start_date
        
        while current_date <= end_date:
            logger.info(f"Processing {current_date}")
            
            # Get current market data
            current_data = self._get_current_data(current_date, market_data)
            
            if not current_data:
                current_date += timedelta(days=1)
                continue
            
            # Detect current regime
            regime = self._detect_regime(current_data)
            self.portfolio_manager.update_regime(regime)
            
            # Generate signals for all strategies
            signals = self.strategy_framework.generate_all_signals(current_data)
            
            # Update strategy data in portfolio manager
            self._update_strategy_data(signals, current_data)
            
            # Calculate portfolio allocation
            allocation = self.portfolio_manager.calculate_portfolio_allocation()
            
            if allocation:
                # Execute trades
                period_return = self._execute_trades(allocation, current_data, signals)
                
                # Update capital
                self.current_capital *= (1 + period_return)
                
                # Store results
                results['equity_curve'].append({
                    'date': current_date,
                    'capital': self.current_capital,
                    'return': period_return
                })
                
                results['allocations'].append({
                    'date': current_date,
                    'allocation': allocation.copy()
                })
                
                results['signals'].append({
                    'date': current_date,
                    'signals': {name: signal[0].__dict__ for name, signal in signals.items()}
                })
                
                results['regime_history'].append({
                    'date': current_date,
                    'regime': {
                        'trend_vs_range': regime.trend_vs_range,
                        'volatility_regime': regime.volatility_regime,
                        'session': regime.session,
                        'liquidity_state': regime.liquidity_state
                    }
                })
                
                # Update portfolio manager
                self.portfolio_manager.update_portfolio_returns(period_return)
            
            current_date += timedelta(days=1)
        
        # Calculate final metrics
        results['performance_metrics'] = self._calculate_performance_metrics(results)
        results['strategy_performance'] = self._calculate_strategy_performance(results)
        
        logger.info(f"Simulation completed. Final capital: ${self.current_capital:,.2f}")
        
        return results
    
    def _get_current_data(self, current_date: datetime, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Get market data up to current date"""
        
        current_data = {}
        
        for symbol, data in market_data.items():
            # Filter data up to current date
            filtered_data = data[data.index <= current_date]
            if len(filtered_data) > 0:
                current_data[symbol] = filtered_data
        
        # If no data available, return empty dict
        if not current_data:
            logger.warning(f"No data available for {current_date}")
            return {}
        
        return current_data
    
    def _detect_regime(self, data: Dict[str, pd.DataFrame]) -> RegimeState:
        """Detect current market regime"""
        
        # Use first available data for regime detection
        if not data:
            return RegimeState()
        
        first_symbol = list(data.keys())[0]
        price_data = data[first_symbol]
        
        if len(price_data) < 20:
            return RegimeState()
        
        # Calculate regime indicators
        returns = price_data['close'].pct_change().dropna()
        
        # Trend vs Range (using ADX-like calculation)
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Simplified ADX calculation
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        atr = tr.rolling(window=14).mean()
        
        dm_plus = np.maximum(high - high.shift(1), 0)
        dm_minus = np.maximum(low.shift(1) - low, 0)
        
        di_plus = 100 * (dm_plus.rolling(window=14).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=14).mean() / atr)
        
        adx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = adx.dropna()
        
        # Determine regime
        if len(adx) > 0 and adx.iloc[-1] > 25:
            trend_vs_range = "trend"
        else:
            trend_vs_range = "range"
        
        # Volatility regime
        vol = returns.std() * np.sqrt(252)
        if vol > 0.25:
            volatility_regime = "high"
        elif vol < 0.15:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"
        
        # Session detection
        current_hour = datetime.now().hour
        if 0 <= current_hour < 8:
            session = "Asia"
        elif 8 <= current_hour < 16:
            session = "London"
        else:
            session = "NY"
        
        # Liquidity state
        if 'volume' in price_data.columns:
            avg_volume = price_data['volume'].mean()
            recent_volume = price_data['volume'].iloc[-1]
            if recent_volume > avg_volume * 1.5:
                liquidity_state = "high"
            elif recent_volume < avg_volume * 0.5:
                liquidity_state = "low"
            else:
                liquidity_state = "normal"
        else:
            liquidity_state = "normal"
        
        return RegimeState(
            trend_vs_range=trend_vs_range,
            volatility_regime=volatility_regime,
            session=session,
            liquidity_state=liquidity_state,
            news_impact="low"
        )
    
    def _update_strategy_data(self, signals: Dict[str, List], data: Dict[str, pd.DataFrame]):
        """Update strategy data in portfolio manager"""
        
        for strategy_name, signal_list in signals.items():
            if not signal_list:
                continue
                
            signal = signal_list[0]
            
            # Generate returns from signal
            if strategy_name in data:
                price_data = data[strategy_name]
                returns = self._generate_returns_from_signal(price_data, signal)
                
                if len(returns) > 0:
                    # Create or update strategy
                    if strategy_name in self.portfolio_manager.strategies:
                        self.portfolio_manager.strategies[strategy_name].returns = returns
                    else:
                        # Get strategy config
                        configs = self.strategy_framework.get_strategy_configs()
                        config = configs.get(strategy_name, {})
                        
                        strategy = StrategyMetrics(
                            name=strategy_name,
                            returns=returns,
                            regime_filter=config.get('regime_filter', 'all'),
                            instrument_class=config.get('instrument_class', 'equity'),
                            mechanism=config.get('mechanism', 'trend'),
                            horizon=config.get('horizon', 'intraday'),
                            session=config.get('session', 'NY')
                        )
                        self.portfolio_manager.add_strategy(strategy)
    
    def _generate_returns_from_signal(self, price_data: pd.DataFrame, signal) -> np.ndarray:
        """Generate returns from signal and price data"""
        
        if len(price_data) < 2:
            return np.array([])
        
        # Calculate price returns
        price_returns = price_data['close'].pct_change().dropna()
        
        # Generate strategy returns based on signal
        strategy_returns = []
        position = 0
        
        for i, (timestamp, price_return) in enumerate(price_returns.items()):
            # Update position based on signal (simplified)
            if i == len(price_returns) - 1:  # Last period
                position = signal.signal
            
            # Calculate strategy return
            strategy_return = position * price_return
            strategy_returns.append(strategy_return)
        
        return np.array(strategy_returns)
    
    def _execute_trades(self, allocation: Dict[str, float], data: Dict[str, pd.DataFrame], signals: Dict[str, List]) -> float:
        """Execute trades based on allocation"""
        
        total_return = 0.0
        
        for strategy_name, weight in allocation.items():
            if strategy_name in data and strategy_name in signals:
                # Get recent price return
                price_data = data[strategy_name]
                if len(price_data) > 1:
                    price_return = price_data['close'].pct_change().iloc[-1]
                    if not np.isnan(price_return):
                        total_return += weight * price_return
        
        return total_return
    
    def _calculate_performance_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        
        if not results['equity_curve']:
            return {}
        
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        
        returns = equity_df['return']
        
        # Basic metrics
        total_return = (self.current_capital / 100000 - 1) * 100
        annualized_return = (self.current_capital / 100000) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return_pct': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'final_capital': self.current_capital
        }
    
    def _calculate_strategy_performance(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Calculate individual strategy performance"""
        
        strategy_performance = {}
        
        for strategy_name in self.portfolio_manager.strategies:
            strategy = self.portfolio_manager.strategies[strategy_name]
            
            if len(strategy.returns) > 0:
                returns = strategy.returns
                
                strategy_performance[strategy_name] = {
                    'total_return': (1 + returns).prod() - 1,
                    'volatility': returns.std() * np.sqrt(252),
                    'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(returns),
                    'hit_rate': (returns > 0).mean()
                }
        
        return strategy_performance
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()
    
    def generate_comprehensive_report(self, results: Dict[str, Any], output_dir: str = "portfolio_reports"):
        """Generate comprehensive portfolio report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate plots
        self._plot_equity_curve(results, output_path)
        self._plot_allocation_heatmap(results, output_path)
        self._plot_correlation_analysis(results, output_path)
        self._plot_regime_analysis(results, output_path)
        self._plot_strategy_performance(results, output_path)
        
        # Save detailed results
        self._save_detailed_results(results, output_path)
        
        # Generate markdown report
        self._generate_markdown_report(results, output_path)
        
        logger.info(f"Comprehensive report generated in {output_path}")
    
    def _plot_equity_curve(self, results: Dict[str, Any], output_path: Path):
        """Plot equity curve and drawdowns"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Equity curve
        equity_df = pd.DataFrame(results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        
        ax1.plot(equity_df.index, equity_df['capital'], linewidth=2, color='blue')
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Capital ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100000, color='red', linestyle='--', alpha=0.7, label='Initial Capital')
        ax1.legend()
        
        # Drawdowns
        returns = equity_df['return']
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        ax2.fill_between(equity_df.index, drawdowns, 0, alpha=0.3, color='red')
        ax2.plot(equity_df.index, drawdowns, color='red', linewidth=1)
        ax2.set_title('Portfolio Drawdowns', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Drawdown', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_allocation_heatmap(self, results: Dict[str, Any], output_path: Path):
        """Plot allocation heatmap over time"""
        
        if not results['allocations']:
            return
        
        allocation_df = pd.DataFrame([a['allocation'] for a in results['allocations']])
        allocation_df.index = [a['date'] for a in results['allocations']]
        
        if allocation_df.empty:
            return
        
        plt.figure(figsize=(15, 8))
        sns.heatmap(allocation_df.T, cmap='YlOrRd', cbar=True, cbar_kws={'label': 'Allocation Weight'})
        plt.title('Portfolio Allocation Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Strategy', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'allocation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot correlation analysis"""
        
        if self.portfolio_manager.correlation_matrix is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Correlation matrix heatmap
        sns.heatmap(self.portfolio_manager.correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   ax=ax1)
        ax1.set_title('Strategy Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Average correlation over time
        if results['allocations']:
            avg_correlations = []
            dates = []
            
            for allocation in results['allocations']:
                if self.portfolio_manager.correlation_matrix is not None:
                    avg_corr = np.mean(np.abs(self.portfolio_manager.correlation_matrix))
                    avg_correlations.append(avg_corr)
                    dates.append(allocation['date'])
            
            if avg_correlations:
                ax2.plot(dates, avg_correlations, linewidth=2, color='blue')
                ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Correlation Threshold')
                ax2.set_title('Average Correlation Over Time', fontsize=14, fontweight='bold')
                ax2.set_ylabel('Average |Correlation|', fontsize=12)
                ax2.set_xlabel('Date', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot regime analysis over time"""
        
        if not results['regime_history']:
            return
        
        # Flatten the regime data structure
        regime_data = []
        for entry in results['regime_history']:
            regime_data.append({
                'date': entry['date'],
                'trend_vs_range': entry['regime']['trend_vs_range'],
                'volatility_regime': entry['regime']['volatility_regime'],
                'session': entry['regime']['session'],
                'liquidity_state': entry['regime']['liquidity_state']
            })
        
        regime_df = pd.DataFrame(regime_data)
        regime_df.set_index('date', inplace=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trend vs Range
        regime_df['trend_vs_range'].value_counts().plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Trend vs Range Distribution', fontsize=12, fontweight='bold')
        axes[0,0].set_ylabel('Count')
        
        # Volatility regime
        regime_df['volatility_regime'].value_counts().plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('Volatility Regime Distribution', fontsize=12, fontweight='bold')
        axes[0,1].set_ylabel('Count')
        
        # Session
        regime_df['session'].value_counts().plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Session Distribution', fontsize=12, fontweight='bold')
        axes[1,0].set_ylabel('Count')
        
        # Liquidity state
        regime_df['liquidity_state'].value_counts().plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('Liquidity State Distribution', fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(output_path / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_strategy_performance(self, results: Dict[str, Any], output_path: Path):
        """Plot individual strategy performance"""
        
        if 'strategy_performance' not in results:
            return
        
        strategy_perf = results['strategy_performance']
        
        if not strategy_perf:
            return
        
        # Create performance comparison
        strategies = list(strategy_perf.keys())
        metrics = ['total_return', 'volatility', 'sharpe', 'max_drawdown']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [strategy_perf[strategy][metric] for strategy in strategies]
            
            bars = axes[i].bar(strategies, values, color=plt.cm.viridis(np.linspace(0, 1, len(strategies))))
            axes[i].set_title(f'Strategy {metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_results(self, results: Dict[str, Any], output_path: Path):
        """Save detailed results to JSON"""
        
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if key == 'equity_curve':
                clean_results[key] = pd.DataFrame(value).to_dict('records')
            elif key == 'signals':
                # Clean signals to avoid circular references
                clean_signals = []
                for signal_entry in value:
                    clean_signal_entry = {
                        'date': signal_entry['date'].isoformat() if hasattr(signal_entry['date'], 'isoformat') else str(signal_entry['date']),
                        'signals': {}
                    }
                    for name, signal_dict in signal_entry['signals'].items():
                        # Convert signal dict to basic types
                        clean_signal = {}
                        for attr, val in signal_dict.items():
                            if isinstance(val, (str, int, float, bool, type(None))):
                                clean_signal[attr] = val
                            elif hasattr(val, 'isoformat'):  # datetime
                                clean_signal[attr] = val.isoformat()
                            else:
                                clean_signal[attr] = str(val)
                        clean_signal_entry['signals'][name] = clean_signal
                    clean_signals.append(clean_signal_entry)
                clean_results[key] = clean_signals
            elif key == 'regime_history':
                # Clean regime history
                clean_regime_history = []
                for regime_entry in value:
                    clean_regime_entry = {
                        'date': regime_entry['date'].isoformat() if hasattr(regime_entry['date'], 'isoformat') else str(regime_entry['date']),
                        'regime': regime_entry['regime']
                    }
                    clean_regime_history.append(clean_regime_entry)
                clean_results[key] = clean_regime_history
            else:
                clean_results[key] = value
        
        # Save a simplified version to avoid circular references
        simplified_results = {
            'summary': {
                'initial_capital': results.get('initial_capital', 0),
                'final_capital': results.get('final_capital', 0),
                'total_return': results.get('total_return', 0),
                'max_drawdown': results.get('max_drawdown', 0),
                'sharpe_ratio': results.get('sharpe_ratio', 0),
                'win_rate': results.get('win_rate', 0)
            },
            'equity_curve': pd.DataFrame(results['equity_curve']).to_dict('records') if 'equity_curve' in results else [],
            'strategy_performance': results.get('strategy_performance', {}),
            'allocation_history': results.get('allocation_history', [])[-100:],  # Last 100 allocations
            'regime_summary': {
                'total_days': len(results.get('regime_history', [])),
                'trend_days': sum(1 for r in results.get('regime_history', []) if r.get('regime', {}).get('trend_vs_range') == 'trend'),
                'range_days': sum(1 for r in results.get('regime_history', []) if r.get('regime', {}).get('trend_vs_range') == 'range')
            }
        }
        
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(simplified_results, f, indent=2, default=convert_numpy)
    
    def _generate_markdown_report(self, results: Dict[str, Any], output_path: Path):
        """Generate markdown report"""
        
        report = f"""# Advanced Portfolio Management System Report

## Executive Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Portfolio Performance

"""
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            report += f"""
- **Total Return:** {metrics.get('total_return_pct', 0):.2f}%
- **Annualized Return:** {metrics.get('annualized_return', 0):.2f}
- **Volatility:** {metrics.get('volatility', 0):.2f}
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}
- **Max Drawdown:** {metrics.get('max_drawdown', 0):.2f}
- **Calmar Ratio:** {metrics.get('calmar_ratio', 0):.2f}
- **Win Rate:** {metrics.get('win_rate', 0):.2f}
- **Final Capital:** ${metrics.get('final_capital', 0):,.2f}

"""
        
        report += """
## Strategy Performance

"""
        
        if 'strategy_performance' in results:
            for strategy, perf in results['strategy_performance'].items():
                report += f"""
### {strategy}
- **Total Return:** {perf.get('total_return', 0):.2f}
- **Volatility:** {perf.get('volatility', 0):.2f}
- **Sharpe Ratio:** {perf.get('sharpe', 0):.2f}
- **Max Drawdown:** {perf.get('max_drawdown', 0):.2f}
- **Hit Rate:** {perf.get('hit_rate', 0):.2f}

"""
        
        report += """
## Key Features

- **Low-Correlation Strategy Basket:** 5 orthogonal strategies across different instruments and mechanisms
- **Kelly Sizing:** Optimal position sizing based on expected returns and correlations
- **Regime Filtering:** Strategies only trade in favorable market conditions
- **Risk Controls:** Volatility targeting, correlation penalties, and kill switches
- **Walk-Forward Backtesting:** Out-of-sample testing with parameter re-estimation

## Strategy Basket

1. **ES Sweep/Reclaim** - Intraday futures strategy for NY open
2. **NQ Breakout Continuation** - High-vol regime breakout strategy
3. **SPY Mean Reversion** - Overnight gap fade in low-vol regimes
4. **EURUSD Carry/Trend** - FX strategy with MA filter
5. **Options IV Crush** - Event-driven volatility strategy

## Risk Management

- Volatility targeting to 10% annualized
- Maximum 25% allocation per strategy
- Correlation threshold of 0.5
- Fractional Kelly sizing (25% of full Kelly)
- Regime-based strategy activation
- Portfolio-level kill switches

"""
        
        with open(output_path / 'portfolio_report.md', 'w') as f:
            f.write(report)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create advanced portfolio system
    system = AdvancedPortfolioSystem(
        initial_capital=100000,
        target_volatility=0.10,
        max_strategy_weight=0.25,
        correlation_threshold=0.5,
        fractional_kelly=0.25
    )
    
    print("Advanced Portfolio Management System initialized")
    print("Features:")
    print("- 5-edge low-correlation strategy basket")
    print("- Kelly sizing with correlation control")
    print("- Regime-based strategy filtering")
    print("- Volatility targeting and risk controls")
    print("- Walk-forward backtesting framework")
    print("- Comprehensive reporting and visualization")
