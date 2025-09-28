#!/usr/bin/env python3
"""
Strategy Backtester with Portfolio Management Integration
Implements walk-forward backtesting with regime detection and correlation analysis
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

logger = logging.getLogger(__name__)

class StrategyBacktester:
    """Advanced strategy backtester with portfolio management"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 rebalance_frequency: str = "daily",
                 walk_forward_periods: int = 30,
                 min_trades_per_strategy: int = 10):
        
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.walk_forward_periods = walk_forward_periods
        self.min_trades_per_strategy = min_trades_per_strategy
        
        # Portfolio manager
        self.portfolio_manager = PortfolioManager()
        
        # Backtest results
        self.results = {
            'equity_curve': [],
            'drawdowns': [],
            'allocations': [],
            'strategy_returns': {},
            'regime_history': [],
            'correlation_history': [],
            'trade_log': []
        }
        
        # Current state
        self.current_capital = initial_capital
        self.current_allocation = {}
        self.trade_count = 0
        
    def add_strategy_data(self, 
                         strategy_name: str,
                         price_data: pd.DataFrame,
                         signal_data: pd.DataFrame,
                         strategy_config: Dict[str, Any]):
        """Add strategy data for backtesting"""
        
        # Generate returns from signals and prices
        returns = self._generate_strategy_returns(price_data, signal_data, strategy_config)
        
        # Create strategy metrics
        strategy = StrategyMetrics(
            name=strategy_name,
            returns=returns,
            regime_filter=strategy_config.get('regime_filter', 'all'),
            instrument_class=strategy_config.get('instrument_class', 'equity'),
            mechanism=strategy_config.get('mechanism', 'trend'),
            horizon=strategy_config.get('horizon', 'intraday'),
            session=strategy_config.get('session', 'NY')
        )
        
        # Add to portfolio manager
        self.portfolio_manager.add_strategy(strategy)
        
        logger.info(f"Added strategy data for {strategy_name}: {len(returns)} returns")
        
    def _generate_strategy_returns(self, 
                                 price_data: pd.DataFrame, 
                                 signal_data: pd.DataFrame, 
                                 config: Dict[str, Any]) -> np.ndarray:
        """Generate strategy returns from price and signal data"""
        
        # Align data
        common_index = price_data.index.intersection(signal_data.index)
        if len(common_index) == 0:
            logger.warning("No common index between price and signal data")
            return np.array([])
            
        prices = price_data.loc[common_index, 'close']
        signals = signal_data.loc[common_index, 'signal']
        
        # Calculate returns
        price_returns = prices.pct_change().dropna()
        
        # Generate strategy returns based on signals
        strategy_returns = []
        position = 0
        
        for i, (timestamp, signal) in enumerate(signals.items()):
            if i == 0:
                continue
                
            # Get price return for this period
            if timestamp in price_returns.index:
                price_return = price_returns[timestamp]
                
                # Update position based on signal
                if signal == 1:  # Buy signal
                    position = 1
                elif signal == -1:  # Sell signal
                    position = 0
                # Hold signal keeps current position
                
                # Calculate strategy return
                strategy_return = position * price_return
                strategy_returns.append(strategy_return)
        
        return np.array(strategy_returns)
    
    def detect_regime(self, price_data: pd.DataFrame, window: int = 20) -> RegimeState:
        """Detect current market regime from price data"""
        
        if len(price_data) < window:
            return RegimeState()
            
        # Get recent data
        recent_data = price_data.tail(window)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) < 5:
            return RegimeState()
        
        # Trend vs Range detection using ADX
        high = recent_data['high']
        low = recent_data['low']
        close = recent_data['close']
        
        # Calculate ADX (simplified)
        tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
        atr = tr.rolling(window=14).mean()
        
        # Calculate directional movement
        dm_plus = np.maximum(high - high.shift(1), 0)
        dm_minus = np.maximum(low.shift(1) - low, 0)
        
        di_plus = 100 * (dm_plus.rolling(window=14).mean() / atr)
        di_minus = 100 * (dm_minus.rolling(window=14).mean() / atr)
        
        adx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = adx.dropna()
        
        # Determine trend vs range
        if len(adx) > 0 and adx.iloc[-1] > 25:
            trend_vs_range = "trend"
        else:
            trend_vs_range = "range"
        
        # Volatility regime
        vol = returns.std() * np.sqrt(252)  # Annualized volatility
        if vol > 0.25:
            volatility_regime = "high"
        elif vol < 0.15:
            volatility_regime = "low"
        else:
            volatility_regime = "normal"
        
        # Session detection (simplified)
        current_hour = datetime.now().hour
        if 0 <= current_hour < 8:
            session = "Asia"
        elif 8 <= current_hour < 16:
            session = "London"
        else:
            session = "NY"
        
        # Liquidity state (simplified based on volume)
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].mean()
            recent_volume = recent_data['volume'].iloc[-1]
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
            news_impact="low"  # Simplified
        )
    
    def run_walk_forward_backtest(self, 
                                start_date: datetime, 
                                end_date: datetime,
                                price_data: Dict[str, pd.DataFrame],
                                signal_data: Dict[str, pd.DataFrame],
                                strategy_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run walk-forward backtest with portfolio management"""
        
        logger.info(f"Starting walk-forward backtest from {start_date} to {end_date}")
        
        # Initialize results
        self.results = {
            'equity_curve': [],
            'drawdowns': [],
            'allocations': [],
            'strategy_returns': {name: [] for name in strategy_configs.keys()},
            'regime_history': [],
            'correlation_history': [],
            'trade_log': []
        }
        
        self.current_capital = self.initial_capital
        
        # Walk forward through time
        current_date = start_date
        period_count = 0
        
        while current_date < end_date:
            period_count += 1
            logger.info(f"Processing period {period_count}: {current_date}")
            
            # Get data for this period
            period_data = self._get_period_data(current_date, price_data, signal_data)
            
            if not period_data:
                current_date += timedelta(days=1)
                continue
            
            # Detect regime for this period
            regime = self.detect_regime(period_data['combined_prices'])
            self.portfolio_manager.update_regime(regime)
            
            # Update strategy data
            for strategy_name, config in strategy_configs.items():
                if strategy_name in period_data['signals']:
                    self._update_strategy_data(
                        strategy_name, 
                        period_data['prices'][strategy_name],
                        period_data['signals'][strategy_name],
                        config
                    )
            
            # Calculate portfolio allocation
            allocation = self.portfolio_manager.calculate_portfolio_allocation()
            
            if allocation:
                # Execute trades based on allocation
                period_returns = self._execute_trades(allocation, period_data)
                
                # Update capital
                self.current_capital *= (1 + period_returns)
                
                # Store results
                self.results['equity_curve'].append({
                    'date': current_date,
                    'capital': self.current_capital,
                    'return': period_returns
                })
                
                self.results['allocations'].append({
                    'date': current_date,
                    'allocation': allocation.copy()
                })
                
                self.results['regime_history'].append({
                    'date': current_date,
                    'regime': {
                        'trend_vs_range': regime.trend_vs_range,
                        'volatility_regime': regime.volatility_regime,
                        'session': regime.session,
                        'liquidity_state': regime.liquidity_state
                    }
                })
                
                # Update portfolio manager
                self.portfolio_manager.update_portfolio_returns(period_returns)
            
            # Move to next period
            current_date += timedelta(days=1)
            
            # Rebalance if needed
            if period_count % self.walk_forward_periods == 0:
                logger.info(f"Rebalancing at period {period_count}")
                self._rebalance_strategies()
        
        # Calculate final metrics
        final_results = self._calculate_final_metrics()
        
        logger.info(f"Backtest completed. Final capital: ${self.current_capital:,.2f}")
        logger.info(f"Total return: {(self.current_capital / self.initial_capital - 1) * 100:.2f}%")
        
        return final_results
    
    def _get_period_data(self, 
                        current_date: datetime, 
                        price_data: Dict[str, pd.DataFrame],
                        signal_data: Dict[str, pd.DataFrame]) -> Optional[Dict[str, Any]]:
        """Get data for a specific period"""
        
        period_data = {
            'prices': {},
            'signals': {},
            'combined_prices': None
        }
        
        # Get data for each strategy
        for strategy_name in price_data.keys():
            if strategy_name in price_data and strategy_name in signal_data:
                # Filter data for this period
                price_df = price_data[strategy_name]
                signal_df = signal_data[strategy_name]
                
                # Get data up to current date
                period_prices = price_df[price_df.index <= current_date]
                period_signals = signal_df[signal_df.index <= current_date]
                
                if len(period_prices) > 0 and len(period_signals) > 0:
                    period_data['prices'][strategy_name] = period_prices
                    period_data['signals'][strategy_name] = period_signals
                    
                    # Use first strategy's prices as combined reference
                    if period_data['combined_prices'] is None:
                        period_data['combined_prices'] = period_prices
        
        return period_data if period_data['prices'] else None
    
    def _update_strategy_data(self, 
                             strategy_name: str,
                             price_data: pd.DataFrame,
                             signal_data: pd.DataFrame,
                             config: Dict[str, Any]):
        """Update strategy data in portfolio manager"""
        
        # Generate returns
        returns = self._generate_strategy_returns(price_data, signal_data, config)
        
        if len(returns) == 0:
            return
        
        # Update or create strategy
        if strategy_name in self.portfolio_manager.strategies:
            # Update existing strategy
            self.portfolio_manager.strategies[strategy_name].returns = returns
        else:
            # Create new strategy
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
    
    def _execute_trades(self, allocation: Dict[str, float], period_data: Dict[str, Any]) -> float:
        """Execute trades based on allocation and return period return"""
        
        # Calculate weighted return across all strategies
        total_return = 0.0
        
        for strategy_name, weight in allocation.items():
            if strategy_name in period_data['prices']:
                # Get recent returns for this strategy
                prices = period_data['prices'][strategy_name]['close']
                if len(prices) > 1:
                    strategy_return = prices.pct_change().iloc[-1]
                    if not np.isnan(strategy_return):
                        total_return += weight * strategy_return
        
        # Log trade
        self.trade_count += 1
        self.results['trade_log'].append({
            'trade_id': self.trade_count,
            'date': period_data['combined_prices'].index[-1] if period_data['combined_prices'] is not None else datetime.now(),
            'allocation': allocation.copy(),
            'return': total_return,
            'capital': self.current_capital
        })
        
        return total_return
    
    def _rebalance_strategies(self):
        """Rebalance strategies using walk-forward approach"""
        # This would involve retraining/recalibrating strategies
        # For now, we'll just recalculate allocations
        pass
    
    def _calculate_final_metrics(self) -> Dict[str, Any]:
        """Calculate final backtest metrics"""
        
        if not self.results['equity_curve']:
            return {}
        
        # Extract equity curve
        equity_df = pd.DataFrame(self.results['equity_curve'])
        equity_df.set_index('date', inplace=True)
        
        # Calculate returns
        returns = equity_df['return']
        
        # Basic metrics
        total_return = (self.current_capital / self.initial_capital - 1) * 100
        annualized_return = (self.current_capital / self.initial_capital) ** (252 / len(returns)) - 1
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
        
        # Strategy performance
        strategy_performance = {}
        for strategy_name in self.results['strategy_returns']:
            if self.results['strategy_returns'][strategy_name]:
                strategy_returns = np.array(self.results['strategy_returns'][strategy_name])
                strategy_performance[strategy_name] = {
                    'total_return': (1 + strategy_returns).prod() - 1,
                    'volatility': strategy_returns.std() * np.sqrt(252),
                    'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
                    'max_drawdown': self._calculate_max_drawdown(strategy_returns)
                }
        
        return {
            'total_return_pct': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'final_capital': self.current_capital,
            'num_trades': self.trade_count,
            'strategy_performance': strategy_performance,
            'equity_curve': equity_df,
            'allocations': self.results['allocations'],
            'regime_history': self.results['regime_history']
        }
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown for a return series"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns.min()
    
    def generate_report(self, results: Dict[str, Any], output_dir: str = "backtest_results"):
        """Generate comprehensive backtest report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate equity curve plot
        self._plot_equity_curve(results, output_path)
        
        # Generate allocation heatmap
        self._plot_allocation_heatmap(results, output_path)
        
        # Generate correlation analysis
        self._plot_correlation_analysis(output_path)
        
        # Generate regime analysis
        self._plot_regime_analysis(results, output_path)
        
        # Save detailed results
        self._save_detailed_results(results, output_path)
        
        logger.info(f"Backtest report generated in {output_path}")
    
    def _plot_equity_curve(self, results: Dict[str, Any], output_path: Path):
        """Plot equity curve and drawdowns"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        equity_df = results['equity_curve']
        ax1.plot(equity_df.index, equity_df['capital'])
        ax1.set_title('Portfolio Equity Curve')
        ax1.set_ylabel('Capital ($)')
        ax1.grid(True)
        
        # Drawdowns
        returns = equity_df['return']
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        
        ax2.fill_between(equity_df.index, drawdowns, 0, alpha=0.3, color='red')
        ax2.plot(equity_df.index, drawdowns, color='red')
        ax2.set_title('Portfolio Drawdowns')
        ax2.set_ylabel('Drawdown')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_allocation_heatmap(self, results: Dict[str, Any], output_path: Path):
        """Plot allocation heatmap over time"""
        
        if not results['allocations']:
            return
        
        # Create allocation matrix
        allocation_df = pd.DataFrame([a['allocation'] for a in results['allocations']])
        allocation_df.index = [a['date'] for a in results['allocations']]
        
        if allocation_df.empty:
            return
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(allocation_df.T, cmap='YlOrRd', cbar=True)
        plt.title('Portfolio Allocation Over Time')
        plt.xlabel('Date')
        plt.ylabel('Strategy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'allocation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self, output_path: Path):
        """Plot correlation analysis"""
        
        if self.portfolio_manager.correlation_matrix is None:
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.portfolio_manager.correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True)
        plt.title('Strategy Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_regime_analysis(self, results: Dict[str, Any], output_path: Path):
        """Plot regime analysis over time"""
        
        if not results['regime_history']:
            return
        
        regime_df = pd.DataFrame(results['regime_history'])
        regime_df.set_index('date', inplace=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trend vs Range
        regime_df['trend_vs_range'].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Trend vs Range Distribution')
        
        # Volatility regime
        regime_df['volatility_regime'].value_counts().plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Volatility Regime Distribution')
        
        # Session
        regime_df['session'].value_counts().plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Session Distribution')
        
        # Liquidity state
        regime_df['liquidity_state'].value_counts().plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Liquidity State Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path / 'regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_results(self, results: Dict[str, Any], output_path: Path):
        """Save detailed results to JSON"""
        
        # Convert numpy types to Python types for JSON serialization
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
                clean_results[key] = value.to_dict('records')
            else:
                clean_results[key] = value
        
        # Save to file
        with open(output_path / 'detailed_results.json', 'w') as f:
            json.dump(clean_results, f, indent=2, default=convert_numpy)

# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create backtester
    backtester = StrategyBacktester(
        initial_capital=100000,
        rebalance_frequency="daily",
        walk_forward_periods=30
    )
    
    # Example: This would be called with real data
    print("Strategy backtester created. Use with real market data for backtesting.")
