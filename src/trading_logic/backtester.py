"""
Backtesting System

Backtest trading strategies on historical data
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float


@dataclass
class BacktestResult:
    """Backtest results"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: pd.Series


class Backtester:
    """Backtest trading strategies"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.equity_curve = []
    
    def run_backtest(self, price_data: pd.DataFrame, signals: pd.DataFrame, 
                    strategy_name: str = "Strategy") -> BacktestResult:
        """Run backtest on historical data"""
        try:
            logger.info(f"Starting backtest for {strategy_name}")
            
            # Initialize
            self.trades = []
            self.equity_curve = [self.initial_capital]
            current_capital = self.initial_capital
            position = None
            
            # Process each time step
            for i, (timestamp, row) in enumerate(price_data.iterrows()):
                current_price = row['close']
                
                # Check for exit signals
                if position:
                    exit_signal = self._check_exit_signal(position, current_price, signals.loc[timestamp] if timestamp in signals.index else None)
                    if exit_signal:
                        trade = self._close_position(position, current_price, timestamp)
                        self.trades.append(trade)
                        current_capital += trade.pnl
                        position = None
                
                # Check for entry signals
                if not position and timestamp in signals.index:
                    signal = signals.loc[timestamp]
                    if signal['action'] in ['BUY', 'SELL']:
                        position = self._open_position(signal, current_price, timestamp, current_capital)
                
                # Update equity curve
                if position:
                    unrealized_pnl = self._calculate_unrealized_pnl(position, current_price)
                    current_equity = current_capital + unrealized_pnl
                else:
                    current_equity = current_capital
                
                self.equity_curve.append(current_equity)
            
            # Close any remaining position
            if position:
                trade = self._close_position(position, price_data['close'].iloc[-1], price_data.index[-1])
                self.trades.append(trade)
                current_capital += trade.pnl
                self.equity_curve[-1] = current_capital
            
            # Calculate results
            result = self._calculate_results()
            logger.info(f"Backtest completed: {result.total_trades} trades, {result.total_return:.2%} return")
            
            return result
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, [], pd.Series())
    
    def _open_position(self, signal: pd.Series, price: float, timestamp: datetime, capital: float) -> Dict:
        """Open a new position"""
        side = 'long' if signal['action'] == 'BUY' else 'short'
        quantity = capital * 0.1 / price  # 10% of capital
        
        return {
            'symbol': signal.get('symbol', 'UNKNOWN'),
            'entry_time': timestamp,
            'entry_price': price,
            'quantity': quantity,
            'side': side,
            'stop_loss': price * 0.95 if side == 'long' else price * 1.05,
            'take_profit': price * 1.15 if side == 'long' else price * 0.85
        }
    
    def _close_position(self, position: Dict, price: float, timestamp: datetime) -> Trade:
        """Close a position and create trade record"""
        pnl = 0
        if position['side'] == 'long':
            pnl = (price - position['entry_price']) * position['quantity']
        else:  # short
            pnl = (position['entry_price'] - price) * position['quantity']
        
        commission = (position['entry_price'] + price) * position['quantity'] * self.commission
        net_pnl = pnl - commission
        
        pnl_pct = net_pnl / (position['entry_price'] * position['quantity'])
        
        return Trade(
            symbol=position['symbol'],
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry_price'],
            exit_price=price,
            quantity=position['quantity'],
            side=position['side'],
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=commission
        )
    
    def _check_exit_signal(self, position: Dict, price: float, signal: pd.Series) -> bool:
        """Check if position should be closed"""
        if signal is None:
            return False
        
        # Stop loss
        if position['side'] == 'long' and price <= position['stop_loss']:
            return True
        elif position['side'] == 'short' and price >= position['stop_loss']:
            return True
        
        # Take profit
        if position['side'] == 'long' and price >= position['take_profit']:
            return True
        elif position['side'] == 'short' and price <= position['take_profit']:
            return True
        
        # Signal-based exit
        if signal['action'] == 'SELL' and position['side'] == 'long':
            return True
        elif signal['action'] == 'BUY' and position['side'] == 'short':
            return True
        
        return False
    
    def _calculate_unrealized_pnl(self, position: Dict, price: float) -> float:
        """Calculate unrealized P&L for open position"""
        if position['side'] == 'long':
            return (price - position['entry_price']) * position['quantity']
        else:  # short
            return (position['entry_price'] - price) * position['quantity']
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest results"""
        if not self.trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, [], pd.Series())
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = total_pnl / self.initial_capital
        
        # Calculate max drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return=total_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            trades=self.trades,
            equity_curve=equity_series
        )
    
    def plot_results(self, result: BacktestResult, save_path: str = None):
        """Plot backtest results"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            ax1.plot(result.equity_curve.index, result.equity_curve.values)
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('Portfolio Value')
            ax1.grid(True)
            
            # Drawdown
            running_max = result.equity_curve.expanding().max()
            drawdown = (result.equity_curve - running_max) / running_max
            ax2.fill_between(drawdown.index, drawdown.values, alpha=0.3, color='red')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Date')
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Backtest results plot saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Error plotting backtest results: {e}")

