# Advanced Portfolio Management System

A comprehensive quantitative trading system implementing low-correlation strategy baskets with Kelly sizing, regime filtering, and advanced risk management.

## 🎯 Key Features

### Low-Correlation Strategy Basket
- **5 Orthogonal Strategies** across different instruments, mechanisms, and timeframes
- **Instrument Diversity**: Futures (ES, NQ), Equity (SPY), FX (EURUSD), Options (VIX)
- **Mechanism Diversity**: Sweep/Reclaim, Breakout, Mean Reversion, Carry, Volatility
- **Timeframe Diversity**: Intraday, Overnight, Swing, Event-driven

### Advanced Risk Management
- **Volatility Targeting**: All strategies normalized to 10% annualized volatility
- **Kelly Sizing**: Optimal position sizing based on expected returns and correlations
- **Correlation Control**: Automatic de-weighting of highly correlated strategies
- **Regime Filtering**: Strategies only trade in favorable market conditions
- **Kill Switches**: Portfolio-level risk controls and drawdown limits

### Portfolio Optimization
- **Risk-Parity Base**: Equal risk contribution across strategies
- **Sharpe Tilt**: Overweight higher Sharpe ratio strategies
- **Correlation Penalty**: Reduce allocation to crowded trades
- **Dynamic Rebalancing**: Walk-forward parameter estimation

## 🏗️ System Architecture

```
Advanced Portfolio System
├── Portfolio Manager (portfolio_manager.py)
│   ├── Volatility Targeting
│   ├── Kelly Sizing
│   ├── Correlation Analysis
│   └── Risk Controls
├── Strategy Framework (strategy_framework.py)
│   ├── 5-Edge Strategy Basket
│   ├── Signal Generation
│   └── Regime Compatibility
├── Backtester (strategy_backtester.py)
│   ├── Walk-Forward Testing
│   ├── Regime Detection
│   └── Performance Analysis
└── Integration (advanced_portfolio_system.py)
    ├── Live Trading Simulation
    ├── Comprehensive Reporting
    └── Visualization
```

## 📊 Strategy Basket

### 1. ES Sweep/Reclaim (Intraday Futures)
- **Instrument**: ES (S&P 500 futures)
- **Mechanism**: Liquidity sweep and reclaim
- **Regime**: All (but optimized for NY open)
- **Logic**: Buy on reclaim after sweep up, sell on reclaim after sweep down

### 2. NQ Breakout Continuation (High-Vol Regime)
- **Instrument**: NQ (Nasdaq futures)
- **Mechanism**: Breakout with volume confirmation
- **Regime**: High volatility only
- **Logic**: Follow breakouts with volume confirmation

### 3. SPY Mean Reversion (Low-Vol Regime)
- **Instrument**: SPY (S&P 500 ETF)
- **Mechanism**: Overnight gap fade
- **Regime**: Low volatility only
- **Logic**: Fade gaps with RSI confirmation

### 4. EURUSD Carry/Trend (Swing)
- **Instrument**: EUR/USD
- **Mechanism**: Carry trade with trend filter
- **Regime**: All (but optimized for London session)
- **Logic**: Trade in direction of trend with positive carry

### 5. Options IV Crush (Event-Driven)
- **Instrument**: VIX (volatility index)
- **Mechanism**: Implied volatility mean reversion
- **Regime**: Event-driven
- **Logic**: Buy high IV, sell on IV crush

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn talib scikit-learn
```

### Quick Start
```bash
# Run the complete system demo
python run_advanced_portfolio.py

# This will:
# 1. Generate sample market data
# 2. Run live trading simulation
# 3. Generate comprehensive reports
# 4. Create visualizations
```

## 📈 Usage Examples

### Basic Usage
```python
from advanced_portfolio_system import AdvancedPortfolioSystem

# Create system
system = AdvancedPortfolioSystem(
    initial_capital=100000,
    target_volatility=0.10,
    max_strategy_weight=0.25,
    correlation_threshold=0.5,
    fractional_kelly=0.25
)

# Run simulation
results = system.run_live_trading_simulation(
    market_data=market_data,
    start_date=start_date,
    end_date=end_date
)

# Generate report
system.generate_comprehensive_report(results, "reports")
```

### Custom Strategy Configuration
```python
from strategy_framework import StrategyFramework, SweepReclaimStrategy

# Create custom strategy
custom_config = {
    'regime_filter': 'high_vol',
    'instrument_class': 'futures',
    'mechanism': 'sweep_reclaim',
    'horizon': 'intraday',
    'session': 'NY',
    'sweep_threshold': 0.003,  # 0.3%
    'reclaim_threshold': 0.001,  # 0.1%
    'lookback_periods': 30
}

strategy = SweepReclaimStrategy(custom_config)
framework = StrategyFramework()
framework.add_strategy(strategy)
```

## 📊 Performance Metrics

### Portfolio-Level Metrics
- **Total Return**: Absolute return over the period
- **Annualized Return**: Return annualized for comparison
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return metric
- **Max Drawdown**: Maximum peak-to-trough decline
- **Calmar Ratio**: Return to max drawdown ratio
- **Win Rate**: Percentage of profitable periods

### Strategy-Level Metrics
- **Individual Returns**: Performance of each strategy
- **Correlation Analysis**: Inter-strategy correlation matrix
- **Regime Performance**: Strategy performance by market regime
- **Allocation Analysis**: Dynamic allocation over time

## 🎛️ Risk Controls

### Volatility Targeting
```python
# All strategies normalized to target volatility
target_volatility = 0.10  # 10% annualized
scaling_factor = target_vol / realized_vol
adjusted_returns = returns * scaling_factor
```

### Kelly Sizing
```python
# Optimal position sizing
kelly_fraction = mean_return / (volatility ** 2)
fractional_kelly = kelly_fraction * 0.25  # 25% of full Kelly
```

### Correlation Control
```python
# Penalize highly correlated strategies
correlation_penalty = 1 / (1 + avg_correlation)
adjusted_weight = base_weight * correlation_penalty
```

### Regime Filtering
```python
# Only trade in favorable regimes
if strategy.mechanism == "breakout" and regime.volatility_regime == "low":
    weight = 0.0  # Disable strategy
```

## 📈 Visualization & Reporting

### Generated Reports
- **Equity Curve**: Portfolio value over time with drawdowns
- **Allocation Heatmap**: Strategy weights over time
- **Correlation Analysis**: Strategy correlation matrix and trends
- **Regime Analysis**: Market regime distribution and impact
- **Strategy Performance**: Individual strategy metrics comparison

### Output Files
```
portfolio_reports/
├── equity_curve.png
├── allocation_heatmap.png
├── correlation_analysis.png
├── regime_analysis.png
├── strategy_performance.png
├── detailed_results.json
└── portfolio_report.md
```

## 🔬 Advanced Features

### Walk-Forward Backtesting
- **Out-of-Sample Testing**: Reserve data for validation
- **Parameter Re-estimation**: Update parameters periodically
- **No Look-Ahead Bias**: Use only past data for decisions

### Regime Detection
- **Trend vs Range**: ADX-based market state detection
- **Volatility Regime**: High/Normal/Low volatility classification
- **Session Detection**: Asia/London/NY session identification
- **Liquidity State**: Volume-based liquidity assessment

### Correlation Analysis
- **Rolling Correlation**: Dynamic correlation tracking
- **Kendall Tau**: Robust correlation measure
- **Correlation Surge Detection**: Automatic risk reduction

## 🚀 Performance Optimization

### Computational Efficiency
- **Vectorized Operations**: NumPy-based calculations
- **Efficient Data Structures**: Pandas for time series
- **Parallel Processing**: Multi-strategy signal generation

### Memory Management
- **Rolling Windows**: Limited historical data retention
- **Garbage Collection**: Automatic cleanup of old data
- **Efficient Storage**: Compressed data formats

## 🔧 Configuration Options

### Portfolio Manager
```python
PortfolioManager(
    target_volatility=0.10,        # 10% annualized volatility
    max_strategy_weight=0.25,      # 25% max allocation per strategy
    correlation_threshold=0.5,      # 50% correlation threshold
    fractional_kelly=0.25          # 25% of full Kelly
)
```

### Strategy Backtester
```python
StrategyBacktester(
    initial_capital=100000,        # Starting capital
    rebalance_frequency="daily",   # Rebalancing frequency
    walk_forward_periods=30,       # Walk-forward window
    min_trades_per_strategy=10     # Minimum trades for analysis
)
```

## 📚 Theory & Implementation

### Kelly Criterion
The system implements fractional Kelly sizing for optimal position sizing:

```
f* = (μ / σ²) × α
```

Where:
- `f*` = Kelly fraction
- `μ` = Expected return
- `σ²` = Variance of returns
- `α` = Fractional Kelly multiplier (0.25)

### Correlation Control
Strategies are penalized based on their correlation to other strategies:

```
w_i = w_i / (1 + ρ̄_i)
```

Where:
- `w_i` = Weight for strategy i
- `ρ̄_i` = Average correlation to other strategies

### Regime Filtering
Strategies are only active in compatible market regimes:

- **Breakout strategies**: High volatility, trending markets
- **Mean reversion**: Low volatility, ranging markets
- **Carry strategies**: All regimes with trend filter
- **Volatility strategies**: Event-driven regimes

## 🤝 Contributing

### Adding New Strategies
1. Inherit from `BaseStrategy`
2. Implement `generate_signal()` method
3. Add to strategy framework
4. Update configuration

### Custom Risk Controls
1. Extend `PortfolioManager` class
2. Implement custom risk methods
3. Integrate with allocation logic

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kelly Criterion for optimal position sizing
- Modern Portfolio Theory for diversification
- Regime-based investing principles
- Quantitative risk management best practices

---

**Note**: This system is for educational and research purposes. Always test thoroughly before deploying with real capital.
