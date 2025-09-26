# BTC Trading Strategy Comparison Report

## Strategy Parameters
- **Time Period**: 2528 data points
- **Initial Capital**: $10,000
- **Sentiment Alpha**: 0.4
- **Lookback Window**: 24 hours

## Performance Comparison

| Metric | Original Strategy | Sentiment-Enhanced | Improvement |
|--------|------------------|-------------------|-------------|
| **Total Trades** | 268 | 256 | -12 |
| **Win Rate** | 27.6% | 27.3% | -0.3% |
| **Total Return** | 1.18% | 1.02% | -0.16% |
| **Max Drawdown** | 0.69% | 0.69% | -0.00% |
| **Final Equity** | $10,118 | $10,102 | $-16 |

## Analysis

### Sentiment Enhancement Impact
The sentiment-enhanced strategy incorporates news sentiment analysis within ±30 minutes of trading decisions. The sentiment multiplier formula is:

**Enhanced Signal = Base Signal × (1 + α × Sentiment)**

Where:
- α = 0.4 (tunable parameter)
- Sentiment = normalized sentiment score [-1, 1]

### Key Findings
- **Sentiment Impact**: Negative impact on overall returns
- **Trade Frequency**: Decreased number of trades
- **Risk Management**: Improved drawdown control

### Recommendation
The original strategy performs better. Consider adjusting sentiment parameters or improving sentiment analysis.

---
*Generated on 2025-09-26 14:59:10*
