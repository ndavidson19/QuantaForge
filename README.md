# Quantaforge

Quantaforge is a comprehensive library for creating and backtesting trading strategies.

## Features

- Define custom trading strategies using various indicators, signals, and risk management techniques.
- Backtest strategies with historical data.
- Optimize strategy parameters.
- Generate detailed performance reports.

## Installation
```bash
pip install quantaforge
```

## Usage
```python
import quantaforge as qf

# Create a new strategy
strategy = qf.Strategy()

# Add a new indicator
strategy.add_indicator(qf.Indicator('SMA', 20))

# Add a new signal
strategy.add_signal(qf.Signal('CrossOver', 'SMA', 'Close'))

# Add a new position
strategy.add_position(qf.Position('Long', 1))

# Add a new risk management
strategy.add_risk_management(qf.RiskManagement('StopLoss', 0.02))

# Add a new portfolio
strategy.add_portfolio(qf.Portfolio('SimplePortfolio', 1000))

# Add a new backtest
strategy.add_backtest(qf.Backtest('SimpleBacktest', 'SimplePortfolio'))

# Add a new optimizer
strategy.add_optimizer(qf.Optimizer('SimpleOptimizer', 'SimpleBacktest'))

# Add a new report
strategy.add_report(qf.Report('SimpleReport', 'SimpleOptimizer'))

# Run the strategy
strategy.run()
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
