# Quantaforge

Quantaforge is a simple yet powerful library for creating and testing trading strategies. It's designed to be easy to understand, even for those new to Python.

## Installation

```bash
pip install quantaforge
```

## Usage

Here's an example of how to create a Volatility Breakout Strategy using Quantaforge:

```python
import quantaforge as qf
import polars as pl
import numpy as np
from datetime import datetime

# Create a new strategy
strategy = qf.Strategy("Volatility Breakout Strategy")

# Add indicators
strategy.add_indicator("SMA_20", SMA(window=20))
strategy.add_indicator("RSI_14", RSI(window=14))

# Add signals
strategy.add_signal("SMA_Crossover", qf.CrossOverSignal("close", "SMA_20"))
strategy.add_signal("RSI_Threshold", qf.ThresholdSignal("RSI_14", buy_threshold=30, sell_threshold=70))

# If you want to change the logic type:
# 1. For entry conditions, use logic_condition(LogicType.OR)
# 2. For exit conditions, use logic_condition(LogicType.AND)
# By default, the logic type is LogicType.OR for both entry and exit conditions.
# This example means that the strategy will enter a trade if any of the entry conditions are met,
# and it will exit a trade if all of the exit conditions are met.
# You can change this behavior by using the logic_condition decorator as shown below.
strategy.check_entry_conditions = qf.logic_condition(qf.LogicType.OR)(strategy.check_entry_conditions)
strategy.check_exit_conditions = qf.logic_condition(qf.LogicType.AND)(strategy.check_exit_conditions)

# Create a backtest
backtest = qf.Backtest(name='VolatilityBreakoutBacktest', strategy=strategy, initial_capital=100000)

# Generate sample data
np.random.seed(42)
dates = pl.datetime_range(
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
    interval='1d',
    eager=True
)

# Creating a synthetic trend for prices
prices = np.linspace(100, 110, len(dates))  # Prices increasing linearly
prices[:len(dates)//2] *= 0.95  # Downtrend for the first half
prices[len(dates)//2:] *= 1.05  # Uptrend for the second half

# Adding some noise to prices
prices += np.random.normal(0, 0.5, len(dates))

data = pl.DataFrame({
    'timestamp': dates,
    'symbol': ['AAPL'] * len(dates),
    'open': prices,
    'high': prices * 1.02,
    'low': prices * 0.98,
    'close': prices * 1.01,
    'volume': np.random.randint(1000000, 2000000, size=len(dates))
})

# Set the data for the backtest
backtest.set_data(data)

# Run the backtest
results = backtest.run()

# Print some results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

This example demonstrates how to:
1. Create a strategy
2. Add pre-made indicators and signals
3. Define entry and exit conditions
4. Add risk management rules
5. Set up a portfolio and backtest
6. Run the backtest and generate a report

You can easily modify this example to create your own custom strategies!

## Features

- Easy-to-understand API for strategy creation
- Pre-made indicators and signals for quick strategy development
- Support for custom indicators and signals
- Flexible entry and exit conditions
- Built-in risk management tools
- Backtesting capabilities
- Performance reporting

## Custom Components

While Quantaforge provides many pre-made components, you can also create custom indicators and signals:

```python
from quantaforge.indicators import Indicator
from quantaforge.signals import Signal

class MyCustomIndicator(Indicator):
    def __init__(self, param1, param2):
        super().__init__(name=f"MyIndicator_{param1}_{param2}")
        self.param1 = param1
        self.param2 = param2

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Your custom calculation logic here
        pass

class MyCustomSignal(Signal):
    def __init__(self, param1, param2):
        super().__init__(name=f"MySignal_{param1}_{param2}")
        self.param1 = param1
        self.param2 = param2

    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Your custom signal generation logic here
        pass

# Usage
strategy.add_indicator("CustomInd", MyCustomIndicator(param1=10, param2=20))
strategy.add_signal("CustomSig", MyCustomSignal(param1=5, param2=15))
```

## License

[MIT](https://choosealicense.com/licenses/mit/)