import polars as pl
import logging
import numpy as np
from datetime import datetime
from quantaforge.strategy import Strategy
from quantaforge.backtest import Backtest
from quantaforge.indicators import SMA, RSI
from quantaforge.signals import CrossOverSignal, ThresholdSignal
from quantaforge.generics import logic_condition, LogicType

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Create a new strategy
strategy = Strategy("Volatility Breakout Strategy")

# Add indicators
strategy.add_indicator("SMA_20", SMA(window=20))
strategy.add_indicator("RSI_14", RSI(window=14))

# Add signals
strategy.add_signal("SMA_Crossover", CrossOverSignal("close", "SMA_20"))
strategy.add_signal("RSI_Threshold", ThresholdSignal("RSI_14", buy_threshold=30, sell_threshold=70))

# If you want to change the logic type:
# 1. For entry conditions, use logic_condition(LogicType.OR)
# 2. For exit conditions, use logic_condition(LogicType.AND)
# By default, the logic type is LogicType.OR for both entry and exit conditions.
# This example means that the strategy will enter a trade if any of the entry conditions are met,
# and it will exit a trade if all of the exit conditions are met.
# You can change this behavior by using the logic_condition decorator as shown below.
strategy.check_entry_conditions = logic_condition(LogicType.OR)(strategy.check_entry_conditions)
strategy.check_exit_conditions = logic_condition(LogicType.AND)(strategy.check_exit_conditions)

# Create a backtest
backtest = Backtest(name='VolatilityBreakoutBacktest', strategy=strategy, initial_capital=100000)

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

logger.info("Indicator values for the last 10 days:")
for row in backtest.data.tail(10).iter_rows(named=True):
    logger.info(row)
# Print some results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
print(f"Profit Factor: {results['profit_factor']:.2f}")