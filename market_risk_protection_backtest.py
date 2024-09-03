import yfinance as yf
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from quantaforge.backtest_new import Backtest
from examples.MarketRiskProtectionStrategy import MarketRiskProtectionStrategy

# Include the updated MarketRiskProtectionStrategy class here

# Fetch historical data
spy_data = yf.download("SPY", start="2000-01-01", end="2023-06-26")
spy_data = spy_data.reset_index()

# Convert to Polars DataFrame and rename columns
spy_data = pl.DataFrame(spy_data)
spy_data = spy_data.rename({
    'Date': 'timestamp',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
})
spy_data = spy_data.with_columns([pl.lit('SPY').alias('symbol')])

# Ensure all required columns are present and in lowercase
required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
spy_data.columns = [col.lower() for col in spy_data.columns]

# Create strategy and backtest instances
strategy = MarketRiskProtectionStrategy()
backtest = Backtest("Market Risk Protection Backtest", strategy, initial_capital=100000)

# Set data and run backtest
backtest.set_data(spy_data)
results = backtest.run()

# Print results
print("Backtest Results:")
for key, value in results.items():
    print(f"{key}: {value}")

# Compare to buy and hold strategy
spy_return = (spy_data['close'].tail(1).item() / spy_data['close'].head(1).item()) - 1
print(f"\nBuy and Hold SPY Return: {spy_return:.2%}")

# Analyze drawdowns
equity_curve = np.array(backtest.equity_curve)
peak = np.maximum.accumulate(equity_curve)
drawdowns = (equity_curve - peak) / peak
max_drawdown = np.min(drawdowns)
print(f"\nStrategy Max Drawdown: {max_drawdown:.2%}")

spy_equity = (spy_data['close'] / spy_data['close'].head(1).item()) * 100000
spy_peak = np.maximum.accumulate(spy_equity.to_numpy())
spy_drawdowns = (spy_equity.to_numpy() - spy_peak) / spy_peak
spy_max_drawdown = np.min(spy_drawdowns)
print(f"SPY Max Drawdown: {spy_max_drawdown:.2%}")

# Plot equity curves
plt.figure(figsize=(12, 6))
plt.plot(spy_data['timestamp'], backtest.equity_curve, label='Strategy')
plt.plot(spy_data['timestamp'], spy_equity, label='Buy and Hold SPY')
plt.title('Strategy vs Buy and Hold SPY')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Run Monte Carlo simulation
mc_results = backtest.monte_carlo_simulation()
print("\nMonte Carlo Simulation Results:")
print(f"5th percentile: {np.percentile(mc_results[-1], 5)}")
print(f"50th percentile: {np.percentile(mc_results[-1], 50)}")
print(f"95th percentile: {np.percentile(mc_results[-1], 95)}")