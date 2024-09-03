import yfinance as yf
import polars as pl
from quantaforge.backtest_new import Backtest
from examples.OzempicStrategy import RevisedOzempicStrategy
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
affected_sectors = ['XLP', 'XLY', 'MO', 'PM', 'TAP', 'BUD', 'STZ', 'MCD', 'YUM', 'DNUT', 'KO', 'PEP']
hedge_symbols = ['XLV', 'LLY', 'NVO']
all_symbols = affected_sectors + hedge_symbols

# Fetch historical data
start_date = "2017-01-01"  # Start a year earlier to have enough data for trend calculation
end_date = "2023-06-26"

# Fetch data for all symbols at once
data = yf.download(all_symbols, start=start_date, end=end_date)

# Prepare the data
merged_data = pl.DataFrame({
    'timestamp': data.index.to_list(),
    'symbol': [all_symbols[0]] * len(data)  # We'll use the first symbol as a placeholder
})

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    merged_data = merged_data.with_columns(
        pl.Series(name=col.lower(), values=data[col][all_symbols[0]].to_list())
    )

# Add data for all symbols
for symbol in all_symbols:
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        merged_data = merged_data.with_columns(
            pl.Series(name=f'{symbol}_{col.lower()}', values=data[col][symbol].to_list())
        )

# Sort by timestamp
merged_data = merged_data.sort('timestamp')

# Add lagged close prices for trend calculation
for symbol in all_symbols:
    merged_data = merged_data.with_columns([
        pl.col(f'{symbol}_close').shift(252).alias(f'{symbol}_close_252'),
        pl.col(f'{symbol}_close').shift(22).alias(f'{symbol}_close_22')
    ])

# Remove rows with NaN values (first year of data)
merged_data = merged_data.drop_nulls()

# Create strategy and backtest instances
strategy = RevisedOzempicStrategy(affected_sectors, hedge_symbols)
backtest = Backtest("Revised Ozempic Strategy Backtest", strategy, initial_capital=1000000)  # Start with $1M

# Set data and run backtest
backtest.set_data(merged_data)
results = backtest.run()

# Print results and analyze
print("Backtest Results:")
for key, value in results.items():
    print(f"{key}: {value}")

# Analyze drawdowns
equity_curve = np.array(backtest.equity_curve)
peak = np.maximum.accumulate(equity_curve)
drawdowns = (equity_curve - peak) / peak
max_drawdown = np.min(drawdowns)
print(f"\nStrategy Max Drawdown: {max_drawdown:.2%}")

# Compare to a benchmark (e.g., S&P 500)
spy_data = yf.download('SPY', start=start_date, end=end_date)
spy_return = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0]) - 1
print(f"\nS&P 500 Return: {spy_return:.2%}")

spy_peak = np.maximum.accumulate(spy_data['Close'])
spy_drawdowns = (spy_data['Close'] - spy_peak) / spy_peak
spy_max_drawdown = np.min(spy_drawdowns)
print(f"S&P 500 Max Drawdown: {spy_max_drawdown:.2%}")

# Plot equity curves
plt.figure(figsize=(12, 6))
plt.plot(merged_data['timestamp'], backtest.equity_curve, label='Ozempic Strategy')
plt.plot(spy_data.index, spy_data['Close'] / spy_data['Close'].iloc[0] * 1000000, label='S&P 500')
plt.title('Ozempic Strategy vs S&P 500')
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