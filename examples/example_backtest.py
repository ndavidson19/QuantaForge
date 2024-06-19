import pandas as pd
from quantaforge.strategy import MovingAverageStrategy
from quantaforge.backtest import Backtest

# Create sample data
data = {
    'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
    'close': [150, 152, 148, 151]
}
df = pd.DataFrame(data)

# Initialize strategy
strategy = MovingAverageStrategy(window=2)

# Run backtest
backtest = Backtest(df, strategy)
final_value = backtest.run()

print("Final Portfolio Value:", final_value)
