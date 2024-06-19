import pandas as pd
from quanta_forge.strategy import MovingAverageStrategy, MomentumStrategy

# Create sample data
data = {
    'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
    'close': [150, 152, 148, 151]
}
df = pd.DataFrame(data)

# Initialize strategies
moving_average_strategy = MovingAverageStrategy(window=2)
momentum_strategy = MomentumStrategy()

# Generate signals
ma_signals = moving_average_strategy.generate_signals(df)
momentum_signals = momentum_strategy.generate_signals(df)

print("Moving Average Signals:\n", ma_signals)
print("Momentum Signals:\n", momentum_signals)
