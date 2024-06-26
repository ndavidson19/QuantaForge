import logging
import polars as pl
from quantaforge.strategy import Strategy
from quantaforge.generics import Indicator, Signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VolatilityIndicator(Indicator):
    def __init__(self, window: int = 20):
        super().__init__(f"Volatility_{window}")
        self.window = window

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        returns = data['close'].pct_change()
        volatility = returns.rolling_std(window_size=self.window) * (252 ** 0.5)  # Annualized volatility
        return data.with_columns(pl.Series(name=self.name, values=volatility))

class BreakoutIndicator(Indicator):
    def __init__(self, window: int = 20):
        super().__init__(f"Breakout_{window}")
        self.window = window

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        high = data['high'].rolling_max(window_size=self.window)
        low = data['low'].rolling_min(window_size=self.window)
        breakout_up = (data['close'] > high.shift(1)).cast(pl.Int32)
        breakout_down = (data['close'] < low.shift(1)).cast(pl.Int32)
        return data.with_columns([
            pl.Series(name=f"{self.name}_up", values=breakout_up),
            pl.Series(name=f"{self.name}_down", values=breakout_down)
        ])

class VolatilityBreakoutStrategy(Strategy):
    def __init__(self, symbol: str, volatility_window: int = 20, breakout_window: int = 20, 
                 volatility_threshold: float = 0.15, max_position_size: float = 0.1):
        super().__init__("Volatility Breakout Strategy")
        self.symbol = symbol
        self.volatility_threshold = volatility_threshold
        self.max_position_size = max_position_size
        
        self.add_indicator(f"Volatility_{volatility_window}", VolatilityIndicator(volatility_window))
        self.add_indicator(f"Breakout_{breakout_window}", BreakoutIndicator(breakout_window))

    def generate_signals(self, data: pl.DataFrame) -> dict:
        volatility = data[f"Volatility_{self.indicators['Volatility_20'].window}"]
        breakout_up = data[f"Breakout_{self.indicators['Breakout_20'].window}_up"]
        breakout_down = data[f"Breakout_{self.indicators['Breakout_20'].window}_down"]

        low_volatility = (volatility < self.volatility_threshold).cast(pl.Int32)

        buy_signal = (low_volatility.shift(1) & breakout_up).cast(pl.Float32)
        sell_signal = (low_volatility.shift(1) & breakout_down).cast(pl.Float32)

        signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_buy": buy_signal,
                f"{self.symbol}_sell": sell_signal,
                f"{self.symbol}_volatility": volatility
            })
        }

        logging.info(f"Generated signals: {signals}")
        return signals

    def manage_risk(self, data: pl.DataFrame) -> dict:
        volatility = data[f"Volatility_{self.indicators['Volatility_20'].window}"]
        current_volatility = volatility.tail(1).item()
        
        # Adjust position size based on volatility
        if current_volatility > self.volatility_threshold:
            position_size = self.max_position_size * (self.volatility_threshold / current_volatility)
        else:
            position_size = self.max_position_size

        risk_signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_position_size": pl.Series([position_size] * len(data)),
                f"{self.symbol}_volatility": volatility
            })
        }

        logging.info(f"Risk management signals: {risk_signals}")
        return risk_signals