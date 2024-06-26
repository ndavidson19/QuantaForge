import logging
import polars as pl
from quantaforge.strategy import Strategy
from quantaforge.generics import Indicator, Signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BollingerBandsIndicator(Indicator):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(f"BollingerBands_{window}_{num_std}")
        self.window = window
        self.num_std = num_std

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        close = data['close']
        sma = close.rolling_mean(window_size=self.window)
        std = close.rolling_std(window_size=self.window)
        upper = sma + (std * self.num_std)
        lower = sma - (std * self.num_std)
        
        return data.with_columns([
            pl.Series(name=f"{self.name}_sma", values=sma),
            pl.Series(name=f"{self.name}_upper", values=upper),
            pl.Series(name=f"{self.name}_lower", values=lower)
        ])

class MeanReversionStrategy(Strategy):
    def __init__(self, symbol: str, bb_window: int = 20, bb_std: float = 2.0, 
                 stop_loss: float = 0.02, take_profit: float = 0.04):
        super().__init__("Mean Reversion Strategy")
        self.symbol = symbol
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        self.add_indicator(f"BB_{bb_window}_{bb_std}", BollingerBandsIndicator(bb_window, bb_std))

    def generate_signals(self, data: pl.DataFrame) -> dict:
        close = data['close']
        bb = self.indicators[f"BB_{20}_{2.0}"]
        upper = data[f"{bb.name}_upper"]
        lower = data[f"{bb.name}_lower"]

        buy_signal = (close <= lower).cast(pl.Float32)
        sell_signal = (close >= upper).cast(pl.Float32)

        signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_buy": buy_signal,
                f"{self.symbol}_sell": sell_signal,
                f"{self.symbol}_close": close,
                f"{self.symbol}_upper": upper,
                f"{self.symbol}_lower": lower
            })
        }

        logging.info(f"Generated signals: {signals}")
        return signals

    def manage_risk(self, data: pl.DataFrame) -> dict:
        close = data['close']
        bb = self.indicators[f"BB_{20}_{2.0}"]
        sma = data[f"{bb.name}_sma"]

        # Calculate distances from current price to stop-loss and take-profit levels
        stop_loss_distance = close * self.stop_loss
        take_profit_distance = close * self.take_profit

        # Generate stop-loss and take-profit signals
        stop_loss_signal = ((close < sma) & (sma - close > stop_loss_distance)).cast(pl.Float32)
        take_profit_signal = ((close > sma) & (close - sma > take_profit_distance)).cast(pl.Float32)

        risk_signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_stop_loss": stop_loss_signal,
                f"{self.symbol}_take_profit": take_profit_signal,
                f"{self.symbol}_sma": sma
            })
        }

        logging.info(f"Risk management signals: {risk_signals}")
        return risk_signals