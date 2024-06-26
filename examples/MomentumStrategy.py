import logging
import polars as pl
from quantaforge.strategy import Strategy
from quantaforge.generics import Indicator, Signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RSIIndicator(Indicator):
    def __init__(self, window: int = 14):
        super().__init__(f"RSI_{window}")
        self.window = window

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        close = data['close']
        delta = close.diff()
        gain = delta.clip(lower_bound=0).rolling_mean(window_size=self.window)
        loss = -delta.clip(upper_bound=0).rolling_mean(window_size=self.window)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return data.with_columns(pl.Series(name=self.name, values=rsi).fill_null(strategy='forward'))  

class MACDIndicator(Indicator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        super().__init__(f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        close = data['close']
        fast_ema = close.ewm_mean(span=self.fast_period, ignore_nulls=True)
        slow_ema = close.ewm_mean(span=self.slow_period, ignore_nulls=True)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=self.signal_period, ignore_nulls=True)
        histogram = macd_line - signal_line
        
        return data.with_columns([
            pl.Series(name=f"{self.name}_line", values=macd_line),
            pl.Series(name=f"{self.name}_signal", values=signal_line),
            pl.Series(name=f"{self.name}_histogram", values=histogram)
        ])

class MomentumStrategy(Strategy):
    def __init__(self, symbol: str, rsi_window: int = 14, rsi_lower: int = 30, rsi_upper: int = 70,
                 macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9,
                 trailing_stop_pct: float = 0.05):
        super().__init__("Momentum Strategy")
        self.symbol = symbol
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.trailing_stop_pct = trailing_stop_pct
        
        self.add_indicator(f"RSI_{rsi_window}", RSIIndicator(rsi_window))
        self.add_indicator(f"MACD_{macd_fast}_{macd_slow}_{macd_signal}", 
                           MACDIndicator(macd_fast, macd_slow, macd_signal))

    def generate_signals(self, data: pl.DataFrame) -> dict:
        rsi = data[f"RSI_{14}"]
        macd_line = data[f"MACD_{12}_{26}_{9}_line"]
        macd_signal = data[f"MACD_{12}_{26}_{9}_signal"]

        rsi_buy = (rsi.shift(1) <= self.rsi_lower) & (rsi > self.rsi_lower)
        rsi_sell = (rsi.shift(1) >= self.rsi_upper) & (rsi < self.rsi_upper)
        macd_buy = (macd_line.shift(1) <= macd_signal.shift(1)) & (macd_line > macd_signal)
        macd_sell = (macd_line.shift(1) >= macd_signal.shift(1)) & (macd_line < macd_signal)


        logging.info(f"RSI Buy Signals: {rsi_buy.sum()}")
        logging.info(f"MACD Buy Signals: {macd_buy.sum()}")
        logging.info(f"RSI Sell Signals: {rsi_sell.sum()}")
        logging.info(f"MACD Sell Signals: {macd_sell.sum()}")


        buy_signal = (rsi_buy & macd_buy).cast(pl.Float32)
        sell_signal = (rsi_sell & macd_sell).cast(pl.Float32)

        signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_buy": buy_signal,
                f"{self.symbol}_sell": sell_signal,
                f"{self.symbol}_rsi": rsi,
                f"{self.symbol}_macd_line": macd_line,
                f"{self.symbol}_macd_signal": macd_signal
            })
        }

        logging.info(f"Generated signals: {signals}")
        return signals

    def manage_risk(self, data: pl.DataFrame) -> dict:
        close = data['close']
        
        # Calculate trailing stop levels
        highest_high = close.cum_max()
        trailing_stop = highest_high * (1 - self.trailing_stop_pct)
        
        # Generate trailing stop signal
        trailing_stop_signal = (close <= trailing_stop).cast(pl.Float32)

        risk_signals = {
            self.symbol: pl.DataFrame({
                f"{self.symbol}_trailing_stop": trailing_stop_signal,
                f"{self.symbol}_trailing_stop_level": trailing_stop
            })
        }

        logging.info(f"Risk management signals: {risk_signals}")
        return risk_signals