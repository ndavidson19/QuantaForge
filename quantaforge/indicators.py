import polars as pl
from typing import Dict, Any
from quantaforge.generics import Indicator

class SMA(Indicator):
    def __init__(self, window: int, column: str = 'close'):
        super().__init__(name=f"SMA_{window}")
        self.window = window
        self.column = column

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            pl.col(self.column).rolling_mean(self.window).alias(self.name)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'window': self.window,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SMA':
        return cls(window=config['window'], column=config.get('column', 'close'))

class EMA(Indicator):
    def __init__(self, span: int, column: str = 'close'):
        super().__init__(name=f"EMA_{span}")
        self.span = span
        self.column = column

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns(
            pl.col(self.column).ewm_mean(span=self.span, ignore_nulls=True, adjust=False).alias(self.name)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'span': self.span,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'EMA':
        return cls(span=config['span'], column=config.get('column', 'close'))

class RSI(Indicator):
    def __init__(self, window: int = 14, column: str = 'close'):
        super().__init__(name=f"RSI_{window}")
        self.window = window
        self.column = column

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        delta = data[self.column].diff()
        gain = delta.clip(lower_bound=0).rolling_mean(self.window)
        loss = -delta.clip(upper_bound=0).rolling_mean(self.window)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        rsi = rsi.fill_null(None, strategy='forward')

        return data.with_columns(rsi.alias(self.name))

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'window': self.window,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RSI':
        return cls(window=config['window'], column=config.get('column', 'close'))

class MACD(Indicator):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, column: str = 'close'):
        super().__init__(name=f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.column = column

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        fast_ema = data[self.column].ewm_mean(span=self.fast_period, ignore_nulls=True, adjust=False)
        slow_ema = data[self.column].ewm_mean(span=self.slow_period, ignore_nulls=True, adjust=False)
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm_mean(span=self.signal_period, ignore_nulls=True)
        histogram = macd_line - signal_line

        return data.with_columns([
            macd_line.alias(f"{self.name}_line"),
            signal_line.alias(f"{self.name}_signal"),
            histogram.alias(f"{self.name}_histogram")
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MACD':
        return cls(
            fast_period=config['fast_period'],
            slow_period=config['slow_period'],
            signal_period=config['signal_period'],
            column=config.get('column', 'close')
        )

class BollingerBands(Indicator):
    def __init__(self, window: int = 20, num_std: float = 2, column: str = 'close'):
        super().__init__(name=f"BB_{window}_{num_std}")
        self.window = window
        self.num_std = num_std
        self.column = column

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        sma = data[self.column].rolling_mean(self.window)
        std = data[self.column].rolling_std(self.window)
        upper = sma + (std * self.num_std)
        lower = sma - (std * self.num_std)

        return data.with_columns([
            sma.alias(f"{self.name}_middle"),
            upper.alias(f"{self.name}_upper"),
            lower.alias(f"{self.name}_lower")
        ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'window': self.window,
            'num_std': self.num_std,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BollingerBands':
        return cls(
            window=config['window'],
            num_std=config['num_std'],
            column=config.get('column', 'close')
        )