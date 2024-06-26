import polars as pl
from typing import Dict, Any
from quantaforge.generics import Signal

class CrossOverSignal(Signal):
    def __init__(self, fast_indicator: str, slow_indicator: str):
        super().__init__(name=f"Crossover_{fast_indicator}_{slow_indicator}")
        self.fast_indicator = fast_indicator
        self.slow_indicator = slow_indicator

    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        crossover = (
            (data[self.fast_indicator] > data[self.slow_indicator]) & 
            (data[self.fast_indicator].shift() <= data[self.slow_indicator].shift())
        ).cast(pl.Int8)
        
        crossunder = (
            (data[self.fast_indicator] < data[self.slow_indicator]) & 
            (data[self.fast_indicator].shift() >= data[self.slow_indicator].shift())
        ).cast(pl.Int8)
        
        return pl.DataFrame({
            f"{self.name}_buy": crossover,
            f"{self.name}_sell": crossunder
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'fast_indicator': self.fast_indicator,
            'slow_indicator': self.slow_indicator
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'CrossOverSignal':
        return cls(fast_indicator=config['fast_indicator'], slow_indicator=config['slow_indicator'])

class ThresholdSignal(Signal):
    def __init__(self, indicator: str, buy_threshold: float, sell_threshold: float):
        super().__init__(name=f"Threshold_{indicator}")
        self.indicator = indicator
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        buy_signal = (data[self.indicator] >= self.buy_threshold).cast(pl.Int8)
        sell_signal = (data[self.indicator] <= self.sell_threshold).cast(pl.Int8)
        
        return pl.DataFrame({
            f"{self.name}_buy": buy_signal,
            f"{self.name}_sell": sell_signal
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'indicator': self.indicator,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'ThresholdSignal':
        return cls(
            indicator=config['indicator'],
            buy_threshold=config['buy_threshold'],
            sell_threshold=config['sell_threshold']
        )

class MACDSignal(Signal):
    def __init__(self, macd_indicator: str):
        super().__init__(name=f"MACD_Signal_{macd_indicator}")
        self.macd_indicator = macd_indicator

    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        macd_line = data[f"{self.macd_indicator}_line"]
        signal_line = data[f"{self.macd_indicator}_signal"]

        buy_signal = ((macd_line > signal_line) & (macd_line.shift() <= signal_line.shift())).cast(pl.Int8)
        sell_signal = ((macd_line < signal_line) & (macd_line.shift() >= signal_line.shift())).cast(pl.Int8)

        return pl.DataFrame({
            f"{self.name}_buy": buy_signal,
            f"{self.name}_sell": sell_signal
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'macd_indicator': self.macd_indicator
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MACDSignal':
        return cls(macd_indicator=config['macd_indicator'])

class BollingerBandsSignal(Signal):
    def __init__(self, bb_indicator: str, column: str = 'close'):
        super().__init__(name=f"BB_Signal_{bb_indicator}")
        self.bb_indicator = bb_indicator
        self.column = column

    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        price = data[self.column]
        upper = data[f"{self.bb_indicator}_upper"]
        lower = data[f"{self.bb_indicator}_lower"]

        buy_signal = (price <= lower).cast(pl.Int8)
        sell_signal = (price >= upper).cast(pl.Int8)

        return pl.DataFrame({
            f"{self.name}_buy": buy_signal,
            f"{self.name}_sell": sell_signal
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'bb_indicator': self.bb_indicator,
            'column': self.column
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'BollingerBandsSignal':
        return cls(bb_indicator=config['bb_indicator'], column=config.get('column', 'close'))