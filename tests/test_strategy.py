import unittest
import polars as pl
from quantaforge.strategy import Strategy
from quantaforge.indicators import SMA
from quantaforge.signals import CrossOverSignal

class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'open': [149.0, 151.0, 147.0, 150.0],
            'high': [151.0, 153.0, 149.0, 152.0],
            'low': [148.0, 150.0, 146.0, 149.0],
            'close': [150.0, 152.0, 148.0, 151.0],
            'volume': [1000000, 1100000, 900000, 1050000]
        })

    def test_add_indicator(self):
        strategy = Strategy("TestStrategy")
        strategy.add_indicator("SMA_2", SMA(2))
        self.assertIn("SMA_2", strategy.indicators)

    def test_add_signal(self):
        strategy = Strategy("TestStrategy")
        strategy.add_signal("CrossOver", CrossOverSignal("close", "SMA_2"))
        self.assertIn("CrossOver", strategy.signals)

    def test_generate_signals(self):
        strategy = Strategy("TestStrategy")
        strategy.add_indicator("SMA_2", SMA(2))
        strategy.add_signal("CrossOver", CrossOverSignal("close", "SMA_2"))
        
        signals = strategy._generate_signals(self.data)
        
        self.assertIsInstance(signals, pl.DataFrame)
        self.assertIn('Crossover_close_SMA_2_buy', signals.columns)
        self.assertIn('Crossover_close_SMA_2_sell', signals.columns)

if __name__ == '__main__':
    unittest.main()