import unittest
import polars as pl
from quantaforge.strategy import MovingAverageStrategy, MomentumStrategy
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.data = {
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'close': [150, 152, 148, 151]
        }
        self.df = pl.DataFrame(self.data)

    def test_moving_average_strategy(self):
        strategy = MovingAverageStrategy(window=2)
        signals = strategy.generate_signals(self.df)
        logging.debug(f"Test moving average strategy signals: {signals}")
        expected = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'close': [None, 151.0, 150.0, 149.5]
        })
        logging.debug(f"Expected moving average strategy signals: {expected}")
        self.assertTrue(signals.equals(expected))

    def test_momentum_strategy(self):
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(self.df)
        logging.debug(f"Test momentum strategy signals: {signals}")
        expected = pl.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'close': [None, 2.0, -4.0, 3.0]
        })
        logging.debug(f"Expected momentum strategy signals: {expected}")
        self.assertTrue(signals.equals(expected))


if __name__ == '__main__':
    unittest.main()
