import unittest
import pandas as pd
import numpy as np
import polars as pl
import logging
from examples.VolatiltyBreakoutStrategy import VolatilityBreakoutStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestVolatilityBreakoutStrategy(unittest.TestCase):
    def setUp(self):
        self.symbol = "AAPL"
        self.strategy = VolatilityBreakoutStrategy(self.symbol)
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        closes = np.random.randn(len(dates)).cumsum() + 100
        highs = closes + np.abs(np.random.randn(len(dates)))
        lows = closes - np.abs(np.random.randn(len(dates)))
        
        mock_data = pl.DataFrame({
            'date': dates,
            'open': closes,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': np.random.randint(1000000, 10000000, size=len(dates))
        })
        
        return mock_data

    def test_indicators(self):
        logging.info("Testing indicators calculation")
        data_with_indicators = self.strategy.indicators['Volatility_20'].calculate(self.mock_data)
        data_with_indicators = self.strategy.indicators['Breakout_20'].calculate(data_with_indicators)
        
        self.assertIn('Volatility_20', data_with_indicators.columns)
        self.assertIn('Breakout_20_up', data_with_indicators.columns)
        self.assertIn('Breakout_20_down', data_with_indicators.columns)
        
        logging.info(f"Calculated indicators: {data_with_indicators.columns}")

    def test_generate_signals(self):
        logging.info("Testing generate_signals method")
        # Calculate indicators first
        data_with_indicators = self.strategy.indicators['Volatility_20'].calculate(self.mock_data)
        data_with_indicators = self.strategy.indicators['Breakout_20'].calculate(data_with_indicators)
        
        signals = self.strategy.generate_signals(data_with_indicators)
        
        self.assertIsInstance(signals, dict)
        self.assertIn(self.symbol, signals)
        
        symbol_signals = signals[self.symbol]
        self.assertIn(f"{self.symbol}_buy", symbol_signals.columns)
        self.assertIn(f"{self.symbol}_sell", symbol_signals.columns)
        self.assertIn(f"{self.symbol}_volatility", symbol_signals.columns)
        
        logging.info(f"Generated signals columns: {symbol_signals.columns}")
        logging.info(f"Buy signals sum: {symbol_signals[f'{self.symbol}_buy'].sum()}")
        logging.info(f"Sell signals sum: {symbol_signals[f'{self.symbol}_sell'].sum()}")

    def test_manage_risk(self):
        logging.info("Testing manage_risk method")
        # Calculate indicators first
        data_with_indicators = self.strategy.indicators['Volatility_20'].calculate(self.mock_data)
        
        risk_signals = self.strategy.manage_risk(data_with_indicators)
        
        self.assertIsInstance(risk_signals, dict)
        self.assertIn(self.symbol, risk_signals)
        
        symbol_risk_signals = risk_signals[self.symbol]
        self.assertIn(f"{self.symbol}_position_size", symbol_risk_signals.columns)
        self.assertIn(f"{self.symbol}_volatility", symbol_risk_signals.columns)
        
        logging.info(f"Risk management signals columns: {symbol_risk_signals.columns}")
        logging.info(f"Average position size: {symbol_risk_signals[f'{self.symbol}_position_size'].mean()}")

if __name__ == '__main__':
    unittest.main()