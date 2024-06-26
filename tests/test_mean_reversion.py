import unittest
import pandas as pd
import numpy as np
import polars as pl
import logging
from examples.MeanReversionStrategy import MeanReversionStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestMeanReversionStrategy(unittest.TestCase):
    def setUp(self):
        self.symbol = "AAPL"
        self.strategy = MeanReversionStrategy(self.symbol)
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate a mean-reverting price series
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        price = price + 10 * np.sin(np.arange(len(dates)) / 20)  # Add cyclical component
        
        mock_data = pl.DataFrame({
            'date': dates,
            'open': price,
            'high': price * 1.02,
            'low': price * 0.98,
            'close': price,
            'volume': np.random.randint(1000000, 10000000, size=len(dates))
        })
        
        return mock_data

    def test_indicators(self):
        logging.info("Testing indicators calculation")
        bb = self.strategy.indicators[f"BB_{20}_{2.0}"]
        data_with_indicators = bb.calculate(self.mock_data)
        
        self.assertIn(f"{bb.name}_sma", data_with_indicators.columns)
        self.assertIn(f"{bb.name}_upper", data_with_indicators.columns)
        self.assertIn(f"{bb.name}_lower", data_with_indicators.columns)
        
        logging.info(f"Calculated indicators: {data_with_indicators.columns}")

    def test_generate_signals(self):
        logging.info("Testing generate_signals method")
        bb = self.strategy.indicators[f"BB_{20}_{2.0}"]
        data_with_indicators = bb.calculate(self.mock_data)
        
        signals = self.strategy.generate_signals(data_with_indicators)
        
        self.assertIsInstance(signals, dict)
        self.assertIn(self.symbol, signals)
        
        symbol_signals = signals[self.symbol]
        self.assertIn(f"{self.symbol}_buy", symbol_signals.columns)
        self.assertIn(f"{self.symbol}_sell", symbol_signals.columns)
        
        logging.info(f"Generated signals columns: {symbol_signals.columns}")
        logging.info(f"Buy signals sum: {symbol_signals[f'{self.symbol}_buy'].sum()}")
        logging.info(f"Sell signals sum: {symbol_signals[f'{self.symbol}_sell'].sum()}")

    def test_manage_risk(self):
        logging.info("Testing manage_risk method")
        bb = self.strategy.indicators[f"BB_{20}_{2.0}"]
        data_with_indicators = bb.calculate(self.mock_data)
        
        risk_signals = self.strategy.manage_risk(data_with_indicators)
        
        self.assertIsInstance(risk_signals, dict)
        self.assertIn(self.symbol, risk_signals)
        
        symbol_risk_signals = risk_signals[self.symbol]
        self.assertIn(f"{self.symbol}_stop_loss", symbol_risk_signals.columns)
        self.assertIn(f"{self.symbol}_take_profit", symbol_risk_signals.columns)
        
        logging.info(f"Risk management signals columns: {symbol_risk_signals.columns}")
        logging.info(f"Stop loss signals sum: {symbol_risk_signals[f'{self.symbol}_stop_loss'].sum()}")
        logging.info(f"Take profit signals sum: {symbol_risk_signals[f'{self.symbol}_take_profit'].sum()}")

if __name__ == '__main__':
    unittest.main()