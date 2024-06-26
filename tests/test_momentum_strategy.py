import unittest
import pandas as pd
import numpy as np
import polars as pl
import logging
from examples.MomentumStrategy import MomentumStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestMomentumStrategy(unittest.TestCase):
    def setUp(self):
        self.symbol = "AAPL"
        self.strategy = MomentumStrategy(self.symbol)
        self.mock_data = self._create_mock_data()

    def _create_mock_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate a price series with momentum
        price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        price = price + 20 * np.sin(np.arange(len(dates)) / 50)  # Add cyclical component
        price = price + np.linspace(0, 30, len(dates))  # Add upward trend
        
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
        rsi = self.strategy.indicators[f"RSI_{14}"]
        macd = self.strategy.indicators[f"MACD_{12}_{26}_{9}"]
        
        data_with_indicators = rsi.calculate(self.mock_data)
        data_with_indicators = macd.calculate(data_with_indicators)
        
        self.assertIn(f"RSI_{14}", data_with_indicators.columns)
        self.assertIn(f"MACD_{12}_{26}_{9}_line", data_with_indicators.columns)
        self.assertIn(f"MACD_{12}_{26}_{9}_signal", data_with_indicators.columns)
        self.assertIn(f"MACD_{12}_{26}_{9}_histogram", data_with_indicators.columns)
        
        logging.info(f"Calculated indicators: {data_with_indicators.columns}")
        logging.info(f"RSI range: {data_with_indicators[f'RSI_{14}'].min()} to {data_with_indicators[f'RSI_{14}'].max()}")
        logging.info(f"MACD line range: {data_with_indicators[f'MACD_{12}_{26}_{9}_line'].min()} to {data_with_indicators[f'MACD_{12}_{26}_{9}_line'].max()}")

    def test_generate_signals(self):
        logging.info("Testing generate_signals method")
        rsi = self.strategy.indicators[f"RSI_{14}"]
        macd = self.strategy.indicators[f"MACD_{12}_{26}_{9}"]
        
        data_with_indicators = rsi.calculate(self.mock_data)
        data_with_indicators = macd.calculate(data_with_indicators)
        
        signals = self.strategy.generate_signals(data_with_indicators)
        
        self.assertIsInstance(signals, dict)
        self.assertIn(self.symbol, signals)
        
        symbol_signals = signals[self.symbol]
        self.assertIn(f"{self.symbol}_buy", symbol_signals.columns)
        self.assertIn(f"{self.symbol}_sell", symbol_signals.columns)
        
        logging.info(f"Generated signals columns: {symbol_signals.columns}")
        logging.info(f"Buy signals sum: {symbol_signals[f'{self.symbol}_buy'].sum()}")
        logging.info(f"Sell signals sum: {symbol_signals[f'{self.symbol}_sell'].sum()}")
        logging.info(f"Number of buy signals: {(symbol_signals[f'{self.symbol}_buy'] > 0).sum()}")
        logging.info(f"Number of sell signals: {(symbol_signals[f'{self.symbol}_sell'] > 0).sum()}")

    def test_manage_risk(self):
        logging.info("Testing manage_risk method")
        risk_signals = self.strategy.manage_risk(self.mock_data)
        
        self.assertIsInstance(risk_signals, dict)
        self.assertIn(self.symbol, risk_signals)
        
        symbol_risk_signals = risk_signals[self.symbol]
        self.assertIn(f"{self.symbol}_trailing_stop", symbol_risk_signals.columns)
        self.assertIn(f"{self.symbol}_trailing_stop_level", symbol_risk_signals.columns)
        
        logging.info(f"Risk management signals columns: {symbol_risk_signals.columns}")
        logging.info(f"Number of trailing stop signals: {(symbol_risk_signals[f'{self.symbol}_trailing_stop'] > 0).sum()}")
        logging.info(f"Trailing stop signals sum: {symbol_risk_signals[f'{self.symbol}_trailing_stop'].sum()}")

if __name__ == '__main__':
    unittest.main()