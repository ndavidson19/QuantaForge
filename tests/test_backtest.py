import unittest
from unittest.mock import Mock, patch
import polars as pl
from quantaforge.strategy import Strategy
from quantaforge.backtest_new import Backtest
from quantaforge.generics import Portfolio, Position
from quantaforge.indicators import SMA
from quantaforge.signals import CrossOverSignal
from quantaforge.order import Order, OrderType
from datetime import datetime

class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.data = pl.DataFrame({
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'timestamp': ['09:30:00', '09:30:00', '09:30:00', '09:30:00'],
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'open': [149.0, 151.0, 147.0, 150.0],
            'high': [151.0, 153.0, 149.0, 152.0],
            'low': [148.0, 150.0, 146.0, 149.0],
            'close': [150.0, 152.0, 148.0, 151.0],
            'volume': [1000000, 1100000, 900000, 1050000]
        })
        
        self.strategy = Strategy("TestStrategy")
        self.strategy.add_indicator("SMA_2", SMA(2))
        self.strategy.add_signal("CrossOver", CrossOverSignal("close", "SMA_2"))

    @patch('quantaforge.backtest.Portfolio')
    def test_backtest_run(self, MockPortfolio):
        mock_portfolio = MockPortfolio.return_value
        mock_portfolio.cash = 100000  # Set initial cash
        mock_portfolio.total_value = 100000
        mock_portfolio.positions = {'AAPL': Position('AAPL', 10, 150, datetime.now())}
        mock_portfolio.calculate_total_value.return_value = 100000

        backtest = Backtest("TestBacktest", self.strategy, initial_capital=100000)
        backtest.set_data(self.data)
        
        # Add some mock trades including an open trade
        backtest.trades = [
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': 10, 'entry_price': 150, 'exit_price': 155, 'commission': 1},
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': -10, 'entry_price': 150, 'exit_price': 145, 'commission': 1},
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': 5, 'entry_price': 148, 'exit_price': None, 'commission': 1}
        ]
        backtest.equity_curve = [100000, 100500, 101000, 100800]
        
        results = backtest.run()

        self.assertIsNotNone(results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)

    def test_backtest_calculate_performance_metrics(self):
        backtest = Backtest("TestBacktest", self.strategy, initial_capital=100000)
        backtest.set_data(self.data)
        
        # Add some mock trades including an open trade
        backtest.trades = [
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': 10, 'entry_price': 150, 'exit_price': 155, 'commission': 1},
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': -10, 'entry_price': 150, 'exit_price': 145, 'commission': 1},
            {'timestamp': datetime.now(), 'symbol': 'AAPL', 'quantity': 5, 'entry_price': 148, 'exit_price': None, 'commission': 1}
        ]
        backtest.equity_curve = [100000, 100500, 101000, 100800]
        
        results = backtest.run()
        
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)

if __name__ == '__main__':
    unittest.main()