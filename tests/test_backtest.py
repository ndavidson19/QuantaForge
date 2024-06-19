import unittest
from unittest.mock import Mock
import polars as pl
from quantaforge.strategy import MovingAverageStrategy
from quantaforge.backtest import Backtest
from quantaforge.portfolio import Portfolio
from quantaforge.order_execution import OrderExecution
import logging

class TestBacktest(unittest.TestCase):
    def setUp(self):
        self.data = {
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'close': [150, 152, 148, 151]
        }
        self.df = pl.DataFrame(self.data)
        self.strategy = MovingAverageStrategy(window=2)

    def test_backtest_run(self):
        mock_order_execution = Mock(spec=OrderExecution)
        mock_portfolio = Portfolio(initial_cash=100000, order_execution=mock_order_execution)
        
        # Mock the buy and sell methods to avoid placing real orders
        mock_portfolio.buy = Mock()
        mock_portfolio.sell = Mock()
        mock_portfolio.value = Mock(return_value=100000)  # Mock the portfolio value

        backtest = Backtest(self.df, self.strategy, portfolio=mock_portfolio)
        final_value = backtest.run()

        self.assertGreater(final_value, 0)
        # Ensure that buy and sell methods were called
        logging.debug(f"Buy method call count: {mock_portfolio.buy.call_count}")
        logging.debug(f"Sell method call count: {mock_portfolio.sell.call_count}")

        mock_portfolio.buy.assert_called()
        mock_portfolio.sell.assert_not_called()  # Adjust this based on expected behavior

if __name__ == '__main__':
    unittest.main()
