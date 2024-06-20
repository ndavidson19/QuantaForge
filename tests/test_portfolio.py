import unittest
from quantaforge.portfolio import Portfolio
from quantaforge.order_execution import OrderExecution

class MockOrderExecution(OrderExecution):
    def __init__(self):
        pass
    def place_order(self, symbol, quantity, action):
        pass

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.order_execution = MockOrderExecution()
        self.portfolio = Portfolio(name='TestPortfolio', initial_cash=1000, order_execution=self.order_execution)

    def test_buy(self):
        self.portfolio.buy('AAPL', 1, 100)
        self.assertEqual(self.portfolio.cash, 900)
        self.assertEqual(self.portfolio.positions['AAPL'], 1)

    def test_sell(self):
        self.portfolio.buy('AAPL', 1, 100)
        self.portfolio.sell('AAPL', 1, 120)
        self.assertEqual(self.portfolio.cash, 1020)
        self.assertEqual(self.portfolio.positions['AAPL'], 0)

    def test_value(self):
        self.portfolio.buy('AAPL', 1, 100)
        current_prices = [110]  # Use list for current prices
        value = self.portfolio.value(current_prices)
        self.assertEqual(value, 1010)

if __name__ == '__main__':
    unittest.main()
