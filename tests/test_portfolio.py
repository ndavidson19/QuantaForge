import unittest
from quantaforge.generics import Portfolio
from datetime import datetime

class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.portfolio = Portfolio(initial_cash=10000)

    def test_open_position(self):
        self.portfolio.open_position('AAPL', 10, 150, datetime.now())
        self.assertEqual(self.portfolio.cash, 8500)
        self.assertIn('AAPL', self.portfolio.positions)
        self.assertEqual(self.portfolio.positions['AAPL'].quantity, 10)

    def test_close_position(self):
        open_time = datetime.now()
        self.portfolio.open_position('AAPL', 10, 150, open_time)
        close_time = datetime.now()
        self.portfolio.close_position('AAPL', 10, 160, close_time)
        self.assertEqual(self.portfolio.cash, 10100)
        self.assertNotIn('AAPL', self.portfolio.positions)

    def test_calculate_total_value(self):
        self.portfolio.open_position('AAPL', 10, 150, datetime.now())
        total_value = self.portfolio.calculate_total_value(datetime.now(), {'AAPL': 160})
        self.assertEqual(total_value, 10100)

    def test_insufficient_funds(self):
        with self.assertRaises(ValueError):
            self.portfolio.open_position('AAPL', 1000, 150, datetime.now())

if __name__ == '__main__':
    unittest.main()