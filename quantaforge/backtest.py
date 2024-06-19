import polars as pl
from quantaforge.portfolio import Portfolio
from quantaforge.strategy import Strategy
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtest:
    def __init__(self, data, strategy: Strategy, portfolio=None):
        self.data = pl.DataFrame(data)
        self.strategy = strategy
        self.portfolio = portfolio if portfolio else Portfolio(initial_cash=100000, order_execution=None)

    def run(self):
        signals = self.strategy.generate_signals(self.data)
        logging.debug(f"Signals: {signals}")
        for i, signal in enumerate(signals['close']):
            logging.debug(f"Processing signal at index {i}: {signal}")
            if signal is None:
                continue  # Skip None values
            if signal > 0:
                logging.debug(f"Buy signal for {self.data['symbol'][i]} at price {self.data['close'][i]}")
                self.portfolio.buy(self.data['symbol'][i], 10, self.data['close'][i])
            elif signal < 0:
                logging.debug(f"Sell signal for {self.data['symbol'][i]} at price {self.data['close'][i]}")
                self.portfolio.sell(self.data['symbol'][i], 10, self.data['close'][i])
        return self.portfolio.value(current_prices=self.data['close'])
