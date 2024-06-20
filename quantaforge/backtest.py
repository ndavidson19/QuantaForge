import polars as pl
from quantaforge.portfolio import Portfolio
from quantaforge.strategy import Strategy
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Backtest:
    def __init__(self, name, strategy: Strategy, portfolio=None):
        self.name = name
        self.data = None
        self.strategy = strategy
        self.portfolio = portfolio if portfolio else Portfolio(initial_cash=100000, order_execution=None)

    def set_data(self, data):
        self.data = pl.DataFrame(data)
        self.portfolio.set_data(self.data) 

    def run(self):
        if self.data is None:
            raise ValueError("Data must be set before running backtest")
        
        signals = self.strategy.generate_signals(self.data)
        logging.debug(f"Signals: {signals}")
        
        for i, signal in enumerate(signals['signal']):
            logging.debug(f"Processing signal at index {i}: {signal}")
            if signal is None:
                continue  # Skip None values
            symbol = self.data['symbol'][i]
            price = self.data['close'][i]
            if signal > 0:
                logging.debug(f"Buy signal for {symbol} at price {price}")
                self.portfolio.buy(symbol, 10, price)
            elif signal < 0:
                logging.debug(f"Sell signal for {symbol} at price {price}")
                self.portfolio.sell(symbol, 10, price)
                
        return self.portfolio.value(current_prices=self.data['close'].to_numpy())
    
    
    
    def calculate_risk(self):
        return self.portfolio.risk_management()
    
    def calculate_performance_metrics(self):
        logging.debug("Calculating performance metrics...")
