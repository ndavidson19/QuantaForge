import polars as pl
import logging
from quantaforge.order_execution import OrderExecution
from quantaforge.order_execution import SimulatedOrderExecution

class Portfolio:
    def __init__(self, name, initial_cash, order_execution=None):
        self.name = name
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.history = []
        self.order_execution = order_execution if order_execution else SimulatedOrderExecution()
        self.data = None

    def buy(self, symbol, quantity, price):
        cost = quantity * price
        print(f"Cost: {cost}")
        print(f"Initial cash: {self.cash}")
        if self.cash >= cost:
            self.cash -= cost
            if symbol in self.positions:
                self.positions[symbol] += quantity
            else:
                self.positions[symbol] = quantity
            self.history.append((symbol, quantity, price, 'buy'))
            self.order_execution.place_order(symbol, quantity, 'BUY')
        else:
            raise ValueError("Insufficient funds")

    def sell(self, symbol, quantity, price):
        if symbol in self.positions and self.positions[symbol] >= quantity:
            self.cash += quantity * price
            self.positions[symbol] -= quantity
            self.history.append((symbol, quantity, price, 'sell'))
            self.order_execution.place_order(symbol, quantity, 'SELL')
        else:
            raise ValueError("Insufficient holdings")

    def value(self, current_prices):
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            price = current_prices[-1]  # Use the last price in the array
            logging.debug(f"Calculating value for {symbol} with quantity {quantity}")
            logging.debug(f"Current price: {price}")
            total_value += quantity * price
            logging.debug(f"Total value: {total_value}")
        return total_value

    def risk_management(self):
        df = pl.DataFrame(self.history)
        risk = df.select(pl.sum("quantity")).to_series()[0]
        return risk

    def set_data(self, data):
        self.data = data