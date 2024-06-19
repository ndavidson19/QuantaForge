import polars as pl
from quantaforge.order_execution import OrderExecution

class Portfolio:
    def __init__(self, initial_cash, order_execution: OrderExecution):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.history = []
        self.order_execution = order_execution

    def buy(self, symbol, quantity, price):
        cost = quantity * price
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
            total_value += quantity * current_prices[symbol]
        return total_value

    def risk_management(self):
        df = pl.DataFrame(self.history)
        risk = df.select(pl.sum("quantity")).to_series()[0]
        return risk
