from ib_insync import IB, MarketOrder, Stock
import logging

class OrderExecution:
    def __init__(self, host='127.0.0.1', port=7497, clientId=1):
        self.ib = IB()
        self.ib.connect(host, port, clientId)

    def place_order(self, symbol, quantity, action):
        contract = Stock(symbol, 'SMART', 'USD')
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def disconnect(self):
        self.ib.disconnect()


class SimulatedOrderExecution:
    def __init__(self):
        self.orders = []

    def place_order(self, symbol, quantity, action):
        order = {
            'symbol': symbol,
            'quantity': quantity,
            'action': action,
            'status': 'filled'
        }
        self.orders.append(order)
        logging.debug(f"Simulated order placed: {order}")

    def get_orders(self):
        return self.orders

