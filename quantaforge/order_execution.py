from ib_insync import IB, MarketOrder, Stock

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
