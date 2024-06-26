from enum import Enum


class Order:
    def __init__(self, symbol, quantity, order_type, price, timestamp):
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.price = price
        self.timestamp = timestamp

    def __str__(self):
        return f"Order(symbol={self.symbol}, quantity={self.quantity}, order_type={self.order_type}, price={self.price}, timestamp={self.timestamp})"
    
    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, config):
        return cls(config['symbol'], config['quantity'], OrderType(config['order_type']), config['price'], config['timestamp'])
    
class SimulatedOrderExecution:
    def __init__(self, order: Order, price: float, quantity: float):
        self.order = order
        self.price = price
        self.quantity = quantity

    def __str__(self):
        return f"SimulatedOrderExecution(order={self.order}, price={self.price}, quantity={self.quantity})"

    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {
            'order': self.order.to_dict(),
            'price': self.price,
            'quantity': self.quantity
        }
    
    @classmethod
    def from_dict(cls, config):
        return cls(Order.from_dict(config['order']), config['price'], config['quantity'])
    

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

                     
