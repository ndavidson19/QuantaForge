from enum import Enum


class Order:
    def __init__(self, name, order_type, **kwargs):
        self.name = name
        self.order_type = order_type
        self.kwargs = kwargs

    def __str__(self):
        return f"Order(name={self.name}, order_type={self.order_type}, kwargs={self.kwargs})"

    def __repr__(self):
        return str(self)
    
    def to_dict(self):
        return {
            'name': self.name,
            'order_type': self.order_type,
            'kwargs': self.kwargs
        }
    
    @classmethod
    def from_dict(cls, config):
        return cls(config['name'], config['order_type'], **config['kwargs'])
    
class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

                     
