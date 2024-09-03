from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ib_insync import IB, Contract, MarketOrder, Stock
from quantaforge.order import Order, OrderType
from quantaforge.strategy import Strategy
from quantaforge.generics import Portfolio

class Broker(ABC):
    @abstractmethod
    async def connect(self):
        pass

    @abstractmethod
    async def disconnect(self):
        pass

    @abstractmethod
    async def get_market_data(self, symbols: List[str]) -> Dict[str, float]:
        pass

    @abstractmethod
    async def place_order(self, order: Order):
        pass

class InteractiveBrokers(Broker):
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        self.ib = IB()
        self.host = host
        self.port = port
        self.client_id = client_id

    async def connect(self):
        await self.ib.connectAsync(self.host, self.port, self.client_id)

    async def disconnect(self):
        await self.ib.disconnectAsync()

    async def get_market_data(self, symbols: List[str]) -> Dict[str, float]:
        contracts = [Stock(symbol, 'SMART', 'USD') for symbol in symbols]
        tickers = await self.ib.reqTickersAsync(*contracts)
        return {ticker.contract.symbol: ticker.marketPrice() for ticker in tickers}

    async def place_order(self, order: Order):
        contract = Stock(order.symbol, 'SMART', 'USD')
        ib_order = MarketOrder('BUY' if order.quantity > 0 else 'SELL', abs(order.quantity))
        trade = await self.ib.placeOrderAsync(contract, ib_order)
        return trade
