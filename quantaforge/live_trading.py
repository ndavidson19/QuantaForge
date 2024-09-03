import asyncio
from typing import Any, Callable, Dict, Union
from quantaforge.brokers import Broker
from quantaforge.trading import TradingEngineBase
from quantaforge.strategy import StrategyBase
from quantaforge.order import Order


class LiveTradingEngine(TradingEngineBase):
    def __init__(self, 
                 name: str, 
                 strategy: StrategyBase, 
                 broker: Broker,
                 initial_capital: float,
                 commission: Union[float, Callable] = 0.001,
                 slippage: Union[float, Callable] = 0.0005,
                 position_sizer: Callable = None,
                 risk_manager: Callable = None):
        super().__init__(name, strategy, initial_capital, commission, slippage, position_sizer, risk_manager)
        self.broker = broker
        self.running = False

    async def run(self):
        await self.broker.connect()
        self.running = True
        self.logger.info("Live trading started")
        
        while self.running:
            market_data = await self.broker.get_market_data(self.strategy.symbols)
            self._process_bar(market_data)
            await asyncio.sleep(1)  # Adjust based on your desired frequency

    async def stop(self):
        self.running = False
        await self.broker.disconnect()
        self.logger.info("Live trading stopped")

    def _execute_order(self, order: Order, row: Dict[str, Any]):
        asyncio.create_task(self._async_execute_order(order, row))

    async def _async_execute_order(self, order: Order, row: Dict[str, Any]):
        trade = await self.broker.place_order(order)
        commission = self._calculate_commission(order)
        total_cost = order.quantity * order.price + commission

        if order.quantity > 0:  # Buy order
            if self.portfolio.cash >= total_cost:
                self.portfolio.open_position(order.symbol, order.quantity, order.price, order.timestamp)
                self.portfolio.cash -= total_cost
            else:
                self.logger.warning(f"Insufficient funds to execute order: {order}")
        else:  # Sell order
            self.portfolio.close_position(order.symbol, abs(order.quantity), order.price, order.timestamp)
            self.portfolio.cash += -total_cost  # Negative cost means cash inflow

        self._record_trade(order, commission)
        self.logger.info(f"Order executed: {trade}")
