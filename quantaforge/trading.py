from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, List, Union, Callable, Tuple
from datetime import datetime
from quantaforge.generics import Portfolio, StrategyBase, Position
from quantaforge.order import Order, OrderType
from quantaforge.performance_metrics import PerformanceMetrics
import numpy as np

class TradingEngineBase(ABC):
    def __init__(self, 
                 name: str, 
                 strategy: StrategyBase, 
                 initial_capital: float,
                 commission: Union[float, Callable] = 0.001,
                 slippage: Union[float, Callable] = 0.0005,
                 position_sizer: Callable = None,
                 risk_manager: Callable = None):
        self.name = name
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.portfolio = Portfolio(initial_capital)
        self.orders: List[Order] = []
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[float] = [initial_capital]
        self.position_sizer = position_sizer or self._default_position_sizer
        self.risk_manager = risk_manager or self._default_risk_manager
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger

    @abstractmethod
    def run(self):
        pass

    def _process_bar(self, row: Dict[str, Any]):
        timestamp = row['timestamp']
        symbol = row['symbol']
        self.logger.debug(f"Processing bar: {timestamp} - {symbol}")

        self._update_portfolio_value(row)
        self._process_pending_orders(row)
        self._check_and_close_positions(row)
        self._check_and_open_positions(row)
        self._apply_risk_management(row)
        self._record_equity(row)

    def _check_and_close_positions(self, row: Dict[str, Any]):
        positions_to_close = []
        for position in self.portfolio.positions.values():
            exit_triggered, exit_reasons = self._check_exit_conditions(row, position.symbol)
            if exit_triggered:
                positions_to_close.append(position.symbol)
                self.logger.debug(f"Exit signal triggered for {position.symbol}. Reasons: {exit_reasons}")

        for symbol in positions_to_close:
            if symbol in self.portfolio.positions:
                self._place_order(symbol, -self.portfolio.positions[symbol].quantity, OrderType.MARKET, row)

    def _check_and_open_positions(self, row: Dict[str, Any]):
        entry_triggered, entry_reasons = self._check_entry_conditions(row)
        if entry_triggered:
            self.logger.debug(f"Entry signal triggered for {row['symbol']}. Reasons: {entry_reasons}")
            quantity = self._calculate_position_size(row)
            self.logger.debug(f"Calculated position size: {quantity}")
            if quantity > 0:
                self._place_order(row['symbol'], quantity, OrderType.MARKET, row)
            else:
                self.logger.warning(f"Calculated position size is zero or negative: {quantity}")

    def _check_entry_conditions(self, row: Dict[str, Any]) -> Tuple[bool, List[str]]:
        reasons = []
        for condition in self.strategy.entry_conditions:
            result = condition.evaluate(row)
            reasons.append(f"{condition.indicator} {condition.operator} {condition.value}: {result}")
        all_conditions_met = all(condition.evaluate(row) for condition in self.strategy.entry_conditions)
        return all_conditions_met, reasons

    def _check_exit_conditions(self, row: Dict[str, Any], symbol: str) -> Tuple[bool, List[str]]:
        reasons = []
        for condition in self.strategy.exit_conditions:
            result = condition.evaluate(row)
            reasons.append(f"{condition.indicator} {condition.operator} {condition.value}: {result}")
        all_conditions_met = all(condition.evaluate(row) for condition in self.strategy.exit_conditions)
        return all_conditions_met, reasons

    def _calculate_position_size(self, row: Dict[str, Any]) -> int:
        return self.position_sizer(self.portfolio.total_value, row['close'])

    def _place_order(self, symbol: str, quantity: int, order_type: OrderType, row: Dict[str, Any]):
        price = self._calculate_execution_price(row, quantity, order_type)
        order = Order(symbol, quantity, order_type, price, row['timestamp'])
        self.orders.append(order)
        self._execute_order(order, row)

    def _calculate_execution_price(self, row: Dict[str, Any], quantity: int, order_type: OrderType) -> float:
        base_price = row['close']
        slippage = self.slippage if isinstance(self.slippage, float) else self.slippage(quantity)
        slippage_adjustment = base_price * slippage * (1 if quantity > 0 else -1)
        return base_price + slippage_adjustment

    def _execute_order(self, order: Order, row: Dict[str, Any]):
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

    def _calculate_commission(self, order: Order) -> float:
        if callable(self.commission):
            return self.commission(order.quantity, order.price)
        return abs(order.quantity * order.price * self.commission)

    def _record_trade(self, order: Order, commission: float):
        entry_price = order.price if order.quantity > 0 else self.portfolio.positions.get(order.symbol, Position(order.symbol, 0, 0, order.timestamp)).entry_price
        exit_price = order.price if order.quantity < 0 else None
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'commission': commission
        })

    def _update_portfolio_value(self, row: Dict[str, Any]):
        prices = {row['symbol']: row['close']}
        self.portfolio.calculate_total_value(row['timestamp'], prices)

    def _process_pending_orders(self, row: Dict[str, Any]):
        # Implement logic to process limit and stop orders
        pass

    def _apply_risk_management(self, row: Dict[str, Any]):
        for risk_rule in self.strategy.risk_management:
            risk_rule.apply(self.portfolio, row)

    def _record_equity(self, row: Dict[str, Any]):
        self.equity_curve.append(self.portfolio.total_value)

    @staticmethod
    def _default_position_sizer(portfolio_value: float, price: float) -> int:
        return int((portfolio_value * 0.01) / price)  # 1% of portfolio value

    @staticmethod
    def _default_risk_manager(returns: np.array, confidence_level: float = 0.95) -> float:
        return np.percentile(returns, 100 * (1 - confidence_level))

    def calculate_performance_metrics(self):
        metrics = PerformanceMetrics(self.equity_curve, self.trades, self.data_frequency)
        return {
            'total_return': metrics.total_return(),
            'annualized_return': metrics.annualized_return(),
            'sharpe_ratio': metrics.sharpe_ratio(),
            'sortino_ratio': metrics.sortino_ratio(),
            'max_drawdown': metrics.max_drawdown(),
            'win_rate': metrics.win_rate(),
            'profit_factor': metrics.profit_factor(),
            'calmar_ratio': metrics.calmar_ratio(),
            'omega_ratio': metrics.omega_ratio(),
            'expectancy': metrics.expectancy(),
            'average_trade': metrics.average_trade(),
            'average_win': metrics.average_win(),
            'average_loss': metrics.average_loss(),
            'trade_count': len(self.trades),
        }