import polars as pl
from typing import List, Dict, Any
from quantaforge.strategy import Strategy
from quantaforge.order import Order, OrderType
from datetime import datetime, timedelta
from quantaforge.generics import Portfolio

class RevisedOzempicStrategy(Strategy):
    def __init__(self, 
                 affected_sectors: List[str] = ['XLP', 'XLY', 'MO', 'PM', 'TAP', 'BUD', 'STZ', 'MCD', 'YUM', 'DNUT', 'KO', 'PEP'],
                 hedge_symbols: List[str] = ['XLV', 'LLY', 'NVO'],
                 rebalance_frequency: int = 30,  # Rebalance every 30 days
                 trend_window: int = 252,  # Look at 1-year trend
                 max_short_allocation: float = 0.4,  # 40% max short allocation
                 max_long_allocation: float = 0.4):  # 40% max long allocation
        super().__init__("Revised Ozempic Strategy")
        self.affected_sectors = affected_sectors
        self.hedge_symbols = hedge_symbols
        self.rebalance_frequency = rebalance_frequency
        self.trend_window = trend_window
        self.max_short_allocation = max_short_allocation
        self.max_long_allocation = max_long_allocation
        self.last_rebalance_date = None

    def generate_orders(self, data: Dict[str, Any], portfolio: Portfolio) -> List[Order]:
        current_date = data['timestamp']
        orders = []

        # Check if it's time to rebalance
        if self.last_rebalance_date is None or (current_date - self.last_rebalance_date).days >= self.rebalance_frequency:
            self.last_rebalance_date = current_date
            orders.extend(self._rebalance_portfolio(data, portfolio))

        return orders

    def _rebalance_portfolio(self, data: Dict[str, Any], portfolio: Portfolio) -> List[Order]:
        rebalance_orders = []
        total_value = portfolio.total_value

        # Close all existing positions
        for symbol, position in portfolio.positions.items():
            rebalance_orders.append(Order(symbol, -position.quantity, OrderType.MARKET, data[f'{symbol}_close'], data['timestamp']))

        # Calculate trend scores
        affected_sector_scores = self._calculate_trend_scores(data, self.affected_sectors)
        hedge_symbol_scores = self._calculate_trend_scores(data, self.hedge_symbols)

        # Allocate short positions in affected sectors
        short_allocation = min(self.max_short_allocation, sum(score < 0 for score in affected_sector_scores.values()) / len(affected_sector_scores) * self.max_short_allocation)
        for symbol, score in affected_sector_scores.items():
            if score < 0:  # Only short if the trend is negative
                allocation = short_allocation * (abs(score) / sum(abs(s) for s in affected_sector_scores.values() if s < 0))
                quantity = int((total_value * allocation) / data[f'{symbol}_close'])
                rebalance_orders.append(Order(symbol, -quantity, OrderType.MARKET, data[f'{symbol}_close'], data['timestamp']))

        # Allocate long positions in hedge symbols
        long_allocation = min(self.max_long_allocation, sum(score > 0 for score in hedge_symbol_scores.values()) / len(hedge_symbol_scores) * self.max_long_allocation)
        for symbol, score in hedge_symbol_scores.items():
            if score > 0:  # Only go long if the trend is positive
                allocation = long_allocation * (score / sum(s for s in hedge_symbol_scores.values() if s > 0))
                quantity = int((total_value * allocation) / data[f'{symbol}_close'])
                rebalance_orders.append(Order(symbol, quantity, OrderType.MARKET, data[f'{symbol}_close'], data['timestamp']))

        return rebalance_orders

    def _calculate_trend_scores(self, data: Dict[str, Any], symbols: List[str]) -> Dict[str, float]:
        scores = {}
        for symbol in symbols:
            start_price = data[f'{symbol}_close_{self.trend_window}']
            end_price = data[f'{symbol}_close']
            trend = (end_price - start_price) / start_price
            # Adjust trend based on recent performance (last month)
            recent_trend = (end_price - data[f'{symbol}_close_22']) / data[f'{symbol}_close_22']
            scores[symbol] = trend * 0.7 + recent_trend * 0.3  # Weight long-term trend more heavily
        return scores