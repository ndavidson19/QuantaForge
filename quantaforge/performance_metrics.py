import numpy as np
from typing import List, Dict

class PerformanceMetrics:
    def __init__(self, equity_curve: List[float], trades: List[Dict], data_frequency: str):
        self.equity_curve = np.array(equity_curve)
        self.trades = trades
        self.data_frequency = data_frequency
        self.returns = np.diff(self.equity_curve) / self.equity_curve[:-1] if len(self.equity_curve) > 1 else np.array([])

    def total_return(self) -> float:
        return (self.equity_curve[-1] / self.equity_curve[0]) - 1 if len(self.equity_curve) > 1 else 0

    def annualized_return(self) -> float:
        total_days = len(self.equity_curve)
        years = total_days / 252  # Assuming 252 trading days per year
        return (1 + self.total_return()) ** (1 / years) - 1 if years > 0 else 0

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.returns) == 0:
            return 0
        excess_returns = self.returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        if len(self.returns) == 0:
            return 0
        excess_returns = self.returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() != 0 else 0

    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min() if len(drawdown) > 0 else 0

    def win_rate(self) -> float:
        if not self.trades:
            return 0
        wins = sum(1 for trade in self.trades if self._is_winning_trade(trade))
        return wins / len(self.trades)

    def profit_factor(self) -> float:
        if not self.trades:
            return 0
        gross_profits = sum(self._calculate_trade_pnl(trade) for trade in self.trades if self._is_winning_trade(trade))
        gross_losses = sum(abs(self._calculate_trade_pnl(trade)) for trade in self.trades if not self._is_winning_trade(trade))
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')

    def calmar_ratio(self) -> float:
        max_dd = self.max_drawdown()
        return self.annualized_return() / abs(max_dd) if max_dd != 0 else 0

    def omega_ratio(self, threshold: float = 0) -> float:
        returns_above_threshold = self.returns[self.returns > threshold]
        returns_below_threshold = self.returns[self.returns <= threshold]
        return returns_above_threshold.sum() / abs(returns_below_threshold.sum()) if returns_below_threshold.sum() != 0 else float('inf')

    def expectancy(self) -> float:
        if not self.trades:
            return 0
        win_rate = self.win_rate()
        average_win = self.average_win()
        average_loss = self.average_loss()
        return (win_rate * average_win) - ((1 - win_rate) * average_loss)

    def average_trade(self) -> float:
        if not self.trades:
            return 0
        return np.mean([self._calculate_trade_pnl(trade) for trade in self.trades])

    def average_win(self) -> float:
        winning_trades = [self._calculate_trade_pnl(trade) for trade in self.trades if self._is_winning_trade(trade)]
        return np.mean(winning_trades) if winning_trades else 0

    def average_loss(self) -> float:
        losing_trades = [self._calculate_trade_pnl(trade) for trade in self.trades if not self._is_winning_trade(trade)]
        return np.mean(losing_trades) if losing_trades else 0

    def _is_winning_trade(self, trade: Dict) -> bool:
        return self._calculate_trade_pnl(trade) > 0

    def _calculate_trade_pnl(self, trade: Dict) -> float:
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price')
        if exit_price is None:
            return 0  # If the trade is not closed, consider PnL as 0
        quantity = trade.get('quantity', 0)
        return quantity * (exit_price - entry_price)