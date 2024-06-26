import numpy as np
from typing import List, Dict
from scipy import stats

class PerformanceMetrics:
    def __init__(self, equity_curve: List[float], trades: List[Dict], data_frequency: str):
        self.equity_curve = np.array(equity_curve)
        self.trades = trades
        self.data_frequency = data_frequency
        self.returns = np.diff(self.equity_curve) / self.equity_curve[:-1]

    def total_return(self) -> float:
        return (self.equity_curve[-1] / self.equity_curve[0]) - 1

    def annualized_return(self) -> float:
        total_days = len(self.equity_curve)
        years = total_days / 252  # Assuming 252 trading days per year
        return (1 + self.total_return()) ** (1 / years) - 1

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        excess_returns = self.returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        return drawdown.min()

    def win_rate(self) -> float:
        wins = sum(1 for trade in self.trades if trade['quantity'] * (trade['price'] - trade['entry_price']) > 0)
        return wins / len(self.trades)

    def profit_factor(self) -> float:
        gross_profits = sum(trade['quantity'] * (trade['price'] - trade['entry_price']) 
                            for trade in self.trades if trade['quantity'] * (trade['price'] - trade['entry_price']) > 0)
        gross_losses = sum(trade['quantity'] * (trade['price'] - trade['entry_price']) 
                           for trade in self.trades if trade['quantity'] * (trade['price'] - trade['entry_price']) < 0)
        return gross_profits / abs(gross_losses) if gross_losses != 0 else np.inf

    def calmar_ratio(self) -> float:
        return self.annualized_return() / abs(self.max_drawdown())

    def omega_ratio(self, threshold: float = 0) -> float:
        returns_above_threshold = self.returns[self.returns > threshold]
        returns_below_threshold = self.returns[self.returns <= threshold]
        return returns_above_threshold.sum() / abs(returns_below_threshold.sum())

    def expectancy(self) -> float:
        win_rate = self.win_rate()
        average_win = self.average_win()
        average_loss = self.average_loss()
        return (win_rate * average_win) - ((1 - win_rate) * average_loss)

    def average_trade(self) -> float:
        return np.mean([trade['quantity'] * (trade['price'] - trade['entry_price']) for trade in self.trades])

    def average_win(self) -> float:
        winning_trades = [trade['quantity'] * (trade['price'] - trade['entry_price']) 
                          for trade in self.trades if trade['quantity'] * (trade['price'] - trade['entry_price']) > 0]
        return np.mean(winning_trades) if winning_trades else 0

    def average_loss(self) -> float:
        losing_trades = [trade['quantity'] * (trade['price'] - trade['entry_price']) 
                         for trade in self.trades if trade['quantity'] * (trade['price'] - trade['entry_price']) < 0]
        return np.mean(losing_trades) if losing_trades else 0

    # Add more performance metrics as needed