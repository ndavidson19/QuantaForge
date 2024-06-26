import numpy as np
import polars as pl
from dataclasses import dataclass, replace

@dataclass
class RiskParameters:
    max_drawdown: float = 0.2
    var_threshold: float = 0.05
    var_confidence: float = 0.95
    max_leverage: float = 2.0
    position_size_limit: float = 0.1
    correlation_threshold: float = 0.7

class RiskManagementModule:
    def __init__(self):
        self.position_sizers = {'fixed': fixed_position_size, 'kelly': kelly_criterion}
        self.stop_loss_strategies = {'fixed': fixed_stop_loss, 'atr': atr_stop_loss}

    def calculate_position_size(self, method, **kwargs):
        return self.position_sizers[method](**kwargs)

    def apply_stop_loss(self, method, **kwargs):
        return self.stop_loss_strategies[method](**kwargs)

    def calculate_var(self, returns, confidence_level=0.95):
        # Calculate Value at Risk
        return np.percentile(returns, (1 - confidence_level) * 100)

class RiskManager:
    def __init__(self, **kwargs):
        default_params = RiskParameters()
        self.risk_params = replace(default_params, **kwargs)

    def calculate_var(self, returns: np.array) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns, (1 - self.risk_params.var_confidence) * 100)

    def calculate_cvar(self, returns: np.array) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self.calculate_var(returns)
        return returns[returns <= var].mean()

    def calculate_max_drawdown(self, equity_curve: np.array) -> float:
        """Calculate Maximum Drawdown"""
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return np.min(drawdown)

    def calculate_sharpe_ratio(self, returns: np.array, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, returns: np.array, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        excess_returns = returns - risk_free_rate / 252  # Assuming daily returns
        downside_returns = excess_returns[excess_returns < 0]
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

    def calculate_position_size(self, volatility: float, account_value: float) -> float:
        """Calculate position size based on volatility"""
        risk_per_trade = account_value * self.risk_params.position_size_limit
        return risk_per_trade / (volatility * np.sqrt(252))

    def check_correlation(self, returns1: np.array, returns2: np.array) -> bool:
        """Check if correlation between two assets is below threshold"""
        correlation = np.corrcoef(returns1, returns2)[0, 1]
        return abs(correlation) < self.risk_params.correlation_threshold

    def apply_risk_limits(self, strategy: 'Strategy', data: pl.DataFrame) -> None:
        """Apply risk limits to a strategy"""
        returns = data['close'].pct_change().dropna()
        
        # Check Value at Risk
        var = self.calculate_var(returns)
        if var < self.risk_params.var_threshold:
            strategy.reduce_exposure(var / self.risk_params.var_threshold)

        # Check Maximum Drawdown
        equity_curve = (1 + returns).cumprod()
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        if max_drawdown > self.risk_params.max_drawdown:
            strategy.reduce_exposure(self.risk_params.max_drawdown / max_drawdown)

        # Check Leverage
        if strategy.get_leverage() > self.risk_params.max_leverage:
            strategy.reduce_leverage(self.risk_params.max_leverage / strategy.get_leverage())

        # Adjust position sizes based on volatility
        volatility = returns.std()
        for position in strategy.get_positions():
            max_size = self.calculate_position_size(volatility, strategy.get_account_value())
            if position.size > max_size:
                strategy.resize_position(position, max_size)