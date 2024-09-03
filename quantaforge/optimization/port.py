import polars as pl
import numpy as np
import cvxpy as cp
from typing import List, Dict, Union, Tuple
from quantforge import QuantForgeAPI

class PortfolioOptimizer:
    def __init__(self, qf_api: QuantForgeAPI):
        self.qf_api = qf_api
        self.prices = None
        self.returns = None
        self.expected_returns = None
        self.cov_matrix = None
        self.weights = None

    async def load_data(self, tickers: List[str], start_date: str, end_date: str):
        self.prices = await self.qf_api.get_historical_prices(tickers, start_date, end_date)
        self.returns = self.prices.select(pl.all().pct_change()).drop_nulls()

    def calculate_expected_returns(self, method: str = "mean_historical_return", **kwargs):
        if method == "mean_historical_return":
            self.expected_returns = self.returns.mean().to_numpy()
        elif method == "ema_historical_return":
            span = kwargs.get("span", 500)
            self.expected_returns = self.returns.ewm_mean(span=span).last().to_numpy()
        elif method == "capm_return":
            market_prices = self.qf_api.get_market_data("SPY")
            market_returns = market_prices.select(pl.all().pct_change()).drop_nulls()
            beta = self.calculate_beta(market_returns)
            market_return = market_returns.mean().to_numpy()[0]
            risk_free_rate = kwargs.get("risk_free_rate", 0.02)
            self.expected_returns = risk_free_rate + beta * (market_return - risk_free_rate)
        else:
            raise ValueError(f"Unknown expected returns method: {method}")

    def calculate_risk_model(self, method: str = "sample_cov", **kwargs):
        if method == "sample_cov":
            self.cov_matrix = self.returns.cov().to_numpy()
        elif method == "exp_cov":
            span = kwargs.get("span", 180)
            self.cov_matrix = self.returns.ewm_cov(span=span).last().to_numpy()
        elif method == "semicovariance":
            benchmark = kwargs.get("benchmark", 0)
            self.cov_matrix = self.calculate_semicovariance(benchmark).to_numpy()
        else:
            raise ValueError(f"Unknown risk model method: {method}")

    def calculate_semicovariance(self, benchmark: float = 0) -> pl.DataFrame:
        diff = self.returns - benchmark
        negative_diff = diff.with_columns(pl.when(pl.all() < 0).then(pl.all()).otherwise(0))
        return (negative_diff.T @ negative_diff) / len(diff)

    def calculate_beta(self, market_returns: pl.DataFrame) -> pl.Series:
        cov_with_market = self.returns.cov(market_returns)
        market_var = market_returns.var().to_numpy()[0]
        return cov_with_market / market_var

    def optimize(self, method: str = "max_sharpe", **kwargs):
        n = len(self.expected_returns)
        w = cp.Variable(n)
        risk_free_rate = kwargs.get('risk_free_rate', 0.02)

        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]

        if method == "max_sharpe":
            portfolio_return = self.expected_returns @ w
            portfolio_risk = cp.quad_form(w, self.cov_matrix)
            sharpe_ratio = (portfolio_return - risk_free_rate) / cp.sqrt(portfolio_risk)
            objective = cp.Maximize(sharpe_ratio)
        elif method == "min_volatility":
            objective = cp.Minimize(cp.quad_form(w, self.cov_matrix))
        elif method == "efficient_risk":
            target_volatility = kwargs.get('target_volatility', 0.1)
            portfolio_return = self.expected_returns @ w
            portfolio_risk = cp.quad_form(w, self.cov_matrix)
            constraints.append(cp.sqrt(portfolio_risk) <= target_volatility)
            objective = cp.Maximize(portfolio_return)
        elif method == "efficient_return":
            target_return = kwargs.get('target_return', 0.1)
            portfolio_return = self.expected_returns @ w
            portfolio_risk = cp.quad_form(w, self.cov_matrix)
            constraints.append(portfolio_return >= target_return)
            objective = cp.Minimize(portfolio_risk)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if problem.status != cp.OPTIMAL:
            raise ValueError("Optimization problem did not converge")

        self.weights = pl.DataFrame({
            'asset': self.returns.columns,
            'weight': w.value
        })
        return self.weights

    def get_portfolio_performance(self) -> Dict[str, float]:
        if self.weights is None:
            raise ValueError("Portfolio must be optimized before getting performance")
        
        weights = self.weights['weight'].to_numpy()
        expected_return = weights @ self.expected_returns
        volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe_ratio = (expected_return - 0.02) / volatility  # Assuming risk-free rate of 2%
        
        return {
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio
        }

    def get_efficient_frontier(self, points: int = 100) -> pl.DataFrame:
        target_returns = np.linspace(self.expected_returns.min(), self.expected_returns.max(), points)
        efficient_portfolios = []

        for target in target_returns:
            self.optimize(method="efficient_return", target_return=target)
            performance = self.get_portfolio_performance()
            efficient_portfolios.append({
                'return': performance['expected_return'],
                'volatility': performance['volatility']
            })

        return pl.DataFrame(efficient_portfolios)

    def get_asset_data(self) -> pl.DataFrame:
        return pl.DataFrame({
            'asset': self.returns.columns,
            'expected_return': self.expected_returns,
            'volatility': np.sqrt(np.diag(self.cov_matrix))
        })

    def get_discrete_allocation(self, total_portfolio_value: float) -> Tuple[Dict[str, int], float]:
        if self.weights is None:
            raise ValueError("Portfolio must be optimized before discrete allocation")
        
        latest_prices = self.prices.select(pl.all().last()).to_dict(False)
        latest_prices = {k: v[0] for k, v in latest_prices.items()}
        
        # Calculate the number of shares for each asset
        shares = {}
        leftover = total_portfolio_value
        for asset, weight in self.weights.iter_rows():
            price = latest_prices[asset]
            asset_value = total_portfolio_value * weight
            number_of_shares = int(asset_value / price)
            shares[asset] = number_of_shares
            leftover -= number_of_shares * price

        return shares, leftover

# Example usage
async def main():
    qf_api = QuantForgeAPI("your_api_key")
    optimizer = PortfolioOptimizer(qf_api)

    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "FB"]
    await optimizer.load_data(tickers, "2018-01-01", "2023-06-01")

    optimizer.calculate_expected_returns(method="mean_historical_return")
    optimizer.calculate_risk_model(method="sample_cov")

    # Standard mean-variance optimization
    weights = optimizer.optimize(method="max_sharpe")
    print("Optimized weights:")
    print(weights)
    print("\nPortfolio performance:")
    print(optimizer.get_portfolio_performance())

    # Get data for plotting
    ef_data = optimizer.get_efficient_frontier()
    print("\nEfficient Frontier:")
    print(ef_data)

    asset_data = optimizer.get_asset_data()
    print("\nAsset Data:")
    print(asset_data)

    # Discrete allocation
    allocation, leftover = optimizer.get_discrete_allocation(total_portfolio_value=100000)
    print("\nDiscrete allocation:")
    print(allocation)
    print(f"Funds remaining: ${leftover:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())