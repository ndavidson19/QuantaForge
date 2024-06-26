from quantforge import Strategy, Indicator, Signal
from quantforge.portfolio import Portfolio
from quantforge.risk import RiskModel
from quantforge.optimization import PortfolioOptimizer
from quantforge.derivatives import Option, Future
import numpy as np
import pandas as pd

class DynamicPortfolioOptimizer(Strategy):
    def __init__(self, user_portfolio, risk_tolerance, income_preference, 
                 esg_preference, rebalance_frequency='monthly'):
        super().__init__("Dynamic Portfolio Optimizer")
        self.user_portfolio = user_portfolio
        self.risk_tolerance = risk_tolerance  # 1-10 scale
        self.income_preference = income_preference  # 0-1 scale
        self.esg_preference = esg_preference  # 0-1 scale
        self.rebalance_frequency = rebalance_frequency
        
        self.risk_model = RiskModel()
        self.portfolio_optimizer = PortfolioOptimizer()
        
        # Initialize indicators
        self.add_indicator(Indicator("MultiFactorRank", {}, name="FactorRank"))
        self.add_indicator(Indicator("DividendYield", {}, name="DivYield"))
        self.add_indicator(Indicator("ESGScore", {}, name="ESGScore"))
        self.add_indicator(Indicator("ImpliedVolatility", {}, name="IV"))
        
    def analyze_portfolio(self, data):
        current_weights = self.user_portfolio.get_weights()
        risk_metrics = self.risk_model.calculate_risk_metrics(self.user_portfolio, data)
        factor_exposures = self.calculate_factor_exposures(self.user_portfolio, data)
        income_metrics = self.calculate_income_metrics(self.user_portfolio, data)
        
        return {
            'current_weights': current_weights,
            'risk_metrics': risk_metrics,
            'factor_exposures': factor_exposures,
            'income_metrics': income_metrics
        }
    
    def optimize_portfolio(self, data, analysis):
        target_risk = self.map_risk_tolerance(self.risk_tolerance)
        constraints = self.generate_constraints(analysis)
        
        optimized_weights = self.portfolio_optimizer.optimize(
            data, 
            target_risk=target_risk,
            income_preference=self.income_preference,
            esg_preference=self.esg_preference,
            constraints=constraints
        )
        
        return optimized_weights
    
    def generate_hedging_signals(self, data, analysis):
        signals = {}
        portfolio_beta = analysis['risk_metrics']['beta']
        
        if portfolio_beta > 1.2:  # If portfolio is too exposed to market risk
            spy_future = Future("ES", data["ES"].iloc[-1], 30)  # E-mini S&P 500 future
            hedge_size = (portfolio_beta - 1) * self.user_portfolio.total_value / spy_future.multiplier
            signals['ES'] = (Signal.SELL, spy_future, hedge_size)
        
        # Options hedge for tail risk
        if analysis['risk_metrics']['var_95'] > 0.1:  # If 95% VaR is greater than 10%
            spy_put = Option("SPY", data["SPY"].iloc[-1] * 0.9, 'put', 30)  # 10% OTM put
            signals['SPY_put'] = (Signal.BUY, spy_put, 0.01)  # Hedge 1% of portfolio value
        
        return signals
    
    def generate_alpha_signals(self, data):
        signals = {}
        factor_ranks = data["FactorRank"]
        top_ranks = factor_ranks.nlargest(5)
        bottom_ranks = factor_ranks.nsmallest(5)
        
        for asset in top_ranks.index:
            signals[asset] = (Signal.BUY, 0.02)  # Allocate 2% to each top-ranked asset
        
        for asset in bottom_ranks.index:
            signals[asset] = (Signal.SELL, 0.02)  # Short 2% of each bottom-ranked asset
        
        return signals
    
    def generate_income_signals(self, data):
        signals = {}
        high_yield_assets = data["DivYield"].nlargest(10)
        
        for asset, yield_ in high_yield_assets.items():
            if yield_ > 0.04:  # If yield is above 4%
                signals[asset] = (Signal.BUY, 0.03)  # Allocate 3% to high-yield assets
        
        # Covered call writing on high IV stocks
        for asset in self.user_portfolio.positions:
            if data[f"{asset}_IV"].iloc[-1] > 0.3:  # If IV is above 30%
                current_price = data[asset].iloc[-1]
                call_option = Option(asset, current_price * 1.05, 'call', 30)  # 5% OTM call
                signals[f"{asset}_call"] = (Signal.SELL, call_option, 1)
        
        return signals
    
    def generate_signals(self, data):
        analysis = self.analyze_portfolio(data)
        optimized_weights = self.optimize_portfolio(data, analysis)
        
        signals = {}
        
        # Core portfolio rebalancing
        for asset, target_weight in optimized_weights.items():
            current_weight = self.user_portfolio.get_weight(asset)
            if abs(current_weight - target_weight) > 0.01:  # 1% threshold
                if current_weight < target_weight:
                    signals[asset] = (Signal.BUY, target_weight - current_weight)
                else:
                    signals[asset] = (Signal.SELL, current_weight - target_weight)
        
        # Add hedging signals
        signals.update(self.generate_hedging_signals(data, analysis))
        
        # Add alpha generation signals
        signals.update(self.generate_alpha_signals(data))
        
        # Add income generation signals
        signals.update(self.generate_income_signals(data))
        
        return signals
    
    def map_risk_tolerance(self, risk_tolerance):
        # Map 1-10 scale to target volatility
        return (risk_tolerance / 10) * 0.2  # Max target volatility of 20%
    
    def generate_constraints(self, analysis):
        constraints = {
            'max_weight': 0.2,  # No single asset can be more than 20% of the portfolio
            'min_weight': -0.1,  # Allow for up to 10% short positions
            'sector_constraints': {
                'Technology': (0.1, 0.3),  # Min 10%, Max 30% in Technology
                'Finance': (0.05, 0.2),    # Min 5%, Max 20% in Finance
                # Add more sector constraints as needed
            },
            'esg_constraint': self.esg_preference  # Minimum ESG score
        }
        return constraints
    
    def calculate_factor_exposures(self, portfolio, data):
        # Implement factor exposure calculation
        pass
    
    def calculate_income_metrics(self, portfolio, data):
        # Implement income metrics calculation
        pass
    
    def on_user_input(self, input_type, value):
        if input_type == 'risk_tolerance':
            self.risk_tolerance = value
        elif input_type == 'income_preference':
            self.income_preference = value
        elif input_type == 'esg_preference':
            self.esg_preference = value
        # Trigger rebalance after user input
        self.rebalance()
    
    def rebalance(self):
        # Implement logic to trigger a portfolio rebalance
        pass

from quantforge.optimization import PortfolioOptimizer
import cvxpy as cp
import numpy as np
import pandas as pd

class EnhancedPortfolioOptimizer(PortfolioOptimizer):
    def __init__(self):
        super().__init__()
    
    def optimize(self, data, target_risk, income_preference, esg_preference, constraints):
        returns = data.pct_change().dropna()
        cov_matrix = returns.cov()
        expected_returns = self.estimate_expected_returns(returns)
        dividend_yields = data['DivYield'].iloc[-1]
        esg_scores = data['ESGScore'].iloc[-1]
        
        n_assets = len(returns.columns)
        weights = cp.Variable(n_assets)
        
        risk = cp.quad_form(weights, cov_matrix)
        returns = expected_returns.T @ weights
        income = dividend_yields.T @ weights
        esg = esg_scores.T @ weights
        
        objective = cp.Maximize(returns - 0.5 * risk + income_preference * income + esg_preference * esg)
        
        constraints = [
            cp.sum(weights) == 1,
            weights >= constraints['min_weight'],
            weights <= constraints['max_weight'],
            risk <= target_risk**2
        ]
        
        for sector, (min_weight, max_weight) in constraints['sector_constraints'].items():
            sector_weights = weights[data['Sector'] == sector]
            constraints.append(cp.sum(sector_weights) >= min_weight)
            constraints.append(cp.sum(sector_weights) <= max_weight)
        
        constraints.append(esg >= constraints['esg_constraint'])
        
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status == 'optimal':
            return pd.Series(weights.value, index=returns.columns)
        else:
            raise ValueError("Optimization failed to converge")
    
    def estimate_expected_returns(self, historical_returns):
        # Implement expected returns estimation (e.g., using CAPM, Black-Litterman, or factor models)
        pass