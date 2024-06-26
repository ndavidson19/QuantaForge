import numpy as np
import pandas as pd
from numba import jit

class SimulationEngine:
    def __init__(self, strategy: Strategy, initial_capital: float):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.portfolio = {"cash": initial_capital}
        for asset in strategy.assets:
            self.portfolio[asset] = 0

    @jit(nopython=True)
    def simulate(self, market_data: Dict[str, pd.DataFrame], params: Dict[str, Any]):
        results = []
        for date, data in market_data.items():
            portfolio_value = self.calculate_portfolio_value(data)
            signals = self.generate_signals(data)
            self.execute_trades(signals, data, portfolio_value)
            results.append({
                "date": date,
                "portfolio_value": portfolio_value,
                "positions": self.portfolio.copy()
            })
        return results

    def generate_signals(self, data: pd.DataFrame):
        signals = {}
        for condition in self.strategy.entry_conditions:
            if self.evaluate_condition(condition, data):
                signals["enter"] = True
        for condition in self.strategy.exit_conditions:
            if self.evaluate_condition(condition, data):
                signals["exit"] = True
        return signals

    def execute_trades(self, signals: Dict[str, bool], data: pd.DataFrame, portfolio_value: float):
        if signals.get("enter"):
            for action in self.strategy.actions:
                if action.type in ["buy", "buy_option"]:
                    quantity = self.calculate_quantity(action.quantity, data, portfolio_value)
                    cost = quantity * data[action.asset]["price"]
                    if cost <= self.portfolio["cash"]:
                        self.portfolio[action.asset] += quantity
                        self.portfolio["cash"] -= cost
        elif signals.get("exit"):
            for action in self.strategy.actions:
                if action.type in ["sell", "sell_option"]:
                    quantity = self.calculate_quantity(action.quantity, data, portfolio_value)
                    revenue = quantity * data[action.asset]["price"]
                    self.portfolio[action.asset] -= quantity
                    self.portfolio["cash"] += revenue

    def calculate_quantity(self, quantity_expr: str, data: pd.DataFrame, portfolio_value: float):
        # Evaluate the quantity expression
        locals().update(data)
        locals()["portfolio"] = self.portfolio
        locals()["portfolio_value"] = portfolio_value
        return eval(quantity_expr)

    def calculate_portfolio_value(self, data: pd.DataFrame):
        return self.portfolio["cash"] + sum(self.portfolio[asset] * data[asset]["price"] for asset in self.strategy.assets if asset != "cash")


from typing import List, Dict, Any

class EnhancedMarketSimulator:
    def __init__(self, assets: List[str], start_date: str, end_date: str, params: Dict[str, Any], event_manager: EventManager):
        self.assets = assets
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.params = params
        self.event_manager = event_manager
        self.correlation_matrix = self.generate_correlation_matrix()

    # ... (previous methods remain the same)

    def apply_event_impacts(self, market_data: Dict[str, pd.DataFrame], event: Event):
        for impact in event.impacts:
            if impact.asset in market_data:
                asset_data = market_data[impact.asset]
                event_date_index = asset_data.index.get_loc(event.date)
                
                # Apply price impact
                asset_data.iloc[event_date_index:, asset_data.columns.get_loc('price')] *= (1 + impact.price_impact)
                
                # Apply volatility impact
                if 'volatility' in self.params:
                    old_vol = self.params[f"{impact.asset}_volatility"]
                    new_vol = old_vol * (1 + impact.volatility_impact)
                    new_returns = np.random.normal(0, new_vol, size=len(asset_data) - event_date_index)
                    new_prices = asset_data.iloc[event_date_index]['price'] * np.exp(np.cumsum(new_returns))
                    asset_data.iloc[event_date_index:, asset_data.columns.get_loc('price')] = new_prices

                # Apply event duration
                if event.duration > 1:
                    fade_factor = np.linspace(1, 0, event.duration)
                    price_impact_fade = impact.price_impact * fade_factor
                    asset_data.iloc[event_date_index:event_date_index+event.duration, asset_data.columns.get_loc('price')] *= (1 + price_impact_fade[:len(asset_data.iloc[event_date_index:event_date_index+event.duration])])

        return market_data

    def generate_market_data(self, num_simulations: int = 1000):
        all_simulations = []
        for _ in range(num_simulations):
            market_data = super().generate_market_data()
            
            for date in pd.date_range(self.start_date, self.end_date):
                events = self.event_manager.get_events_for_date(date)
                for event in events:
                    if np.random.random() < event.probability:
                        market_data = self.apply_event_impacts(market_data, event)
            
            all_simulations.append(market_data)

        return all_simulations

    def monte_carlo_simulation(self, strategy: 'Strategy', num_simulations: int = 1000):
        all_simulations = self.generate_market_data(num_simulations)
        results = []

        for sim_data in all_simulations:
            strategy_result = strategy.backtest(sim_data)
            results.append(strategy_result.total_return)

        return results
