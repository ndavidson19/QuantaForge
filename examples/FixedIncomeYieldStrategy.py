from quantaforge.strategy import Strategy
from quantaforge.generics import Indicator, Signal
import logging
import polars as pl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YieldCurveIndicator(Indicator):
    def __init__(self, name="YieldCurve"):
        super().__init__(name)

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Implement yield curve calculation logic here
        # This is a placeholder implementation
        return data

class BondYieldIndicator(Indicator):
    def __init__(self, bond: str):
        super().__init__(f"{bond}_Yield")
        self.bond = bond

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Implement bond yield calculation logic here
        # This is a placeholder implementation
        return data

class BondDurationIndicator(Indicator):
    def __init__(self, bond: str):
        super().__init__(f"{bond}_Duration")
        self.bond = bond

    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        # Implement bond duration calculation logic here
        # This is a placeholder implementation
        return data

class FixedIncomeYieldStrategy(Strategy):
    def __init__(self, bond_universe, target_duration=5, max_position_size=0.1, 
                 min_credit_rating='BBB', rebalance_frequency='monthly'):
        super().__init__("Fixed Income Yield Strategy")
        self.bond_universe = bond_universe
        self.target_duration = target_duration
        self.max_position_size = max_position_size
        self.min_credit_rating = min_credit_rating
        self.rebalance_frequency = rebalance_frequency
        
        # Initialize indicators
        self.add_indicator("Treasury_Curve", YieldCurveIndicator())
        for bond in bond_universe:
            self.add_indicator(f"{bond}_Yield", BondYieldIndicator(bond))
            self.add_indicator(f"{bond}_Duration", BondDurationIndicator(bond))

    def analyze_yield_curve(self, data):
        curve = data["Treasury_Curve"].iloc[-1]
        slope_2_10 = curve['10Y'] - curve['2Y']
        slope_3m_10y = curve['10Y'] - curve['3M']
        
        return {
            'slope_2_10': slope_2_10,
            'slope_3m_10y': slope_3m_10y,
            'is_inverted': slope_2_10 < 0 or slope_3m_10y < 0
        }

    def calculate_yield_spreads(self, data):
        treasury_10y = data["Treasury_Curve"].iloc[-1]['10Y']
        spreads = {}
        for bond in self.bond_universe:
            if self.get_bond_type(bond) != 'Treasury':
                bond_yield = data[f"{bond}_Yield"].iloc[-1]
                spreads[bond] = bond_yield - treasury_10y
        return spreads
    
    def rank_bonds(self, data):
        rankings = []
        for bond in self.bond_universe:
            ytm = data[f"{bond}_Yield"].iloc[-1]
            duration = data[f"{bond}_Duration"].iloc[-1]
            credit_rating = self.get_credit_rating(bond)
            
            if self.rating_to_numeric(credit_rating) <= self.rating_to_numeric(self.min_credit_rating):
                rankings.append((bond, ytm, duration, credit_rating))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)  # Sort by YTM, descending


    def generate_signals(self, data):
        logging.info(f"Generating signals for date: {data.index[-1]}")
        
        if not self.is_rebalance_day(data.index[-1]):
            logging.info("Not a rebalance day, no signals generated.")
            return {}

        curve_analysis = self.analyze_yield_curve(data)
        yield_spreads = self.calculate_yield_spreads(data)
        ranked_bonds = self.rank_bonds(data)
        
        logging.info(f"Curve analysis: {curve_analysis}")
        logging.info(f"Yield spreads: {yield_spreads}")
        logging.info(f"Ranked bonds: {ranked_bonds}")
        
        signals = {}
        portfolio_duration = 0
        allocated_weight = 0

        for bond, ytm, duration, _ in ranked_bonds:
            if allocated_weight >= 1.0 or portfolio_duration >= self.target_duration:
                break
            
            # Determine position size based on remaining duration budget
            remaining_duration = max(0, self.target_duration - portfolio_duration)
            position_size = min(self.max_position_size, remaining_duration / duration)
            position_size = min(position_size, 1.0 - allocated_weight)
            
            if position_size > 0:
                signals[bond] = pl.DataFrame({
                    f"{bond}_buy": pl.Series([position_size] * len(data)),
                    f"{bond}_ytm": pl.Series([ytm] * len(data)),
                    f"{bond}_duration": pl.Series([duration] * len(data))
                })
                portfolio_duration += position_size * duration
                allocated_weight += position_size
            
            logging.info(f"Signal generated for {bond}: position size = {position_size}")

        logging.info(f"Generated signals: {signals}")
        return signals

    def manage_risk(self, data):
        signals = {}
        portfolio_duration = self.calculate_portfolio_duration(data)
        
        if abs(portfolio_duration - self.target_duration) > 0.5:
            # Adjust portfolio duration if it deviates too much from the target
            for bond in self.get_positions():
                bond_duration = data[f"{bond}_Duration"].iloc[-1]
                position_size = self.get_position_size(bond)
                
                if portfolio_duration > self.target_duration:
                    # Reduce duration by selling longer-duration bonds
                    if bond_duration > self.target_duration:
                        adjustment = min(position_size * 0.1, portfolio_duration - self.target_duration)
                        signals[bond] = pl.DataFrame({
                            f"{bond}_sell": pl.Series([adjustment] * len(data)),
                            f"{bond}_duration": pl.Series([bond_duration] * len(data))
                        })
                else:
                    # Increase duration by buying longer-duration bonds
                    if bond_duration > self.target_duration:
                        adjustment = min(self.max_position_size - position_size, self.target_duration - portfolio_duration)
                        signals[bond] = pl.DataFrame({
                            f"{bond}_buy": pl.Series([adjustment] * len(data)),
                            f"{bond}_duration": pl.Series([bond_duration] * len(data))
                        })
        
        logging.info(f"Risk management signals: {signals}")
        return signals

    def calculate_portfolio_duration(self, data):
        portfolio_duration = 0
        for bond in self.get_positions():
            bond_duration = data[f"{bond}_Duration"].iloc[-1]
            position_size = self.get_position_size(bond)
            portfolio_duration += bond_duration * position_size
        return portfolio_duration

    def is_rebalance_day(self, date):
        # For testing purposes, always return True
        # In a real scenario, you'd implement the actual rebalance day logic
        return True


    def get_bond_type(self, bond):
        # Placeholder implementation
        return 'Treasury' if bond.startswith('T') else 'Corporate'

    def get_credit_rating(self, bond):
        # Placeholder implementation
        return 'AAA' if bond.startswith('T') else 'A'

    def rating_to_numeric(self, rating):
        # Placeholder implementation
        rating_scale = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4, 'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10}
        return rating_scale.get(rating, 11)  # Return 11 for unknown ratings

    def get_positions(self):
        # Placeholder implementation
        return self.bond_universe

    def get_position_size(self, bond):
        # Placeholder implementation
        return 0.1  # Assume equal weight for simplicity