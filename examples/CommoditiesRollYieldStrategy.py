from forge import Strategy, Indicator, Signal
from forge.data import FuturesDataManager
import pandas as pd
import numpy as np

class CommoditiesRollYieldStrategy(Strategy):
    def __init__(self, commodities, lookback_period=30, rebalance_frequency='monthly',
                 max_position_size=0.1, use_short_positions=True):
        super().__init__("Commodities Roll Yield Strategy")
        self.commodities = commodities
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.max_position_size = max_position_size
        self.use_short_positions = use_short_positions
        
        self.futures_data_manager = FuturesDataManager()
        
        # Initialize indicators
        for commodity in commodities:
            self.add_indicator(Indicator("RollYield", {"commodity": commodity}, name=f"{commodity}_RollYield"))
            self.add_indicator(Indicator("FuturesCurve", {"commodity": commodity}, name=f"{commodity}_Curve"))

    def calculate_roll_yield(self, near_price, far_price, days_between):
        return (np.log(near_price) - np.log(far_price)) * (365 / days_between)

    def analyze_futures_curve(self, data, commodity):
        curve = data[f"{commodity}_Curve"].iloc[-1]
        near_price, next_price = curve['near'], curve['next']
        days_between = (curve['next_expiry'] - curve['near_expiry']).days
        
        roll_yield = self.calculate_roll_yield(near_price, next_price, days_between)
        curve_state = "Backwardation" if roll_yield > 0 else "Contango"
        
        return {
            'roll_yield': roll_yield,
            'curve_state': curve_state,
            'near_price': near_price,
            'next_price': next_price,
            'days_to_expiry': (curve['near_expiry'] - pd.Timestamp.now()).days
        }

    def rank_commodities(self, data):
        rankings = []
        for commodity in self.commodities:
            analysis = self.analyze_futures_curve(data, commodity)
            rankings.append((commodity, analysis['roll_yield']))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def generate_signals(self, data):
        if not self.is_rebalance_day(data.index[-1]):
            return {}

        rankings = self.rank_commodities(data)
        signals = {}
        
        # Long positions for commodities in backwardation
        for commodity, roll_yield in rankings[:3]:  # Top 3 positive roll yields
            if roll_yield > 0:
                signals[commodity] = (Signal.BUY, self.max_position_size)
        
        # Short positions for commodities in contango (if enabled)
        if self.use_short_positions:
            for commodity, roll_yield in rankings[-3:]:  # Bottom 3 negative roll yields
                if roll_yield < 0:
                    signals[commodity] = (Signal.SELL, self.max_position_size)
        
        return signals

    def manage_risk(self, data):
        for commodity in self.commodities:
            position = self.get_position(commodity)
            if position:
                analysis = self.analyze_futures_curve(data, commodity)
                
                # Close position if roll yield changes sign
                if (position.direction == 'long' and analysis['roll_yield'] < 0) or \
                   (position.direction == 'short' and analysis['roll_yield'] > 0):
                    return {commodity: (Signal.CLOSE, position.size)}
                
                # Roll to next contract if close to expiry
                if analysis['days_to_expiry'] <= 5:
                    return {commodity: (Signal.ROLL, position.size)}
        
        return {}

    def is_rebalance_day(self, date):
        if self.rebalance_frequency == 'monthly':
            return date.day == 1
        elif self.rebalance_frequency == 'weekly':
            return date.weekday() == 0
        else:
            return True  # Daily rebalancing

    def on_data(self, data):
        signals = self.generate_signals(data)
        risk_signals = self.manage_risk(data)
        signals.update(risk_signals)
        return signals
    


class RollYieldIndicator(Indicator):
    def __init__(self, params):
        super().__init__("RollYield", params)
        self.commodity = params['commodity']
        self.futures_data_manager = FuturesDataManager()

    def calculate(self, data):
        futures_data = self.futures_data_manager.get_futures_data(self.commodity)
        near_contract = futures_data['near_contract']
        next_contract = futures_data['next_contract']
        
        near_price = data[near_contract]
        next_price = data[next_contract]
        days_between = (futures_data['next_expiry'] - futures_data['near_expiry']).days
        
        roll_yield = (np.log(near_price) - np.log(next_price)) * (365 / days_between)
        return roll_yield

class FuturesCurveIndicator(Indicator):
    def __init__(self, params):
        super().__init__("FuturesCurve", params)
        self.commodity = params['commodity']
        self.futures_data_manager = FuturesDataManager()

    def calculate(self, data):
        futures_data = self.futures_data_manager.get_futures_data(self.commodity)
        return {
            'near': data[futures_data['near_contract']],
            'next': data[futures_data['next_contract']],
            'far': data[futures_data['far_contract']],
            'near_expiry': futures_data['near_expiry'],
            'next_expiry': futures_data['next_expiry'],
            'far_expiry': futures_data['far_expiry']
        }

# Register custom indicators
from quantforge import register_indicator
register_indicator("RollYield", RollYieldIndicator)
register_indicator("FuturesCurve", FuturesCurveIndicator)
```