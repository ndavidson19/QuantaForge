from quantforge import Strategy, Indicator, Signal
from quantforge.derivatives import Option
import numpy as np
from scipy.stats import beta
from typing import Dict, List
import asyncio

class SophisticatedOptionsStrategy(Strategy):
    def __init__(self, underlying, lookback_period=252, rebalance_frequency=5, 
                 initial_capital=1000000, risk_free_rate=0.02):
        super().__init__("Sophisticated Options Strategy")
        self.underlying = underlying
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.day_count = 0
        
        # Initialize Bayesian priors
        self.prior_alpha = 1
        self.prior_beta = 1
        
        # Initialize Kelly Criterion parameters
        self.win_rate = 0.5
        self.avg_win = 0
        self.avg_loss = 0
        
        # Initialize Monte Carlo parameters
        self.num_simulations = 10000
        self.var_threshold = 0.05
        
        # Initialize AdvancedAmericanOptionPricingSystem
        self.option_pricer = AdvancedAmericanOptionPricingSystem("your_quantforge_api_key")
        
        # Add indicators
        self.add_indicator(Indicator("ATR", {"window": 14}, name="ATR"))
        self.add_indicator(Indicator("RSI", {"window": 14}, name="RSI"))

    async def generate_signals(self, data):
        self.day_count += 1
        
        if self.day_count % self.rebalance_frequency != 0:
            return {}
        
        if len(data) < self.lookback_period:
            return {}
        
        current_price = data[self.underlying].iloc[-1]
        
        # Update Bayesian probabilities
        returns = data[self.underlying].pct_change().dropna()
        positive_returns = (returns > 0).sum()
        negative_returns = (returns <= 0).sum()
        self.prior_alpha += positive_returns
        self.prior_beta += negative_returns
        
        # Calculate probability of positive return
        prob_positive = beta.mean(self.prior_alpha, self.prior_beta)
        
        # Update Kelly Criterion parameters
        self.win_rate = positive_returns / (positive_returns + negative_returns)
        self.avg_win = returns[returns > 0].mean()
        self.avg_loss = abs(returns[returns <= 0].mean())
        
        # Calculate Kelly fraction
        kelly_fraction = self.win_rate - ((1 - self.win_rate) / (self.avg_win / self.avg_loss))
        kelly_fraction = max(0, min(kelly_fraction, 1))  # Limit to [0, 1]
        
        # Run Monte Carlo simulation
        simulated_returns = self.run_monte_carlo(data)
        var = np.percentile(simulated_returns, self.var_threshold * 100)
        
        # Determine option parameters
        atm_strike = round(current_price, 0)
        expiry = 30 / 365  # 30 days
        
        # Price options
        call_price = await self.price_option(self.underlying, atm_strike, expiry, 'call')
        put_price = await self.price_option(self.underlying, atm_strike, expiry, 'put')
        
        signals = {}
        
        if prob_positive > 0.55 and data["RSI"].iloc[-1] < 70:
            # Bullish sentiment: Buy call option
            position_size = kelly_fraction * self.initial_capital / call_price['price']
            signals[f"{self.underlying}_call_{atm_strike}"] = (Signal.BUY, position_size)
        elif prob_positive < 0.45 or data["RSI"].iloc[-1] > 70:
            # Bearish sentiment: Buy put option
            position_size = kelly_fraction * self.initial_capital / put_price['price']
            signals[f"{self.underlying}_put_{atm_strike}"] = (Signal.BUY, position_size)
        
        # Adjust position size based on VaR
        for option, (signal, size) in signals.items():
            if abs(var) > data["ATR"].iloc[-1]:
                signals[option] = (signal, size * (data["ATR"].iloc[-1] / abs(var)))
        
        return signals

    def run_monte_carlo(self, data):
        returns = data[self.underlying].pct_change().dropna()
        mu = returns.mean()
        sigma = returns.std()
        
        simulations = np.random.normal(mu, sigma, (self.lookback_period, self.num_simulations))
        return simulations.sum(axis=0)

    async def price_option(self, symbol: str, strike: float, expiry: float, option_type: str) -> Dict[str, float]:
        try:
            result = await self.option_pricer.price_option(symbol, strike, expiry, option_type)
            return result
        except Exception as e:
            self.logger.error(f"Error pricing option: {str(e)}")
            return {'price': np.nan, 'greeks': {}, 'model_risk': {}}

    async def on_data(self, data):
        return await self.generate_signals(data)

# Usage
async def main():
    strategy = SophisticatedOptionsStrategy("SPY")
    api = QuantForgeAPI("your_quantforge_api_key")
    
    while True:
        try:
            # Fetch latest market data
            market_data = await api.get_market_data("SPY")
            
            # Generate signals
            signals = await strategy.on_data(market_data)
            
            if signals:
                print("Generated signals:")
                for option, (signal, size) in signals.items():
                    print(f"{option}: {signal.name} {size:.2f} contracts")
            
            # In a real system, you would execute these signals here
            
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
        
        await asyncio.sleep(strategy.rebalance_frequency * 60)  # Wait for rebalance period

if __name__ == "__main__":
    asyncio.run(main())