import asyncio
from typing import Dict, List
import pandas as pd
from scipy.stats import norm

class OptionMarketMaker:
    def __init__(self, pricing_system: AdvancedAmericanOptionPricingSystem, symbols: List[str], 
                 max_position: int = 100, spread_multiplier: float = 1.5, 
                 mispricing_threshold: float = 0.02):
        self.pricing_system = pricing_system
        self.symbols = symbols
        self.max_position = max_position
        self.spread_multiplier = spread_multiplier
        self.mispricing_threshold = mispricing_threshold
        self.positions = {symbol: 0 for symbol in symbols}
        self.pnl = 0.0

    async def fetch_option_chain(self, symbol: str) -> pd.DataFrame:
        # This method should fetch the current option chain from your data provider
        # For this example, we'll create a mock option chain
        market_data = await self.pricing_system.get_market_data(symbol)
        S = market_data['spot_price']
        r = market_data['yield_curve']['rates'][0]  # Assume short-term rate
        sigma = market_data['volatility_surface']['implied_volatility'][0]  # ATM vol

        expiries = [30/365, 60/365, 90/365]
        strikes = [0.9*S, S, 1.1*S]
        
        options = []
        for expiry in expiries:
            for strike in strikes:
                for option_type in ['call', 'put']:
                    # Use Black-Scholes for quick pricing of European options
                    d1 = (np.log(S/strike) + (r + 0.5*sigma**2)*expiry) / (sigma*np.sqrt(expiry))
                    d2 = d1 - sigma*np.sqrt(expiry)
                    
                    if option_type == 'call':
                        price = S*norm.cdf(d1) - strike*np.exp(-r*expiry)*norm.cdf(d2)
                    else:
                        price = strike*np.exp(-r*expiry)*norm.cdf(-d2) - S*norm.cdf(-d1)
                    
                    options.append({
                        'symbol': symbol,
                        'expiry': expiry,
                        'strike': strike,
                        'type': option_type,
                        'bid': price * 0.99,  # Simulated bid-ask spread
                        'ask': price * 1.01
                    })
        
        return pd.DataFrame(options)

    async def price_option(self, symbol: str, strike: float, expiry: float, option_type: str) -> float:
        result = await self.pricing_system.price_option(symbol, strike, expiry, option_type)
        return result['price']

    def calculate_spread(self, model_price: float) -> Tuple[float, float]:
        half_spread = self.spread_multiplier * model_price * self.mispricing_threshold
        return model_price - half_spread, model_price + half_spread

    async def make_markets(self):
        while True:
            for symbol in self.symbols:
                option_chain = await self.fetch_option_chain(symbol)
                
                for _, option in option_chain.iterrows():
                    model_price = await self.price_option(option['symbol'], option['strike'], 
                                                          option['expiry'], option['type'])
                    
                    model_bid, model_ask = self.calculate_spread(model_price)
                    
                    if model_bid > option['ask'] and self.positions[symbol] < self.max_position:
                        # Our model thinks the option is underpriced, so we buy
                        quantity = min(10, self.max_position - self.positions[symbol])
                        self.positions[symbol] += quantity
                        self.pnl -= quantity * option['ask']
                        print(f"Bought {quantity} {option['symbol']} {option['type']} @ {option['ask']}")
                    
                    elif model_ask < option['bid'] and self.positions[symbol] > -self.max_position:
                        # Our model thinks the option is overpriced, so we sell
                        quantity = min(10, self.max_position + self.positions[symbol])
                        self.positions[symbol] -= quantity
                        self.pnl += quantity * option['bid']
                        print(f"Sold {quantity} {option['symbol']} {option['type']} @ {option['bid']}")
                
                # Hedge delta risk
                await self.hedge_delta(symbol)
            
            # Wait for a short period before next market-making cycle
            await asyncio.sleep(1)

    async def hedge_delta(self, symbol: str):
        total_delta = 0
        market_data = await self.pricing_system.get_market_data(symbol)
        spot_price = market_data['spot_price']
        
        option_chain = await self.fetch_option_chain(symbol)
        
        for _, option in option_chain.iterrows():
            if self.positions[symbol] != 0:
                result = await self.pricing_system.price_option(option['symbol'], option['strike'], 
                                                                option['expiry'], option['type'])
                option_delta = result['greeks']['delta']
                total_delta += self.positions[symbol] * option_delta
        
        # Hedge using the underlying
        hedge_quantity = -int(total_delta)
        if hedge_quantity > 0:
            print(f"Buying {hedge_quantity} shares of {symbol} @ {spot_price} for delta hedge")
            self.pnl -= hedge_quantity * spot_price
        elif hedge_quantity < 0:
            print(f"Selling {-hedge_quantity} shares of {symbol} @ {spot_price} for delta hedge")
            self.pnl += -hedge_quantity * spot_price

    async def run(self):
        print("Starting option market-making strategy...")
        await self.make_markets()

# Usage
async def main():
    api_key = "your_quantforge_api_key"
    pricing_system = AdvancedAmericanOptionPricingSystem(api_key)
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    market_maker = OptionMarketMaker(pricing_system, symbols)
    await market_maker.run()

if __name__ == "__main__":
    asyncio.run(main())