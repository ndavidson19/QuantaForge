import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, List, Callable, Union
from dataclasses import dataclass
import logging
import numba
from numba import njit, prange
import py_vollib_vectorized
import py_lets_be_rational
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import pickle
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ... (Previous OptionParams and AdvancedAmericanOptionPricer classes remain the same)

class HestonModel:
    def __init__(self, v0: float, kappa: float, theta: float, sigma: float, rho: float):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

    @njit(parallel=True)
    def simulate_paths(self, S0: float, T: float, r: float, steps: int, paths: int) -> Tuple[np.ndarray, np.ndarray]:
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.zeros((paths, steps + 1))
        v = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        v[:, 0] = self.v0
        
        for i in prange(paths):
            for j in range(1, steps + 1):
                z1 = np.random.normal(0, 1)
                z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.normal(0, 1)
                
                S[i, j] = S[i, j-1] * np.exp((r - 0.5 * v[i, j-1]) * dt + np.sqrt(v[i, j-1]) * sqrt_dt * z1)
                v[i, j] = v[i, j-1] + self.kappa * (self.theta - v[i, j-1]) * dt + self.sigma * np.sqrt(v[i, j-1]) * sqrt_dt * z2
                v[i, j] = max(v[i, j], 0)  # Ensure variance is non-negative
        
        return S, v

class MertonJumpDiffusionModel:
    def __init__(self, lambda_: float, mu_j: float, sigma_j: float):
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j

    @njit(parallel=True)
    def simulate_paths(self, S0: float, T: float, r: float, sigma: float, steps: int, paths: int) -> np.ndarray:
        dt = T / steps
        sqrt_dt = np.sqrt(dt)
        
        S = np.zeros((paths, steps + 1))
        S[:, 0] = S0
        
        for i in prange(paths):
            for j in range(1, steps + 1):
                z = np.random.normal(0, 1)
                N = np.random.poisson(self.lambda_ * dt)
                J = np.sum(np.random.normal(self.mu_j, self.sigma_j, N))
                
                S[i, j] = S[i, j-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * sqrt_dt * z + J)
        
        return S

class ParallelMonteCarloAmericanPricer:
    def __init__(self, model: Union[HestonModel, MertonJumpDiffusionModel], num_processes: int = mp.cpu_count()):
        self.model = model
        self.num_processes = num_processes

    def price(self, S0: float, K: float, T: float, r: float, paths: int, steps: int, option_type: str) -> float:
        with mp.Pool(self.num_processes) as pool:
            paths_per_process = paths // self.num_processes
            results = pool.map(partial(self._simulate_and_price, S0, K, T, r, steps, option_type), 
                               [paths_per_process] * self.num_processes)
        
        return np.mean(results)

    def _simulate_and_price(self, S0: float, K: float, T: float, r: float, steps: int, option_type: str, paths: int) -> float:
        if isinstance(self.model, HestonModel):
            S, _ = self.model.simulate_paths(S0, T, r, steps, paths)
        else:  # MertonJumpDiffusionModel
            S = self.model.simulate_paths(S0, T, r, self.model.sigma, steps, paths)
        
        dt = T / steps
        discount_factor = np.exp(-r * dt)
        
        if option_type == 'call':
            payoff = np.maximum(S - K, 0)
        else:  # put
            payoff = np.maximum(K - S, 0)
        
        option_value = np.zeros_like(payoff)
        option_value[:, -1] = payoff[:, -1]
        
        for i in range(steps - 1, 0, -1):
            expected_value = np.mean(option_value[:, i+1])
            exercise_value = payoff[:, i]
            option_value[:, i] = np.maximum(exercise_value, expected_value * discount_factor)
        
        return np.mean(option_value[:, 0])

class CacheManager:
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)

    def get(self, key: str) -> Any:
        value = self.redis_client.get(key)
        if value:
            return pickle.loads(value)
        return None

    def set(self, key: str, value: Any, expiration: int = 3600):
        self.redis_client.setex(key, expiration, pickle.dumps(value))

class AdvancedAmericanOptionPricingSystem:
    def __init__(self, api_key: str):
        self.qf = QuantForgeAPI(api_key)
        self.pde_pricer = AdvancedAmericanOptionPricer()
        self.calibrator = MarketDataCalibrator()
        self.greeks_calculator = AdvancedGreeksCalculator(self.pde_pricer)
        self.model_risk_analyzer = ModelRiskAnalyzer(self.pde_pricer)
        self.cache_manager = CacheManager()
        
        # Initialize stochastic volatility and jump-diffusion models
        self.heston_model = HestonModel(v0=0.04, kappa=2, theta=0.04, sigma=0.3, rho=-0.7)
        self.merton_model = MertonJumpDiffusionModel(lambda_=1, mu_j=-0.1, sigma_j=0.1)
        
        self.mc_pricer_heston = ParallelMonteCarloAmericanPricer(self.heston_model)
        self.mc_pricer_merton = ParallelMonteCarloAmericanPricer(self.merton_model)

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        cache_key = f"market_data_{symbol}"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            data = await self.qf.get_market_data(symbol)
            self.cache_manager.set(cache_key, data)
            return data
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {str(e)}")
            raise

    async def price_option(self, symbol: str, strike: float, expiry: float, option_type: str, model: str = 'pde') -> Dict[str, Any]:
        cache_key = f"option_price_{symbol}_{strike}_{expiry}_{option_type}_{model}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            return cached_result

        market_data = await self.get_market_data(symbol)
        
        local_vol = self.calibrator.calibrate_local_volatility(market_data['volatility_surface'])
        yield_curve = self.calibrator.calibrate_yield_curve(market_data['yield_curve'])
        dividend_yield = self.calibrator.calibrate_yield_curve(market_data['dividend_yield_curve'])
        
        params = OptionParams(
            S=market_data['spot_price'],
            K=strike,
            T=expiry,
            r=yield_curve,
            q=dividend_yield,
            sigma=local_vol,
            option_type=option_type,
            discrete_dividends=market_data['discrete_dividends']
        )
        
        try:
            if model == 'pde':
                price = self.pde_pricer.price(params)
                greeks = self.greeks_calculator.calculate_greeks(params)
            elif model == 'heston':
                price = self.mc_pricer_heston.price(params.S, params.K, params.T, params.r(0), 10000, 100, params.option_type)
                greeks = None  # Monte Carlo Greeks calculation would be computationally expensive
            elif model == 'merton':
                price = self.mc_pricer_merton.price(params.S, params.K, params.T, params.r(0), 10000, 100, params.option_type)
                greeks = None
            else:
                raise ValueError(f"Unknown model: {model}")

            model_risk = self.model_risk_analyzer.analyze_model_risk(params)
            
            result = {
                'price': price,
                'greeks': greeks,
                'model_risk': model_risk
            }
            
            self.cache_manager.set(cache_key, result)
            return result
        except Exception as e:
            logging.error(f"Error pricing option: {str(e)}")
            raise

    async def run(self):
        logging.info("Starting advanced American option pricing system...")
        while True:
            try:
                # Example: Price an American call option on AAPL with strike $150 expiring in 30 days
                for model in ['pde', 'heston', 'merton']:
                    result = await self.price_option('AAPL', 150, 30/365, 'call', model)
                    
                    logging.info(f"AAPL $150 Call (30 days to expiry) - {model.upper()} Model:")
                    logging.info(f"Price: ${result['price']:.2f}")
                    if result['greeks']:
                        logging.info("Greeks:")
                        for greek, value in result['greeks'].items():
                            logging.info(f"  {greek.capitalize()}: {value:.6f}")
                    logging.info("Model Risk Analysis:")
                    for metric, value in result['model_risk'].items():
                        logging.info(f"  {metric.replace('_', ' ').capitalize()}: {value:.6f}")
                    logging.info("---")

            except Exception as e:
                logging.error(f"Error in pricing loop: {str(e)}")

            await asyncio.sleep(5)  # Wait for 5 seconds before next pricing

# FastAPI setup
app = FastAPI()

class OptionRequest(BaseModel):
    symbol: str
    strike: float
    expiry: float
    option_type: str
    model: str = 'pde'

@app.post("/price_option")
async def price_option_api(option_request: OptionRequest):
    pricing_system = app.state.pricing_system
    try:
        result = await pricing_system.price_option(
            option_request.symbol,
            option_request.strike,
            option_request.expiry,
            option_request.option_type,
            option_request.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Usage
async def main():
    api_key = "your_quantforge_api_key"
    pricing_system = AdvancedAmericanOptionPricingSystem(api_key)
    
    # Set up FastAPI app
    app.state.pricing_system = pricing_system
    
    # Run the pricing system in the background
    pricing_task = asyncio.create_task(pricing_system.run())
    
    # Start the FastAPI server
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())