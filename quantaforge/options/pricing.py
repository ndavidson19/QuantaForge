import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.interpolate import CubicSpline
from typing import Dict, Tuple, List, Callable, Any
from dataclasses import dataclass
import logging
import numba
from numba import njit
import py_vollib_vectorized
import py_lets_be_rational
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from quantforge import QuantForgeAPI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class OptionParams:
    S: float
    K: float
    T: float
    r: Callable[[float], float]  # Time-dependent interest rate
    q: Callable[[float], float]  # Time-dependent dividend yield
    sigma: Callable[[float, float], float]  # Local volatility function
    option_type: str
    discrete_dividends: List[Tuple[float, float]]  # List of (time, amount) tuples

class AdvancedAmericanOptionPricer:
    def __init__(self, S_max: float = 400, M: int = 400, N: int = 2000):
        self.S_max = S_max
        self.M = M
        self.N = N

    def _setup_grid(self, K: float) -> Tuple[np.ndarray, np.ndarray]:
        # Use a non-uniform grid with concentration around the strike
        S_center = np.linspace(0.5*K, 1.5*K, self.M//2)
        S_left = np.linspace(0, 0.5*K, self.M//4+1)[:-1]
        S_right = np.linspace(1.5*K, self.S_max, self.M//4+1)[1:]
        S = np.concatenate([S_left, S_center, S_right])
        dS = np.diff(S)
        return S, dS

    def _setup_time_grid(self, T: float, discrete_dividends: List[Tuple[float, float]]) -> np.ndarray:
        # Create a non-uniform time grid with points at dividend dates
        div_times = [div[0] for div in discrete_dividends]
        t = np.sort(np.unique(np.concatenate([
            np.linspace(0, T, self.N+1),
            div_times
        ])))[::-1]  # Reverse for backward time
        return t

    @staticmethod
    @njit
    def _create_coefficient_matrices(S: np.ndarray, dS: np.ndarray, dt: float, r: float, q: float, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        M = len(S) - 1
        alpha = 0.5 * sigma[1:M]**2 * S[1:M]**2 / dS[:-1] / (dS[:-1] + dS[1:])
        beta = -0.5 * sigma[1:M]**2 * S[1:M]**2 / dS[:-1] / dS[1:] - (r - q)
        gamma = 0.5 * sigma[1:M]**2 * S[1:M]**2 / dS[1:] / (dS[:-1] + dS[1:]) + (r - q) * S[1:M] / (dS[:-1] + dS[1:])
        
        A = sp.diags([alpha[1:], beta, gamma[:-1]], [-1, 0, 1], shape=(M-1, M-1)).tocsr()
        B = sp.diags([-1.5, 2, -0.5], [-1, 0, 1], shape=(M-1, M-1)).tocsr() / dt
        C = sp.diags([1/3, 4/3, 1/3], [-1, 0, 1], shape=(M-1, M-1)).tocsr() / dt
        
        return A.data, A.indices, A.indptr, B.data, B.indices, B.indptr, C.data, C.indices, C.indptr

    @staticmethod
    @njit
    def _apply_penalty(V: np.ndarray, payoff: np.ndarray, penalty: float) -> np.ndarray:
        return np.maximum(V, payoff) + penalty * np.maximum(payoff - V, 0)**2

    def price(self, params: OptionParams) -> float:
        S, dS = self._setup_grid(params.K)
        t = self._setup_time_grid(params.T, params.discrete_dividends)
        
        M = len(S) - 1
        N = len(t) - 1
        
        # Initialize grid
        V = np.zeros((M+1, N+1))
        
        # Set up payoff
        if params.option_type == 'call':
            payoff = np.maximum(S - params.K, 0)
        else:  # put
            payoff = np.maximum(params.K - S, 0)
        
        V[:, -1] = payoff
        
        # Pre-compute coefficient matrices
        A_data, A_indices, A_indptr, B_data, B_indices, B_indptr, C_data, C_indices, C_indptr = \
            self._create_coefficient_matrices(S, dS, t[0]-t[1], params.r(0), params.q(0), params.sigma(S, 0))
        
        A = sp.csr_matrix((A_data, A_indices, A_indptr))
        B = sp.csr_matrix((B_data, B_indices, B_indptr))
        C = sp.csr_matrix((C_data, C_indices, C_indptr))
        
        penalty = 1e6  # Large penalty for constraint violation
        
        # BDF2 time-stepping with penalty method
        for j in range(N-1, -1, -1):
            r = params.r(t[j])
            q = params.q(t[j])
            sigma = params.sigma(S, t[j])
            
            # Update coefficient matrices if necessary
            if j < N-1:
                A_data, A_indices, A_indptr, _, _, _, _, _, _ = \
                    self._create_coefficient_matrices(S, dS, t[j]-t[j+1], r, q, sigma)
                A = sp.csr_matrix((A_data, A_indices, A_indptr))
            
            if j == N-1:
                rhs = C @ V[1:M, j+1]
            else:
                rhs = C @ V[1:M, j+1] - (1/3) * B @ V[1:M, j+2]
            
            # Apply penalty method
            def objective(v):
                return np.sum((B - (2/3) * A) @ v - rhs) + np.sum(self._apply_penalty(v, payoff[1:M], penalty))
            
            # Use a nonlinear solver (e.g., L-BFGS-B) to minimize the objective
            V[1:M, j] = scipy.optimize.minimize(objective, V[1:M, j+1], method='L-BFGS-B').x
            
            # Apply discrete dividends
            for div_time, div_amount in params.discrete_dividends:
                if t[j+1] <= div_time < t[j]:
                    S_ex = S - div_amount
                    V[:, j] = np.interp(S, S_ex, V[:, j])
        
        # Interpolate to get option price
        return np.interp(params.S, S, V[:, 0])

class MarketDataCalibrator:
    def __init__(self):
        self.gp_model = GaussianProcessRegressor(kernel=RBF() + WhiteKernel(), alpha=1e-10)

    def calibrate_local_volatility(self, market_data: Dict[str, np.ndarray]) -> Callable[[float, float], float]:
        X = np.column_stack([market_data['moneyness'], market_data['time_to_maturity']])
        y = market_data['implied_volatility']
        self.gp_model.fit(X, y)
        
        def local_vol(S: float, t: float) -> float:
            moneyness = np.log(S / market_data['forward_price'])
            X_pred = np.array([[moneyness, t]])
            iv, iv_std = self.gp_model.predict(X_pred, return_std=True)
            return py_lets_be_rational.implied_volatility_to_lv(iv[0], moneyness, t, iv_std[0])
        
        return local_vol

    def calibrate_yield_curve(self, market_data: Dict[str, np.ndarray]) -> Callable[[float], float]:
        spline = CubicSpline(market_data['maturities'], market_data['rates'])
        return lambda t: spline(t)

class AdvancedGreeksCalculator:
    def __init__(self, pricer: AdvancedAmericanOptionPricer):
        self.pricer = pricer

    def calculate_greeks(self, params: OptionParams) -> Dict[str, float]:
        price = self.pricer.price(params)
        
        delta = self._calculate_delta(params, price)
        gamma = self._calculate_gamma(params, price)
        vega = self._calculate_vega(params, price)
        theta = self._calculate_theta(params, price)
        rho = self._calculate_rho(params, price)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def _calculate_delta(self, params: OptionParams, price: float) -> float:
        dS = 0.01 * params.S
        params_up = OptionParams(**{**params.__dict__, 'S': params.S + dS})
        params_down = OptionParams(**{**params.__dict__, 'S': params.S - dS})
        return (self.pricer.price(params_up) - self.pricer.price(params_down)) / (2 * dS)

    def _calculate_gamma(self, params: OptionParams, price: float) -> float:
        dS = 0.01 * params.S
        params_up = OptionParams(**{**params.__dict__, 'S': params.S + dS})
        params_down = OptionParams(**{**params.__dict__, 'S': params.S - dS})
        return (self.pricer.price(params_up) - 2*price + self.pricer.price(params_down)) / (dS**2)

    def _calculate_vega(self, params: OptionParams, price: float) -> float:
        def bump_sigma(S: float, t: float) -> float:
            return params.sigma(S, t) * 1.01
        
        params_bumped = OptionParams(**{**params.__dict__, 'sigma': bump_sigma})
        return (self.pricer.price(params_bumped) - price) / (0.01 * params.sigma(params.S, 0))

    def _calculate_theta(self, params: OptionParams, price: float) -> float:
        dT = 1/365
        params_forward = OptionParams(**{**params.__dict__, 'T': params.T - dT})
        return -(self.pricer.price(params_forward) - price) / dT

    def _calculate_rho(self, params: OptionParams, price: float) -> float:
        def bump_r(t: float) -> float:
            return params.r(t) + 0.0001
        
        params_bumped = OptionParams(**{**params.__dict__, 'r': bump_r})
        return (self.pricer.price(params_bumped) - price) / 0.0001

class ModelRiskAnalyzer:
    def __init__(self, pricer: AdvancedAmericanOptionPricer):
        self.pricer = pricer

    def analyze_model_risk(self, params: OptionParams, num_simulations: int = 1000) -> Dict[str, float]:
        prices = []
        for _ in range(num_simulations):
            perturbed_params = self._perturb_params(params)
            prices.append(self.pricer.price(perturbed_params))
        
        return {
            'mean_price': np.mean(prices),
            'std_price': np.std(prices),
            'var_95': np.percentile(prices, 5),
            'var_99': np.percentile(prices, 1),
            'expected_shortfall_95': np.mean([p for p in prices if p <= np.percentile(prices, 5)])
        }

    def _perturb_params(self, params: OptionParams) -> OptionParams:
        # Implement parameter perturbation logic here
        # This could involve adding random noise to volatility, interest rates, etc.
        pass

class AdvancedAmericanOptionPricingSystem:
    def __init__(self, api_key: str):
        self.qf = QuantForgeAPI(api_key)
        self.pricer = AdvancedAmericanOptionPricer()
        self.calibrator = MarketDataCalibrator()
        self.greeks_calculator = AdvancedGreeksCalculator(self.pricer)
        self.model_risk_analyzer = ModelRiskAnalyzer(self.pricer)

    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        try:
            data = await self.qf.get_market_data(symbol)
            return data
        except Exception as e:
            logging.error(f"Error fetching market data for {symbol}: {str(e)}")
            raise

    async def price_option(self, symbol: str, strike: float, expiry: float, option_type: str) -> Dict[str, Any]:
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
            price = self.pricer.price(params)
            greeks = self.greeks_calculator.calculate_greeks(params)
            model_risk = self.model_risk_analyzer.analyze_model_risk(params)
            
            return {
                'price': price,
                'greeks': greeks,
                'model_risk': model_risk
            }
        except Exception as e:
            logging.error(f"Error pricing option: {str(e)}")
            raise
    async def run(self):
        logging.info("Starting advanced American option pricing system...")
        while True:
            try:
                # Example: Price an American call option on AAPL with strike $150 expiring in 30 days
                result = await self.price_option('AAPL', 150, 30/365, 'call')
                
                logging.info(f"AAPL $150 Call (30 days to expiry):")
                logging.info(f"Price: ${result['price']:.2f}")
                logging.info("Greeks:")
                for greek, value in result['greeks'].items():
                    logging.info(f"  {greek.capitalize()}: {value:.6f}")
                logging.info("Model Risk Analysis:")
                for metric, value in result['model_risk'].items():
                    logging.info(f"  {metric.replace('_', ' ').capitalize()}: {value:.6f}")

            except Exception as e:
                logging.error(f"Error in pricing loop: {str(e)}")

            await asyncio.sleep(5)  # Wait for 5 seconds before next pricing

# Usage
async def main():
    api_key = "your_quantforge_api_key"
    pricing_system = AdvancedAmericanOptionPricingSystem(api_key)
    await pricing_system.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


