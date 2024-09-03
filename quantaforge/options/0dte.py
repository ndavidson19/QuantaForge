import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class AdvancedEdgeworthOptionPricer:
    def __init__(self):
        self.S = None  # Current stock price
        self.r = None  # Risk-free rate
        self.params = None  # Model parameters

    def set_market_data(self, S, r):
        self.S = S
        self.r = r

    def set_parameters(self, sigma, beta_tilde, rho, alpha_Q, delta_tilde, eta, lambda_t, mu_j, sigma_j):
        self.params = {
            'sigma': sigma,
            'beta_tilde': beta_tilde,
            'rho': rho,
            'alpha_Q': alpha_Q,
            'delta_tilde': delta_tilde,
            'eta': eta,
            'lambda_t': lambda_t,
            'mu_j': mu_j,
            'sigma_j': sigma_j
        }

    def log_char_func_continuous(self, u, tau):
        sigma = self.params['sigma']
        beta_tilde = self.params['beta_tilde']
        rho = self.params['rho']
        alpha_Q = self.params['alpha_Q']
        delta_tilde = self.params['delta_tilde']
        eta = self.params['eta']

        mu_tilde = self.r - 0.5 * sigma**2

        term1 = 1j * u * mu_tilde * tau / (sigma * np.sqrt(tau)) - 0.5 * u**2
        term2 = -1j * u**3 * beta_tilde * rho / (2 * sigma * np.sqrt(tau))
        term3 = -u**2 * ((alpha_Q + delta_tilde) / (2 * sigma) + beta_tilde**2 / (4 * sigma**2)) * tau
        term4 = 1/24 * beta_tilde**2 / sigma**2 * u**2 * (4 * u**2 - rho**2 * u**2 * (3 * u**2 - 8)) * tau
        term5 = eta / (6 * sigma) * u**4 * tau

        return term1 + term2 + term3 + term4 + term5

    def log_char_func_jumps(self, u, tau):
        sigma = self.params['sigma']
        lambda_t = self.params['lambda_t']
        mu_j = self.params['mu_j']
        sigma_j = self.params['sigma_j']

        mu_j_bar = np.exp(mu_j / (sigma * np.sqrt(tau)) + 0.5 * sigma_j**2 / (sigma**2 * tau)) - 1
        
        return tau * lambda_t * (np.exp(1j * u * mu_j / (sigma * np.sqrt(tau)) - 
                                        0.5 * u**2 * sigma_j**2 / (sigma**2 * tau)) - 
                                 1 - 1j * u * mu_j_bar)

    def char_func(self, u, tau):
        return np.exp(self.log_char_func_continuous(u, tau) + self.log_char_func_jumps(u, tau))

    def integrand(self, u, d2, flag, tau):
        if flag:
            char_func = self.char_func(u - 1j, tau)
        else:
            char_func = self.char_func(u, tau)
        return np.real(np.exp(-1j * u * d2) * char_func / (1j * u))

    def price(self, K, T):
        if isinstance(K, pd.Series):
            K = K.iloc[0]
        if isinstance(T, pd.Series):
            T = T.iloc[0]

        sigma = self.params['sigma']
        d1 = (np.log(self.S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Use a finite integration interval
        u_max = 100 / (sigma * np.sqrt(T))
        
        try:
            integral1, _ = quad(self.integrand, 0, u_max, args=(d2, True, T), limit=1000, epsabs=1e-8, epsrel=1e-8)
            integral2, _ = quad(self.integrand, 0, u_max, args=(d2, False, T), limit=1000, epsabs=1e-8, epsrel=1e-8)
        except Exception as e:
            print(f"Integration error: {e}")
            return np.nan

        term1 = self.S * (0.5 + 1/np.pi * integral1)
        term2 = K * np.exp(-self.r * T) * (0.5 + 1/np.pi * integral2)

        return term1 - term2

    def implied_volatility(self, price, K, T):
        if isinstance(K, pd.Series):
            K = K.iloc[0]
        if isinstance(T, pd.Series):
            T = T.iloc[0]
        if isinstance(price, pd.Series):
            price = price.iloc[0]

        def objective(vol):
            d1 = (np.log(self.S / K) + (self.r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
            d2 = d1 - vol * np.sqrt(T)
            model_price = self.S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
            return (model_price - price)**2

        result = minimize(objective, x0=0.2, method='BFGS')
        return result.x[0]

    def calculate_greeks(self, K, T):
        epsilon = 1e-5
        price = self.price(K, T)
        delta = (self.price(K, T, S=self.S + epsilon) - price) / epsilon
        gamma = (self.price(K, T, S=self.S + epsilon) + self.price(K, T, S=self.S - epsilon) - 2 * price) / epsilon**2
        theta = (self.price(K, T + epsilon) - price) / epsilon
        vega = (self.price(K, T, sigma=self.params['sigma'] + epsilon) - price) / epsilon
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega
        }

class ParameterEstimator:
    def __init__(self, pricer):
        self.pricer = pricer

    def estimate_parameters(self, market_data):
        def objective(params):
            self.pricer.set_parameters(*params)
            total_error = 0
            for _, row in market_data.iterrows():
                model_price = self.pricer.price(row.Strike, row.T)
                if np.isnan(model_price):
                    return np.inf
                total_error += (model_price - row.Price)**2
            return total_error

        initial_guess = [0.2, 0.3, -0.5, 0.1, 0.1, 0.1, 1.0, -0.01, 0.05]
        bounds = [(0.01, 1), (0.01, 1), (-1, 1), (-1, 1), (-1, 1), (0, 1), (0, 10), (-0.1, 0.1), (0.01, 0.5)]
        
        result = minimize(objective, x0=initial_guess, method='L-BFGS-B', bounds=bounds)
        return result.x

class TradingStrategy:
    def __init__(self, pricer):
        self.pricer = pricer

    def delta_hedge(self, K, T, position_size):
        greeks = self.pricer.calculate_greeks(K, T)
        hedge_ratio = -greeks['delta'] * position_size
        return hedge_ratio

    def evaluate_strategy(self, initial_stock_price, final_stock_price, K, T, position_size):
        initial_option_price = self.pricer.price(K, T)
        self.pricer.set_market_data(final_stock_price, self.pricer.r)
        final_option_price = self.pricer.price(K, T)
        
        option_pnl = (final_option_price - initial_option_price) * position_size
        
        hedge_ratio = self.delta_hedge(K, T, position_size)
        stock_pnl = (final_stock_price - initial_stock_price) * hedge_ratio
        
        total_pnl = option_pnl + stock_pnl
        return total_pnl

# Example usage
if __name__ == "__main__":
    # Initialize the pricer
    pricer = AdvancedEdgeworthOptionPricer()
    pricer.set_market_data(S=100, r=0.05)

    # Generate sample market data
    market_data = pd.DataFrame({
        'Strike': [95, 100, 105],
        'T': [1/365, 1/365, 1/365],
        'Price': [5.5, 2.5, 0.8]
    })

    # Estimate parameters
    estimator = ParameterEstimator(pricer)
    estimated_params = estimator.estimate_parameters(market_data)
    pricer.set_parameters(*estimated_params)

    # Price options and calculate implied volatilities
    for _, row in market_data.iterrows():
        price = pricer.price(row.Strike, row.T)
        iv = pricer.implied_volatility(price, row.Strike, row.T)
        print(f"Strike: {row.Strike}, Price: {price:.4f}, IV: {iv:.4f}")

    # Implement a simple trading strategy
    strategy = TradingStrategy(pricer)
    initial_stock_price = 100
    final_stock_price = 101
    K = 100
    T = 1/365
    position_size = 100  # Number of contracts

    pnl = strategy.evaluate_strategy(initial_stock_price, final_stock_price, K, T, position_size)
    print(f"Strategy P&L: ${pnl:.2f}")

    # Plot implied volatility smile
    strikes = np.linspace(90, 110, 21)
    ivs = [pricer.implied_volatility(pricer.price(K, 1/365), K, 1/365) for K in strikes]
    
    plt.figure(figsize=(10, 6))
    plt.plot(strikes, ivs)
    plt.title("0DTE Implied Volatility Smile")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.show()