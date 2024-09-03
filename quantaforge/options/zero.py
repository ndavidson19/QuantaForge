import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

class EdgeworthOptionPricer:
    def __init__(self, S, K, T, r, sigma, beta_tilde, rho, lambda_t, mu_j, sigma_j):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Spot volatility
        self.beta_tilde = beta_tilde  # Volatility of volatility
        self.rho = rho  # Leverage (correlation between price and volatility changes)
        self.lambda_t = lambda_t  # Jump intensity
        self.mu_j = mu_j  # Jump mean
        self.sigma_j = sigma_j  # Jump standard deviation

    def char_func_continuous(self, u):
        tau = self.T
        sigma = self.sigma
        beta_tilde = self.beta_tilde
        rho = self.rho
        r = self.r

        mu_tilde = r - 0.5 * sigma**2

        term1 = np.exp(1j * u * mu_tilde * tau / (sigma * np.sqrt(tau)) - 0.5 * u**2)
        term2 = np.exp(-1j * u**3 * beta_tilde * rho / (2 * sigma * np.sqrt(tau)))
        term3 = np.exp(-0.25 * u**2 * beta_tilde**2 * tau / sigma**2)
        
        return term1 * term2 * term3

    def char_func_jumps(self, u):
        tau = self.T
        sigma = self.sigma
        lambda_t = self.lambda_t
        mu_j = self.mu_j
        sigma_j = self.sigma_j

        mu_j_bar = np.exp(mu_j / (sigma * np.sqrt(tau)) + 0.5 * sigma_j**2 / (sigma**2 * tau)) - 1
        
        return np.exp(tau * lambda_t * (np.exp(1j * u * mu_j / (sigma * np.sqrt(tau)) - 
                                               0.5 * u**2 * sigma_j**2 / (sigma**2 * tau)) - 
                                        1 - 1j * u * mu_j_bar))

    def char_func(self, u):
        return self.char_func_continuous(u) * self.char_func_jumps(u)

    def integrand(self, u, d2, flag):
        if flag:
            char_func = self.char_func(u - 1j)
        else:
            char_func = self.char_func(u)
        return np.real(np.exp(-1j * u * d2) * char_func / (1j * u))

    def price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        integral1, _ = quad(self.integrand, 0, np.inf, args=(d2, True))
        integral2, _ = quad(self.integrand, 0, np.inf, args=(d2, False))

        term1 = self.S * (0.5 + 1/np.pi * integral1)
        term2 = self.K * np.exp(-self.r * self.T) * (0.5 + 1/np.pi * integral2)

        return term1 - term2

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1/365  # Time to maturity (1 day)
r = 0.05  # Risk-free rate
sigma = 0.2  # Spot volatility
beta_tilde = 0.3  # Volatility of volatility
rho = -0.5  # Leverage
lambda_t = 1  # Jump intensity
mu_j = -0.01  # Jump mean
sigma_j = 0.05  # Jump standard deviation

pricer = EdgeworthOptionPricer(S, K, T, r, sigma, beta_tilde, rho, lambda_t, mu_j, sigma_j)
option_price = pricer.price()
print(f"0DTE option price: {option_price:.4f}")