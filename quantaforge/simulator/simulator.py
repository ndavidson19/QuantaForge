import polars as pl
import numpy as np
from arch import arch_model
from scipy.stats import norm
import concurrent.futures
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from data_loader import DataLoader
from data_streamer import DataStreamer, StdoutStreamer
import traceback

import logging
import os
import yaml

class ConfigManager:
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_data_loader_config(self) -> Dict[str, Any]:
        return self.config.get('data_loader', {})

    def get_kafka_config(self) -> Dict[str, Any]:
        return self.config.get('kafka', {})

    def get_simulation_config(self) -> Dict[str, Any]:
        general_config = self.config.get('simulations', {}).get('general', {})
        return {
            'start_date': general_config.get('start_date'),
            'end_date': general_config.get('end_date'),
            'symbols': general_config.get('symbols', [])
        }

    def get_simulation_params(self, simulation_type: str) -> Dict[str, Any]:
        if simulation_type == 'deterministic':
            return self.config.get('simulations', {}).get('deterministic', {})
        elif simulation_type in ['garch', 'mean_reversion', 'jump_diffusion']:
            return self.config.get('simulations', {}).get('statistical', {}).get(simulation_type, {})
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")

    def get_all_simulation_params(self) -> Dict[str, Any]:
        return self.config.get('simulations', {})
    
    
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Create loggers
simulation_logger = setup_logger('simulation_logger', 'logs/simulation.log')
data_logger = setup_logger('data_logger', 'logs/data.log')
streaming_logger = setup_logger('streaming_logger', 'logs/streaming.log')

class SimulatorBase(ABC):
    def __init__(self, data_loader, data_streamer):
        self.data_loader = data_loader
        self.data_streamer = data_streamer

    @abstractmethod
    def _run_simulation_impl(self, symbol: str, start_date: str, end_date: str, params: Dict, mode: str, speed: float) -> pl.DataFrame:
        pass

    def _stream_data(self, topic: str, data: pl.DataFrame, mode: str, speed: float):
        if mode == 'batch':
            self.data_streamer.stream_data_fast(topic, data)
        elif mode == 'realtime':
            self.data_streamer.stream_data_realtime(topic, data, speed, 'daily')
        elif mode == 'fast-forward':
            self.data_streamer.stream_data_realtime(topic, data, speed, 'daily')
        else:
            raise ValueError(f"Invalid mode: {mode}")

class DeterministicSimulator(SimulatorBase):
    def _run_simulation_impl(self, symbol: str, start_date: str, end_date: str, params: Dict, mode: str, speed: float) -> pl.DataFrame:
        data = self.data_loader.load_historical_data(symbol, start_date, end_date)
        # Apply deterministic modifications based on params if needed
        adjusted_close = data['close'] * params.get('price_adjustment', 1.0)
        simulated_data = data.with_columns([pl.Series("simulated_close", adjusted_close)])
        return simulated_data

class StatisticalSimulator(SimulatorBase):
    def _run_simulation_impl(self, symbol: str, start_date: str, end_date: str, params: Dict, mode: str, speed: float) -> pl.DataFrame:
        historical_data = self.data_loader.load_historical_data(symbol, start_date, end_date)
        
        simulation_method = params.get('method', 'garch')
        if simulation_method == 'garch':
            simulated_data = self._garch_simulation(historical_data, params)
        elif simulation_method == 'mean_reversion':
            simulated_data = self._mean_reversion_simulation(historical_data, params)
        elif simulation_method == 'jump_diffusion':
            simulated_data = self._jump_diffusion_simulation(historical_data, params)
        else:
            raise ValueError(f"Unknown simulation method: {simulation_method}")
        simulated_data = simulated_data.with_columns(pl.lit(simulation_method).alias('method'))

        return simulated_data

    def _garch_simulation(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        returns = data['close'].pct_change().drop_nulls()
        
        garch_model = arch_model(returns, vol='Garch', p=params.get('p', 1), q=params.get('q', 1))
        garch_result = garch_model.fit(disp='off')
        
        n = len(data)
        new_returns = np.zeros(n)
        new_volatility = np.zeros(n)
        
        omega = garch_result.params['omega']
        alpha = garch_result.params['alpha[1]']
        beta = garch_result.params['beta[1]']
        
        for t in range(1, n):
            new_volatility[t] = np.sqrt(omega + alpha * new_returns[t-1]**2 + beta * new_volatility[t-1]**2)
            new_returns[t] = new_volatility[t] * norm.rvs()
        
        new_prices = data['close'].to_numpy() * np.exp(np.cumsum(new_returns))
        
        return data.with_columns([
            pl.Series("simulated_close", new_prices),
            pl.Series("simulated_returns", new_returns)
        ])

    def _mean_reversion_simulation(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        returns = data['close'].pct_change().drop_nulls()
        
        mean = params.get('long_term_mean', returns.mean())
        speed = params.get('reversion_speed', 0.1)
        volatility = params.get('volatility', returns.std())
        
        n = len(data)
        new_returns = np.zeros(n)
        
        for t in range(1, n):
            drift = speed * (mean - new_returns[t-1])
            diffusion = volatility * norm.rvs()
            new_returns[t] = new_returns[t-1] + drift + diffusion
        
        new_prices = data['close'].to_numpy() * np.exp(np.cumsum(new_returns))
        
        return data.with_columns([
            pl.Series("simulated_close", new_prices),
            pl.Series("simulated_returns", new_returns)
        ])

    def _jump_diffusion_simulation(self, data: pl.DataFrame, params: dict) -> pl.DataFrame:
        returns = data['close'].pct_change().drop_nulls()
        
        mu = params.get('drift', returns.mean())
        sigma = params.get('volatility', returns.std())
        lambda_jump = params.get('jump_intensity', 0.1)
        jump_mean = params.get('jump_mean', 0)
        jump_std = params.get('jump_std', sigma * 2)
        
        n = len(data)
        dt = 1/252  # Assuming daily data
        
        poisson_rv = np.random.poisson(lambda_jump * dt, n)
        normal_rv = norm.rvs(size=n)
        jump_rv = norm.rvs(loc=jump_mean, scale=jump_std, size=n)
        
        new_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * normal_rv + poisson_rv * jump_rv
        new_prices = data['close'].to_numpy() * np.exp(np.cumsum(new_returns))
        
        return data.with_columns([
            pl.Series("simulated_close", new_prices),
            pl.Series("simulated_returns", new_returns)
        ])
    
class Simulator:
    def __init__(self, config_file: str, use_kafka: bool = False):
        self.config_manager = ConfigManager(config_file)
        data_loader_config = self.config_manager.get_data_loader_config()
        
        self.data_loader = DataLoader(data_loader_config.get('data_dir', 'data/historical'))
        
        if use_kafka:
            kafka_config = self.config_manager.get_kafka_config()
            self.data_streamer = DataStreamer(kafka_config.get('bootstrap_servers', ['localhost:9092']))
        else:
            self.data_streamer = StdoutStreamer()
        
        self.deterministic_simulator = DeterministicSimulator(self.data_loader, self.data_streamer)
        self.statistical_simulator = StatisticalSimulator(self.data_loader, self.data_streamer)

    def run(self, symbols: List[str] = None, **kwargs):
        sim_config = self.config_manager.get_simulation_config()
        sim_config.update(kwargs)
        
        symbols = symbols or sim_config.get('symbols', [])
        if not symbols:
            raise ValueError("At least one symbol must be provided")

        method = sim_config.get('method', 'garch')
        start_date = sim_config.get('start_date')
        end_date = sim_config.get('end_date')
        mode = sim_config.get('mode', 'batch')
        speed = sim_config.get('speed', 1.0)
        
        simulation_params = self.config_manager.get_simulation_params(method)
        sim_config.update(simulation_params)
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._run_single_simulation, symbol, method, start_date, end_date, sim_config, mode, speed)
                       for symbol in symbols]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return results

    def _run_single_simulation(self, symbol: str, method: str, start_date: str, end_date: str, sim_config: Dict, mode: str, speed: float):
        try:
            simulation_logger.info(f"Starting simulation for {symbol} from {start_date} to {end_date}")
            
            if method == 'deterministic':
                simulated_data = self.deterministic_simulator._run_simulation_impl(symbol, start_date, end_date, sim_config, mode, speed)
            elif method in ['garch', 'mean_reversion', 'jump_diffusion']:
                simulated_data = self.statistical_simulator._run_simulation_impl(symbol, start_date, end_date, sim_config, mode, speed)
            else:
                raise ValueError(f"Unknown simulation method: {method}")

            simulation_logger.info(f"Simulation completed for {symbol}")
            return (symbol, simulated_data)
        except Exception as e:
            simulation_logger.error(f"Error in simulation for {symbol}: {str(e)}")
            simulation_logger.error(traceback.format_exc())
            raise

    def _stream_data(self, topic: str, data: pl.DataFrame, mode: str, speed: float):
        if mode == 'batch':
            self.data_streamer.stream_data_fast(topic, data)
        else:
            self.data_streamer.stream_data_realtime(topic, data, speed, 'daily')

    async def stream_results(self, results, mode='fast', speed=1.0, time_dilation='daily'):
        for symbol, data in results:
            # Ensure 'method' column exists, if not, use the method from sim_config
            if 'method' not in data.columns:
                method = self.config_manager.get_simulation_config().get('method', 'unknown')
                data = data.with_columns(pl.lit(method).alias('method'))
            
            topic = f"{symbol}_{data['method'][0]}"
            if mode == 'fast':
                await self.data_streamer.stream_data_fast(topic, data)
            else:
                await self.data_streamer.stream_data_realtime(topic, data, speed, time_dilation)