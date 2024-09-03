import logging
import os
import yfinance as yf


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

# data_loader.py

import polars as pl
import numpy as np
from datetime import datetime

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_cache = {}

    def load_historical_data(self, symbol, start_date, end_date):
        # Assuming historical_data is a DataFrame in Polars
        historical_data = self._fetch_data_for_symbol(symbol)

        # Parsing the start_date and end_date strings to Date types
        start_date_parsed = pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
        end_date_parsed = pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")

        # Filter the historical data by date range
        filtered_data = historical_data.filter(
            (pl.col('date') >= start_date_parsed) &
            (pl.col('date') <= end_date_parsed)
        )

        return filtered_data
    
    def _fetch_data_for_symbol(self, symbol):
        if symbol in self.data_cache:
            return self.data_cache[symbol]
        
        file_path = f"{self.data_dir}/{symbol}.parquet"
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Fetching data for {symbol} from Yahoo Finance.")
            data = self._fetch_from_yfinance(symbol)
            # Save the fetched data as a Parquet file for future use
            data.write_parquet(file_path)
        else:
            data = pl.read_parquet(file_path)
        
        self.data_cache[symbol] = data
        
        return data
    

    def _fetch_from_yfinance(self, symbol):
        # Fetch historical data using yfinance
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="max")
        # Convert to a Polars DataFrame
        df = pl.DataFrame({
            'date': pl.Series(hist.index, dtype=pl.Date),
            'open': hist['Open'],
            'high': hist['High'],
            'low': hist['Low'],
            'close': hist['Close'],
            'volume': hist['Volume'],
        })

        return df
    
    def _generate_sample_data(self, symbol: str, start_date: str, end_date: str):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        date_range = pl.date_range(start, end, "1d", eager=True)
        dates = date_range.to_list()
        n = len(dates)
        
        initial_price = 100
        returns = np.random.normal(0, 0.02, n)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        df = pl.DataFrame({
            'date': dates,
            'open': prices * np.random.uniform(0.99, 1.01, n),
            'high': prices * np.random.uniform(1.01, 1.03, n),
            'low': prices * np.random.uniform(0.97, 0.99, n),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, n)
        })
        
        file_path = f"{self.data_dir}/{symbol}.parquet"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.write_parquet(file_path)
        data_logger.info(f"Sample data generated and saved to {file_path}")