import polars as pl
import logging

from quantaforge.model import Model

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class Strategy:
    def __init__(self):
        self.indicators = []
        self.signals = []
        self.positions = []
        self.risk_management = []
        self.portfolio = None
        self.backtest = None
        self.optimizer = None
        self.report = None

    def add_indicator(self, indicator):
        self.indicators.append(indicator)

    def add_model(self, model: Model):
        self.models.append(model)

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.parameters[key] = value
        logging.debug(f"Strategy parameters set: {self.parameters}")

    def get_parameters(self):
        return self.parameters

    def preprocess_data(self, data):
        logging.debug("Preprocessing data...")
        return data

    def validate_signal(self, signal):
        logging.debug(f"Validating signal: {signal}")
        return True

    def combine_signals(self, signals_list):
        logging.debug(f"Combining signals: {signals_list}")
        combined_signals = pl.DataFrame()
        for signals in signals_list:
            combined_signals = combined_signals.hstack(signals)
        return combined_signals

    def generate_signals(self, data):
        '''
        General method to generate signals based on the strategy
        '''
        data = pl.DataFrame(data)
        logging.debug(f"Original data: {data}")
        for indicator in self.indicators:
            data = indicator(data)
        logging.debug(f"Generated signals: {data}")
        return data

class MovingAverageStrategy(Strategy):
    def __init__(self, window):
        self.window = window

    def generate_signals(self, data):
        data = pl.DataFrame(data)
        logging.debug(f"Original data: {data}")
        result = data.with_columns([pl.col('close').rolling_mean(self.window).alias('close')])
        logging.debug(f"Generated signals (moving average): {result}")
        return result

class MomentumStrategy(Strategy):
    def generate_signals(self, data):
        data = pl.DataFrame(data)
        logging.debug(f"Original data: {data}")
        result = data.with_columns([pl.col('close').diff().alias('close')])
        logging.debug(f"Generated signals (momentum): {result}")
        return result
    