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
        self.models = []
        self.parameters = {}

    def add_indicator(self, indicator):
        self.indicators.append(indicator)
    
    def add_signal(self, signal):
        self.signals.append(signal)

    def add_report(self, report):
        self.report = report

    def add_position(self, position):
        self.positions.append(position)
    
    def add_risk_management(self, risk_management):
        self.risk_management.append(risk_management)
    
    def add_portfolio(self, portfolio):
        self.portfolio = portfolio
    
    def add_backtest(self, backtest):
        self.backtest = backtest

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    def add_report(self, report):
        self.report = report
    
    def run(self):
        logging.debug("Running strategy...")
        if self.backtest:
            return self.backtest.run()
        return None
        
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

    def combine_signals(self, signals_list):
        logging.debug(f"Combining signals: {signals_list}")
        combined_signals = pl.DataFrame()
        for signals in signals_list:
            combined_signals = combined_signals.hstack(signals)
        return combined_signals

    def generate_signals(self, data):
        data = pl.DataFrame(data)
        logging.debug(f"Original data: {data}")
        all_signals = []
        for indicator in self.indicators:
            data = indicator.calculate(data)
        for signal in self.signals:
            indicator_data = data.select(pl.col(signal.get_indicator_column_name()))
            print(indicator_data)
            signal_data = signal.generate(data, indicator_data)
            all_signals.append(signal_data)
        combined_signals = self.combine_signals(all_signals)
        logging.debug(f"Generated signals: {combined_signals}")
        return combined_signals
    
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
    