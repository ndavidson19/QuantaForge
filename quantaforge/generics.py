from abc import ABC, abstractmethod
from typing import Union, Dict, Any, List
from dataclasses import dataclass
import polars as pl
import numpy as np

class Signal(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(self, data: pl.DataFrame) -> pl.DataFrame:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Signal':
        return cls(**config)
    
    @staticmethod
    def sell_signal(data: pl.DataFrame, name: str) -> pl.DataFrame:
        return pl.DataFrame({
            f"{name}_sell": pl.Series([1] * len(data))
        })
    
    @staticmethod
    def buy_signal(data: pl.DataFrame, name: str) -> pl.DataFrame:
        return pl.DataFrame({
            f"{name}_buy": pl.Series([1] * len(data))
        })
    
    @staticmethod
    def hold_signal(data: pl.DataFrame, name: str) -> pl.DataFrame:
        return pl.DataFrame({
            f"{name}_hold": pl.Series([1] * len(data))
        })
    
    @staticmethod
    def adjust_signal(data: pl.DataFrame, name: str) -> pl.DataFrame:
        return pl.DataFrame({
            f"{name}_adjust": pl.Series([1] * len(data))
        })
    
    

class Indicator(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(self, data: pl.DataFrame) -> pl.DataFrame:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Indicator':
        return cls(**config)

class Position:
    def __init__(self, asset: str, quantity: float, entry_price: float, entry_time: Union[str, np.datetime64]):
        self.asset = asset
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None

    def close(self, exit_price: float, exit_time: Union[str, np.datetime64]):
        self.exit_price = exit_price
        self.exit_time = exit_time

    def is_open(self) -> bool:
        return self.exit_price is None

    def calculate_pnl(self, current_price: float) -> float:
        if self.is_open():
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.exit_price - self.entry_price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset': self.asset,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': str(self.entry_time),
            'exit_price': self.exit_price,
            'exit_time': str(self.exit_time) if self.exit_time else None
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Position':
        position = cls(config['asset'], config['quantity'], config['entry_price'], config['entry_time'])
        if config['exit_price'] is not None:
            position.close(config['exit_price'], config['exit_time'])
        return position

class Portfolio:
    def __init__(self, initial_cash: float):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

    def open_position(self, asset: str, quantity: float, price: float, timestamp: Union[str, np.datetime64]):
        if asset in self.positions:
            raise ValueError(f"Position already exists for asset {asset}")
        
        cost = quantity * price
        if cost > self.cash:
            raise ValueError("Insufficient funds to open position")
        
        self.cash -= cost
        self.positions[asset] = Position(asset, quantity, price, timestamp)

    def close_position(self, asset: str, price: float, timestamp: Union[str, np.datetime64]):
        if asset not in self.positions:
            raise ValueError(f"No open position for asset {asset}")
        
        position = self.positions[asset]
        position.close(price, timestamp)
        self.cash += position.quantity * price
        
        self.closed_positions.append(position)
        del self.positions[asset]

    def calculate_total_value(self, current_prices: Dict[str, float]) -> float:
        open_positions_value = sum(
            position.calculate_pnl(current_prices[asset]) + (position.quantity * position.entry_price)
            for asset, position in self.positions.items()
        )
        return self.cash + open_positions_value

    def to_dict(self) -> Dict[str, Any]:
        return {
            'cash': self.cash,
            'positions': {asset: position.to_dict() for asset, position in self.positions.items()},
            'closed_positions': [position.to_dict() for position in self.closed_positions]
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Portfolio':
        portfolio = cls(config['cash'])
        portfolio.positions = {asset: Position.from_dict(pos_config) for asset, pos_config in config['positions'].items()}
        portfolio.closed_positions = [Position.from_dict(pos_config) for pos_config in config['closed_positions']]
        return portfolio

class RiskManagement(ABC):
    @abstractmethod
    def check(self, portfolio: Portfolio, current_prices: Dict[str, float]) -> Dict[str, Any]:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'RiskManagement':
        return cls(**config)

class StopLoss(RiskManagement):
    def __init__(self, stop_loss_percentage: float):
        self.stop_loss_percentage = stop_loss_percentage

    def check(self, portfolio: Portfolio, current_prices: Dict[str, float]) -> Dict[str, Any]:
        actions = {}
        for asset, position in portfolio.positions.items():
            current_price = current_prices[asset]
            pnl_percentage = (current_price - position.entry_price) / position.entry_price * 100
            if pnl_percentage <= -self.stop_loss_percentage:
                actions[asset] = {'action': 'close', 'reason': 'stop_loss'}
        return actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'stop_loss_percentage': self.stop_loss_percentage
        }

class TakeProfit(RiskManagement):
    def __init__(self, take_profit_percentage: float):
        self.take_profit_percentage = take_profit_percentage

    def check(self, portfolio: Portfolio, current_prices: Dict[str, float]) -> Dict[str, Any]:
        actions = {}
        for asset, position in portfolio.positions.items():
            current_price = current_prices[asset]
            pnl_percentage = (current_price - position.entry_price) / position.entry_price * 100
            if pnl_percentage >= self.take_profit_percentage:
                actions[asset] = {'action': 'close', 'reason': 'take_profit'}
        return actions

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            'take_profit_percentage': self.take_profit_percentage
        }
    
class Model:
    def __init__(self, name: str):
        self.name = name

    def fit(self, data: pl.DataFrame):
        pass

    def predict(self, data: pl.DataFrame) -> pl.DataFrame:
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'Model':
        return cls(**config)
    


@dataclass
class Condition:
    indicator: str
    operator: str
    value: Any

@dataclass
class Action:
    type: str
    asset: str
    quantity: Any

class StrategyBase:
    def __init__(self, name: str):
        self.name = name
        self.indicators: Dict[str, Indicator] = {}
        self.signals: Dict[str, Signal] = {}
        self.positions: List[Position] = []
        self.risk_management: List[RiskManagement] = []
        self.portfolio: Portfolio = None
        self.models: List[Model] = []
        self.parameters: Dict[str, Any] = {}
        self.entry_conditions: List[Condition] = []
        self.exit_conditions: List[Condition] = []
        self.actions: List[Action] = []

    def add_indicator(self, name: str, indicator: Indicator):
        self.indicators[name] = indicator

    def add_signal(self, name: str, signal: Signal):
        self.signals[name] = signal

    def add_position(self, position: Position):
        self.positions.append(position)

    def add_risk_management(self, risk_management: RiskManagement):
        self.risk_management.append(risk_management)

    def add_portfolio(self, portfolio: Portfolio):
        self.portfolio = portfolio

    def add_model(self, model: Model):
        self.models.append(model)

    def set_parameters(self, **kwargs):
        for key, value in kwargs.items():
            self.parameters[key] = value

    def add_entry_condition(self, condition: Condition):
        self.entry_conditions.append(condition)

    def add_exit_condition(self, condition: Condition):
        self.exit_conditions.append(condition)

    def add_action(self, action: Action):
        self.actions.append(action)

    def run(self, data: pl.DataFrame):
        for name, indicator in self.indicators.items():
            data = indicator.calculate(data)
        
        for name, signal in self.signals.items():
            data = signal.generate(data)
        
        for condition in self.entry_conditions:
            pass  # Implement logic to check entry conditions
        
        for condition in self.exit_conditions:
            pass  # Implement logic to check exit conditions
        
        for action in self.actions:
            pass  # Implement logic to execute actions

    def to_config(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'parameters': self.parameters,
            'indicators': [indicator.to_dict() for indicator in self.indicators.values()],
            'signals': [signal.to_dict() for signal in self.signals.values()],
            'entry_conditions': [condition.__dict__ for condition
                                    in self.entry_conditions],
            'exit_conditions': [condition.__dict__ for condition
                                    in self.exit_conditions],
            'actions': [action.__dict__ for action in self.actions],
        }
    
