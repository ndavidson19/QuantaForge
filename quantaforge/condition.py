from dataclasses import dataclass
from typing import Union, List, Callable
import operator

@dataclass
class Condition:
    indicator: str
    operator: str
    value: Union[float, str]

class ConditionBuilder:
    def __init__(self, indicator: str):
        self.indicator = indicator

    def above(self, value: float) -> Condition:
        return Condition(self.indicator, '>', value)

    def below(self, value: float) -> Condition:
        return Condition(self.indicator, '<', value)

    def equal_to(self, value: float) -> Condition:
        return Condition(self.indicator, '==', value)

    def not_equal_to(self, value: float) -> Condition:
        return Condition(self.indicator, '!=', value)

class ConditionEvaluator:
    @staticmethod
    def evaluate(condition: Condition, data: dict) -> bool:
        op_map = {
            '>': operator.gt,
            '<': operator.lt,
            '==': operator.eq,
            '!=': operator.ne,
            '>=': operator.ge,
            '<=': operator.le
        }
        return op_map[condition.operator](data[condition.indicator], condition.value)

class Strategy:
    # ... (other methods)

    def add_entry_condition(self, condition: Condition):
        self.entry_conditions.append(condition)

    def add_exit_condition(self, condition: Condition):
        self.exit_conditions.append(condition)

    def entry_rule(self, func: Callable[['Strategy'], List[Condition]]):
        self.entry_conditions = func(self)

    def exit_rule(self, func: Callable[['Strategy'], List[Condition]]):
        self.exit_conditions = func(self)

# Usage example:
strategy = Strategy("MyStrategy")
sma_fast = ConditionBuilder("SMA_10")
sma_slow = ConditionBuilder("SMA_30")

@strategy.entry_rule
def entry(s):
    return [
        sma_fast.above(sma_slow),
        ConditionBuilder("RSI").below(30)
    ]

@strategy.exit_rule
def exit(s):
    return [
        sma_fast.below(sma_slow),
        ConditionBuilder("RSI").above(70)
    ]