from dataclasses import dataclass
from typing import Union, List, Callable
import operator

@dataclass
class Condition:
    indicator: str
    operator: str
    value: Union[float, str]
    def evaluate(self, data: dict) -> bool:
        op_map = {
            '>': operator.gt,
            '<': operator.lt,
            '==': operator.eq,
            '!=': operator.ne,
            '>=': operator.ge,
            '<=': operator.le
        }
        if self.indicator not in data:
            return False
        actual_value = data[self.indicator]
        return op_map[self.operator](actual_value, self.value)
    
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
