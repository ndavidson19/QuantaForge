import logging
import json
import yaml
from typing import Dict, List
from quantaforge.strategy import Strategy

class StrategyManager:
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}

    def add_strategy(self, strategy: Strategy):
        self.strategies[strategy.name] = strategy
        logging.info(f"Added strategy: {strategy.name}")

    def remove_strategy(self, strategy_name: str):
        if strategy_name in self.strategies:
            del self.strategies[strategy_name]
            logging.info(f"Removed strategy: {strategy_name}")
        else:
            logging.warning(f"Strategy not found: {strategy_name}")

    def get_strategy(self, strategy_name: str) -> Strategy:
        return self.strategies.get(strategy_name)

    def list_strategies(self) -> List[str]:
        return list(self.strategies.keys())

    def run_all_strategies(self):
        for strategy in self.strategies.values():
            strategy.run()

    def run_strategy(self, strategy_name: str):
        strategy = self.get_strategy(strategy_name)
        if strategy:
            strategy.run()
        else:
            logging.error(f"Strategy not found: {strategy_name}")

    def load_strategies_from_config(self, config_path: str):
        file_extension = config_path.split('.')[-1].lower()
        
        try:
            with open(config_path, 'r') as file:
                if file_extension == 'json':
                    config = json.load(file)
                elif file_extension in ['yml', 'yaml']:
                    config = yaml.safe_load(file)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            for strategy_config in config['strategies']:
                strategy = Strategy.from_config(strategy_config)
                self.add_strategy(strategy)
            
            logging.info(f"Loaded {len(config['strategies'])} strategies from {config_path}")
        except Exception as e:
            logging.error(f"Error loading strategies from {config_path}: {str(e)}")

    def save_strategies_to_config(self, config_path: str):
        file_extension = config_path.split('.')[-1].lower()
        
        config = {
            'strategies': [strategy.to_config() for strategy in self.strategies.values()]
        }
        
        try:
            with open(config_path, 'w') as file:
                if file_extension == 'json':
                    json.dump(config, file, indent=2)
                elif file_extension in ['yml', 'yaml']:
                    yaml.dump(config, file)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            logging.info(f"Saved {len(self.strategies)} strategies to {config_path}")
        except Exception as e:
            logging.error(f"Error saving strategies to {config_path}: {str(e)}")