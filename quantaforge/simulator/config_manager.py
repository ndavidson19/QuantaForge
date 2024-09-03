# File: src/config_manager.py

import yaml

class ConfigManager:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_simulation_params(self, simulation_type):
        return self.config['simulations'][simulation_type]

    def get_kafka_config(self):
        return self.config['kafka']

    def get_data_loader_config(self):
        return self.config['data_loader']
