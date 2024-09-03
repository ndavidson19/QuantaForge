# File: src/utils/logger.py

import logging
import os

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