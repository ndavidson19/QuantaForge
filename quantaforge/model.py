import polars as pl
import logging

class Model:
    def __init__(self):
        pass
    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")  
    
    def evaluate(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")
    
    