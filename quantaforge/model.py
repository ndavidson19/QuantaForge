import polars as pl
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        pass
    def predict(self, X):
        raise NotImplementedError("Subclasses must implement this method")
    
    def train(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")  
    
    def evaluate(self, X, y):
        raise NotImplementedError("Subclasses must implement this method")
    

class SimpleThresholdModel(Model):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def predict(self, X):
        return (X > self.threshold).astype(int)

class LinearRegressionModel(Model):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self, X, y):
        X_scaled = self.scaler.transform(X)
        return self.model.score(X_scaled, y)

class RandomForestModel(Model):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.score(X, y)
