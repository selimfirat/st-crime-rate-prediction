import numpy as np


class RandomPredictor:
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        return np.random.randint(0, 2, (len(X), 33))
    
class OnesPredictor:
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        return np.ones((len(X), 33))
    
class ZerosPredictor:
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        return np.zeros((len(X), 33))
    
class AveragePredictor:
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        return np.round(X.mean() * np.ones((len(X), 33)))