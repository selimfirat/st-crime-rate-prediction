from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class SVRPredictor:
    
    def __init__(self):
        self.clf = MultiOutputRegressor(LinearSVR())
        self.oh = OneHotEncoder()
    
    def train(self, X, y):
        X = X.reshape((len(X), 1))
        X = self.oh.fit_transform(X)
        print(X.shape, y.shape)
        self.clf.fit(X, y)
        
    def predict(self, X):
        X = X.reshape((len(X), 1))
        X = self.oh.transform(X)
        return self.clf.predict(X)