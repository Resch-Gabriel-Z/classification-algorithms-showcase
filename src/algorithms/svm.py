import numpy as np


class SVM:
    def __init__(self, C=1.0, learning_rate=0.01, max_iters=1000, tol=1e-4):
        self.C = C
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        
        self.weights = None
        self.bias = None
        
    def fit(self,X,y):
        
        n_samples, n_features = X.shape
        
        # Data preprocessing
        y_ = np.where(y <=0,-1,1)
        
        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0
        pass
    
    def predict(self,X):
        pass

