import numpy as np


class MultipleLinearRegression:  

    def __init__(self,n,iters= 1000):
  
        self.weights = np.zeros(n)
        self.bias = 0
        self.lr = 0.1
        self.iters = iters
    
    def predict(self,inputs):
        INPUTS = np.array(inputs)
        return np.dot(INPUTS,self.weights) + self.bias

    def train(self,inputs,targets):
        N = len(inputs)
        INPUTS = np.array(inputs)
        TARGETS = np.array(targets)
        predictions = np.dot(INPUTS,self.weights) + self.bias
        ERRORS = TARGETS - predictions
        self.weights += (1/N * self.lr * np.dot(ERRORS,INPUTS))
        self.bias +=(1/N * self.lr * np.sum(ERRORS))

    
    def square_error(self,input,target):
        return (target - self.predict(input))**2
    
    def mse(self,inputs,targets):
        mse = 0
        for index,input in enumerate(inputs):
            mse+=self.square_error(input,targets[index])
        return mse/len(inputs)
