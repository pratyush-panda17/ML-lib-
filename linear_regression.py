import numpy as np
import matplotlib.pyplot as plt
import pandas
import math


class LinearRegression:
    def __init__(self,iters = 1000): 
        self.slope = 0
        self.bias = 0
        self.lr = 0.6
        self.iters = iters

    def predict(self, input):
        return (self.slope*input) + self.bias
    
    def single_train(self,input , target):
        error = target - self.predict(input) 
        self.slope += error * input * self.lr
        self.bias += error * self.lr
    
    def train(self,inputs,targets):
        N = len(inputs)
        predictions = np.array([self.predict(input) for input in inputs])
        errors  = np.array(targets)-predictions
        self.slope += (np.dot(errors,np.array(inputs))/N)*self.lr
        self.bias += (np.sum(errors) * self.lr)/N 
    
    def training_block(self,inputs,targets):
        for i in range(self.iters):
            self.train(inputs,targets)
          
    def squared_error(self,input,target):
        prediction = self.predict(input)
        return (target-prediction)**2

    def mse(self,inputs,targets):
        mse = 0
        for index,input in enumerate(inputs):
            mse+=self.squared_error(input,targets[index])
        return mse/len(inputs)
    
    def r2_score(target,predictions):
        target= np.array(target)
        p = np.array(predictions)
        num = np.sum(np.square(target-p))
        mean = np.mean(target)
        den = np.sum(np.square(target - mean))
        return 1 - (num/den)
       
     






