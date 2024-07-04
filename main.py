import numpy as np
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from sklearn import datasets




class LogisticRegression():
    def __init__(self):
        self.slope = (random.random()*2) - 1
        self.bias = (random.random()*2) - 1
        self.lr = 0.1
        self.figure = plt.figure()
    
    def set_lr(self,rate):   
        self.lr = rate
    
    def get_lr(self):
        return self.lr

    def predict(self, input):
        prediction = (self.slope*input) + self.bias
        if input>=prediction:
            return 1
        else:
            return 0
    
    def train(self,input,target):
        prediction = self.predict(input)
        error  = target - prediction
        self.slope += (self.lr*error*input) 
        self.bias +=(self.lr*error)

    def accuracy(self,inputs,targets):
        count = 0
        for index,input in enumerate(inputs):   
            if self.predict(input) == targets[index]:
                count+=1
        return(count/len(inputs))*100

class MultipleLogisticRegression():
    def __init__(self,n):
        weights = []
        for i in range(n):
            weights.append((random.random()*2)-1)

        self.weights = np.array(weights)
        self.bias = (random.random()*2)-1
        self.lr = 0.1
        self.figure = plt.figure()
    
    def set_lr(self,rate):   
        self.lr = rate
    
    def get_lr(self):
        return self.lr
    
    def predict(self,inputs):
        INPUTS = np.array(inputs)
        prediction = np.dot(INPUTS,self.weights) + self.bias
        if input>=prediction:
            return 1
        else:
            return 0
    
    def train(self,inputs,target):
        output = self.predict(inputs)
        error = target - output
        INPUTS =np.array(inputs)
        self.weights +=(INPUTS * self.lr * error)
        self.bias += (self.lr * error)
    
    def accuracy(self,inputs,targets):
        count = 0
        for index,input in enumerate(inputs):   
            if self.predict(input) == targets[index]:
                count+=1
        return(count/len(inputs))*100






    
    