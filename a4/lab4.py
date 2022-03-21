#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 03:13:06 2022

@author: nickfang
"""

import numpy as np
import pandas as pd
import os
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


boston = load_boston()

for i in np.arange(len(boston)):
    print(list(boston.keys())[i])

data = np.c_[boston['data'], boston['target']]
columns = np.append(boston['feature_names'], 'MEDV')

df= pd.DataFrame(data, columns= columns)
df.head()

high = []
for row in df['MEDV']:
    if row > 40:
        high.append(1)
    else:
        high.append(0)


#6
df['high-priced'] = high
print(df['high-priced'])
print(df['high-priced'].sum())
plt.scatter(df['MEDV'], df['RM'], c = df['high-priced'])
#plt.scatter(df['MEDV'], df['high-priced'])
#plt.scatter(df["RM"], df['high-priced'])

#Higher priced homes are more likely to have a greater number of RM(rooms per dwelling)

#Q7
x1 = df[['RM', 'MEDV']]
y1 = df[['high-priced']]

print(x1.shape, y1.shape)

#Q8
x1.insert(0, 'bias', 1)
x1.head(5)

x1 = x1.to_numpy()
y1 = y1.to_numpy()
print(type(x1))
print(type(y1))
m, n = x1.shape

import random
w_ = [random.uniform(0, 1.0) for _ in range(1+x1.shape[1])]
w_[1:]
w_[0]

#9
w = np.linspace(0,1,5,endpoint=True)[1:-1]

#def perceptron(x, y, w, step):
 #step = learning rate   
 #   m, n = x1.shape
  #  theta = np.zeros((n+1,1))
    
   # incorrect = []
    
#class myPerceptron():
 #   def __init__(self, eta=0.1, n_iter = 1000):
  #      self.eta = eta
   #     self.n_iter = n_iter
        
    #def fit(self, x, y):
     #   self.w_ = [random.uniform(0.0, 1.0) for _ in range(1+x.shape[1])]
      #  self.errors_ = []
        
       # for _ in range(self.n_iter):
        #    errors = 0
         #   for xi, label in zip(x, y):
          #      update = self.eta * (self.predict(xi))
           #     self.w_[1:] += update * xi
            #    self.w_[0] += update
             #   errors += int(update != 0.0)
           # self.errors_.append(errors)
       # return self
    
   # def net_input(self, x):
    #    return np.dot(x, self.w_[1:]) + self.w_[0]

   # def predict(self, x):
    #    return np.where(self.net_input(x) >= 0.0, 1, -1)

#ppn = myPerceptron(n_iter=2000)
#ppn.fit(x1, y1) #training


#plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)

#Q10-11

#https://m-abdin.medium.com/an-intuitive-overview-of-a-perceptron-with-python-implementation-part-1-fundamentals-519a2af2dd81
class myPerceptron(object):
    
    def __init__(self, eta=0.01, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        #Set random seed
        rgen = np.random.RandomState(self.random_state)
        
        #Generate a random number of normal distribution, weight W
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            #  Iterate all samples and update weights according to the perceptual rules
            errors = 0
            for xi, target in zip(X, y):
                # print(xi,target)
                update = self.eta * (target - self.predict(xi))
                self.w_[0] += update
                self.w_[1:] += update * xi
                #Prediction error: Update if not 0, then the judgment error
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        z = self.w_[0] * 1 + np.dot(X, self.w_[1:])
        return z
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

ppn = myPerceptron(eta=0.1, n_iter=100)
ppn.fit(x1, y1)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')


#Q12
    
#PART3    
#Q14
df= pd.DataFrame(data, columns= columns)
df.head()

target = "MEDV"
features = ["CRIM", 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
            'TAX', 'PTRATIO', 'B', 'LSTAT']


X = df[features]
Y = df[target]

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


sscaler = preprocessing.StandardScaler()
sscaler.fit(X)
X_std = sscaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_std, Y, test_size=0.2, random_state=99)

#training test split 4:1, test 102 rows, train 404 rows

#Q15
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(hidden_layer_sizes=50, max_iter=1000)

mlp.fit(X,Y)
mlp.score(X,Y)

#output
y_predict = mlp.predict(X)

plt.plot(X, y_predict)
plt.legend(['original', 'predicted'])

#size = 2, max iter reached and optimization has not coverged yet, 38.45%
#size = 5, max iter reached and optimization has not coverged yet, 54.41%
#size = 10, max iter reached and optimization has not coverged yet, 67.32%
#size = 20, max iter reached and optimization has not coverged yet, 69.69%
#size = 50, max iter reached and optimization has not coverged yet, 80.32%

#results got a little better with more layers

#Q16
mlp = MLPRegressor(hidden_layer_sizes=(50,50), max_iter=1000)

mlp.fit(X,Y)
mlp.score(X,Y)

#size = (2,2) 68.05%
#size = (5,5) 65.20%
#size = (10,10) 72.66%
#size = (20,20) 72.31%
#size = (50,50) 70.47%

#results marginally improved with more layers

#Q17
mlp = MLPRegressor(hidden_layer_sizes=(100,100), max_iter=1000)
mlp.fit(X,Y)
mlp.score(X,Y)

#Hidden layer of (100, 100) improved to 82%
#Running the MLP to (1000, 1000) actually produced a lower score of 70.93%,
#designing the next part of the model would be to change variables such as 
#learning rate, activation









