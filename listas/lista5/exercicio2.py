'''aluno: Rennan de Lucena Gaio'''

import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd



def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(self, theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
    return total_cost

def gradient(self, theta, x, y):
    # Computes the gradient of the cost function at the point theta
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

def fit(self, x, y, theta):
    opt_weights = fmin_tnc(func=cost_function, x0=theta, fprime=gradient,args=(x, y.flatten()))
    return opt_weights[0]



T=70000
data = pd.read_csv('dataset2.csv', sep=',')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(np.zeros((X.shape[1], 1)))
theta = np.zeros((X.shape[1], 1))


parameters = fit(X, y, theta)

print(parameters)
