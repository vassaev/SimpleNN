'''
Created on 2023-11-10 

@author: Vassaev A.V.
'''
#!C:\Python\Python38\ python
# -*- coding: UTF8 -*-

import numpy as np 
import random as random
import NNet as net

# Activation function
def f(x, m):
    if (x < 0): 
        return x * 0.01
    elif (x > 1): 
        return (x - 1) * 0.01 + 1
    else:
        return x

# Derivative of the activation function
def df(x):
    if (x < 0):
        return 0.01
    elif (x > 1):
        return 0.01
    else:
        return 1

# function to initiate a vector
def fi1(x):
    return random.uniform(-0.5, 0.5)
# function to initiate a matrix
def fi2(x,y):
    return random.uniform(-0.5, 0.5)

# input vector 
x = [
[1],
[2],
[5],
[4],
[3],
[6]
]

# expected results
d = [
[2/8, 3/8, 4/8, 5/8, 6/8],
[3/8, 4/8, 5/8, 6/8, 0],
[3/8, 0, 0, 0, 0],
[6/8, 0, 0, 0, 0],
[0, 0, 0, 0, 0],
[0, 0, 0, 0, 0]
]
# mix data
for i in range(0, len(x) - 1):
    j = random.randint(i, len(x) - 1)
    if i != j:
        y = x[i]
        x[i] = x[j]
        x[j] = y
        y = d[i]
        d[i] = d[j]
        d[j] = y
        

init1 = np.vectorize(fi1)
init2 = np.vectorize(fi2)
random.seed()

# create the neuron network with 3 layers: 
#     1 neuron - input layer, 11 neurons - inner layer, 5 neurons - output layer
n = net.NNet(1, [11], 5, init1, init2)

# the first neuron network training speed
_lambda = 1/32
# the first train
n.train(x, d, f, df, _lambda, 1)
_lambda = n._mm/16
print(f'Lambda = {_lambda}')
# Training with flexible training speed
k = 0
for i in range(1,20000):
    lm = n._mm
    n.train(x, d, f, df, _lambda, 5)
    print(f'max mistake for iteration {i} = {n._mm}')
    if n._mm < 1/16:
        break;
    if (abs(lm - n._mm)/lm < 1/64):
        k = k + 1
        if (k > 8):
            k = 0
            _lambda = _lambda*17/16
            print(f'Lambda = {_lambda}')
    elif (lm*17/16 < n._mm):
        k = 0
        _lambda = n._mm/16
        print(f'Lambda = {_lambda}')
    else:
        k = 0

# print the vector of deviations from targets
for i in range(0, len(d)):
    r = x[i]
    for li in range(0, len(n._l)):
        n._l[li].propagation(r, f)
        r = n._l[li]._f

    m = n._l[-1]._f - d[i]
    print(f'm = {m}')
    
for i in range(0, len(n._l)):
    print(f'w[{i}] = {n._l[i]._w}')
    print(f'b[{i}] = {n._l[i]._b}')

# result of work of the trained network
for i in range(0, len(x)):    
    r = n.propagation([x[i][0]], f)
    for j in range(0, len(r)):
        r[j] = round(r[j]*8)
    k = round(x[i][0]);
    print(f'R({k}) = {r}')
