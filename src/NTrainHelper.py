'''
Created on 17/12/2023

@author: vassaev
'''
import random as random

class NTrainHelper(object):

    def __init__(self):
        super().__init__()
        
class NTrainFlexibleLambda(NTrainHelper):
    def __init__(self, f, df, em, cnt):
        super().__init__()
        self._f = f
        self._df = df
        self._em = em
        self._cnt = cnt
        
        
    def train(self, n, x, d):
        # the first neuron network training speed
        _lambda = self._em/2
        # the first train
        n.train(x, d, self._f, self._df, _lambda, 1)
        _lambda = n._sd*self._em
        print(f'Lambda = {_lambda}')
        # Training with flexible training speed
        k = 0
        for i in range(1, self._cnt):
            lm = n._sd
            n.train(x, d, self._f, self._df, _lambda , 3)
            print(f'max mistake and standard deviation for iteration ({len(x)},{i}) = {n._mm, n._sd}')
            if n._mm < self._em:
                break;
            if (abs(lm - n._sd)/lm < self._em/8):
                k = k + 1
                if (k > 5):
                    k = 0
                    _lambda = _lambda + _lambda*self._em/2
                    print(f'Lambda = {_lambda}')
            elif (lm + lm*self._em < n._mm):
                k = 0
                _lambda = n._sd*self._em/2
                print(f'Lambda = {_lambda}')
            else:
                k = 0
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
        
        return n._mm

    