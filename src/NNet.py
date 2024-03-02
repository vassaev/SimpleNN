import NLayer as layer
import numpy as np
from scipy.cluster._hierarchy import nn_chain

class NNet:
    def __init__(self, size_input, size_layers, size_output, init1, init2):
        l = []
        i = size_input
        for j in range(0, len(size_layers)):
            l.append(layer.NLayer(i,size_layers[j],init1,init2))
            i = size_layers[j]
        
        l.append(layer.NLayer(i,size_output,init1,init2))
        self._l = l
        self._negative_result_is_zerro = 0
        
    def clone(self):
        nn = NNet()
        nn._l = []
        for i in range(0, len(self._l)):
            nn._l.append(self._l[i].clone())
        nn._mm = self._mm
        nn._negative_result_is_zerro = self._negative_result_is_zerro
        return nn
    
    # Function to get a result
    def propagation(self, x, function_activation):
        r = x
        for i in range(0, len(self._l)):
            r = self._l[i].propagation(r, function_activation)
        return r
    
    # Train function
    def train(self, x, d, f, df, _lambda, cnt):
        for s in range(0, cnt):
            for i in range(0, len(d)):
                r = self.propagation(x[i], f)
                m = r - d[i]
                
                for li in range(len(self._l) - 1, -1, -1):
                    m = self._l[li].back_propagation(m, df, _lambda)

        self._mm = 0
        self._sd = 0
        for i in range(0, len(d)):
            r = self.propagation(x[i], f)
            m = r - d[i]
            t = max(abs(m))
            if self._mm < t:
                self._mm = t
            sd = 0
            j = len(m) - 1;
            while j >= 0:
                sd += m[j] * m[j]
                j -= 1
            self._sd += sd/len(m)
        self._sd = np.sqrt(self._sd/len(d))
