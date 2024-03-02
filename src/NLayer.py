import numpy as np 
class NLayer:
    def __init__(self, input_count, neuron_count, init1, init2):
        # vector of values
        self._v = np.zeros(neuron_count)
        # vector of activation functions
        self._f = np.zeros(neuron_count)
        # matrix
        self._w = np.fromfunction(init2, [neuron_count, input_count])
        # vector
        self._b = np.fromfunction(init1, [neuron_count])
        # mistake of the previous layer
        self._m = np.zeros(input_count)

    def clone(self):
        nl = NLayer()
        nl._v = self._v.copy()
        nl._f = self._f.copy()
        nl._w = self._w.copy()
        nl._b = self._b.copy()
        nl._m = self._m.copy()
        return nl
        
    def propagation(self, rx, function_activation):
        mi = 0
        self._v[0] = self._b[0] + np.dot(self._w[0], rx)
        for i in range(1, len(self._b)):
            self._v[i] = self._b[i] + np.dot(self._w[i], rx)
            if (self._v[i] > self._v[mi]):
                mi = i
                
        for i in range(0, len(self._b)):
            self._f[i] = function_activation(self._v[i], self._v[mi])
        self._rx = rx
        return self._f

    def back_propagation(self, e, function_derivative_activation, learning_rate):
        for i in range(0, len(self._m)):
            t = [r[i] for r in self._w] 
            self._m[i] = np.dot(t, e)

        for i in range(0, len(self._b)):
            dlt = e[i] * function_derivative_activation(self._v[i])
            s = learning_rate * dlt
            self._b[i] = self._b[i] - s
            for j in range(0, len(self._m)):
                self._w[i][j] = self._w[i][j] - s * self._rx[j]

        return self._m
'''    
class LayerDef(NLayer):
    def __init__(self, b, w):
        ln = len(b)
        self._v = np.zeros(ln)
        self._f = np.zeros(ln)
        self._w = w
        self._b = b
        self._m = np.zeros(len(w[0]))
'''