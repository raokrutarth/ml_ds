


from math import e
from numpy import log
from pprint import pprint

def sigmoid(inp):
    '''
        Sigmoid activication function for a neuron. Other
        options include perceptron, tanh, ReLU
    '''
    return 1/(1 + e**(-inp))

values = [0.001, 1000, -10, -1, 32.55, -1]
for v in values:
    print('sigmoid({}): {}'.format(v, sigmoid(v)))
print()

def cross_entropy(y, neuron_out):
    '''
        Cost calculation per neuron when compared
        with the expected output (i.e. label). Other
        options include quadratic cost. Cross entropy
        cost has a straight forward derivitive so it
        can be used for gradient descent

        params: float (0, 1)
    '''
    return -1*(y*log(neuron_out) + (1-y)*log(1-neuron_out))

values = [(0.001, 0.00099), (0.0000011, 0.00009), (0.1, 0.1), (0.9910, 0.1), (0.3255, 0.30), (0.30, 0.1)]
res = []
for y, nout in values:
    res.append([y, nout, cross_entropy(y, nout)])
res.sort(key=lambda x:x[2])
for r in res:
    print('cross_entropy_cost({}, {}): {}'.format(r[0], r[1], r[2]))



