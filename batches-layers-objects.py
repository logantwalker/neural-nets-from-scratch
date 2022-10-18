from turtle import forward
import numpy as np

np.random.seed(0)

# X typically denotes your data set in ML space
X = [
    [1.0,2.0,3.0,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # create weights the size of n_inputs * how many neurons we want in the layer
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)

print(layer2.output)