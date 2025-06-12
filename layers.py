import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None

        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def prop_forward(self, input):
        self.input = input
        return np.dot(self.weight, self.input) + self.bias

    def prop_backward(self, output):
        pass
