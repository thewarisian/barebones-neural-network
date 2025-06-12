import numpy as np
import backprop as bp

class Layer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None

        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def prop_forward(self, input):
        self.input = input
        return np.dot(self.weight, self.input) + self.bias

    def prop_backward(self, out_grad, learn_rate):
        in_grad, self.weight, self.bias = bp.grad_descent(self.input, self.weight, self.bias, out_grad, learn_rate) 
        return in_grad
