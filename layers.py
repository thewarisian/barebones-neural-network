import numpy as np
import backprop as bp

class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.input = None
        self.output_unactivated = None
        self.output = None

        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

        self.activation = activation

    def prop_forward(self, input):
        self.input = input
        self.output_unactivated = np.dot(self.weight, self.input) + self.bias
        self.output = self.output_unactivated if self.activation is None else self.activation(self.output_unactivated)
        return self.output

    def prop_backward(self, out_grad, learn_rate):
        in_grad, self.weight, self.bias = bp.grad_descent(self.input, self.output_unactivated, 
                                                          self.weight, self.bias, 
                                                          out_grad, learn_rate, self.activation) 
        return in_grad