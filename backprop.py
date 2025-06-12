import numpy as np

def grad_descent(input, weight, bias, out_grad, learn_rate):
    in_grad = np.dot(weight.T, out_grad)
    weight -= learn_rate * np.dot(out_grad, input.T)
    bias -= learn_rate * out_grad

    return in_grad, weight, bias
