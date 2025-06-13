import numpy as np

def grad_descent(input, output, weight, bias, out_grad, learn_rate, activation, activation_prime):
    in_grad = np.dot(weight.T, out_grad)
    if activation is not None:
        in_grad *= activation_prime(output)

    weight -= learn_rate * np.dot(out_grad, input.T)
    bias -= learn_rate * out_grad

    return in_grad, weight, bias
