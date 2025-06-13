import numpy as np

#Gradient descent
def grad_descent(input, output, weight, bias, out_grad, learn_rate, activation):
    in_grad = np.dot(weight.T, out_grad)
    if activation is not None:
        out_grad *= ddx[tanh](output)

    weight -= learn_rate * np.dot(out_grad, input.T)
    bias -= learn_rate * out_grad

    return in_grad, weight, bias

#Activation functions
tanh = lambda x: np.tanh(x)
tanh_prime = lambda x: 1 - np.tanh(x)**2

#Loss functions
def mse(nn_output, correct_output):
    return np.mean(np.power((correct_output - nn_output), 2))
def mse_prime(nn_output, correct_output):
    return 2*(nn_output - correct_output) / np.size(correct_output)

#differentiate
ddx = {tanh : tanh_prime, mse : mse_prime}