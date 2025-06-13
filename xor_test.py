import numpy as np
from layers import Layer
import backprop as bp

inputs = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))       
expected_output = np.reshape([0, 1, 1, 0], (4, 1, 1))

network = np.array([Layer(2, 3, bp.tanh), Layer(3, 1, bp.tanh)])

#Training the Neural Network
learn_rate = 0.1
for _ in range(100000):
    for inp, exp_out in zip(inputs, expected_output):
        #forward propagation
        output = inp
        for layer in network:
            output = layer.prop_forward(output)
        
        #backward propagation
        out_grad = bp.ddx[bp.mse](output, exp_out)
        for layer in reversed(network):
            out_grad = layer.prop_backward(out_grad, learn_rate)

for _ in inputs:
    output = _
    for layer in network:
        output = layer.prop_forward(output)
    print(output) # Gives output approaching [0 1 1 0] as you increase the number of epochs