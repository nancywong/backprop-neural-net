import math

class Neuron:
    """Units for neural network.
    Can be an input, hidden, or output neuron."""

    def __init__(self):
        self.value = 0 # activation value
        self.err = 0 # error term for back propagating (should be 0 in input units)


    def activate(self, net_input):
        """Sinoid activation function for a given unit."""
        activation = 1 / (1 + math.exp(-net_input))
        self.value = activation # store output for backward pass


    def calculate_err(self, net_err):
        """If output unit, net error = target value - output value.
        If hidden unit, net error = sum of output units' error."""
        error_term = net_err * self.value * (1 - self.value)
        self.err = error_term # store error for weight change

        return error_term


    def __repr__(self):
        s = ('Value:  ' + str(self.value) + '\n' +
             'Error:  ' + str(self.err) + '\n')
        return s


