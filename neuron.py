import math

class Neuron:
    """Units for neural network.
    Can be an input, hidden, or output neuron."""

    def __init__(self):
        self.value = 0.0 # activation value
        self.err = 0.0 # error term for back propagating
        self.layer = '' # Input, Hidden, or Output (used for printing)


    def activate(self, net_input):
        """Sinoid activation function for a given unit.
        Stores activation output during forward pass to use in backward pass."""
        activation = 1.0 / (1.0 + math.exp(-net_input))
        self.value = activation # store output for backward pass


    def calculate_err(self, net_err):
        """If output unit, net error = target value - output value.
        If hidden unit, net error = sum of output units' error.

        Since the network has hidden layer(s), want to use the derivative of
        the activation function.

        Stores error term in presynaptic neuron during backward pass to use for
        weight changes."""
        error_term = net_err * (self.value * (1.0 - self.value)) # chain rule
        self.err = error_term # store error for weight change


    def __repr__(self):
        s = (self.layer + ' Unit\n' +
             'Value:  ' + str(self.value) + '\n' +
             'Error:  ' + str(self.err) + '\n')
        return s


