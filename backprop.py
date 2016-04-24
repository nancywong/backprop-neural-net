"""Implementation of backpropgating neural network."""
class Neuron:
    """Units for neural network.
    Can be an input, hidden, or output neuron."""

    def __init__(self):
        self.value = 0
        self.pre_neighbors = {} # map input Neurons -> weights
        self.post_neighbors = {} # map output Neurons -> weights


    def update_weight(self, neuron, weight):
        self.pre_neighbors[neuron] = weight


    def __repr__(self):
        s = ('value:  ' + str(self.value) + '\n')
             # '  pre:  \n' + str(self.pre_neighbors.__repr__()) + '\n' +
             # '  post: \n' + str(self.post_neighbors.__repr__()) + '\n')
        return s


class BackPropagationNeuralNetwork:
    """Generalized delta rule network"""

    def __init__(self, params_list, inputs_list, targets_list):
        # Set parameters
        self.num_input_units = int(params_list[0])
        self.num_hidden_units = int(params_list[1])
        self.num_output_units = int(params_list[2])

        self.LEARNING_CONSTANT = params_list[3]
        self.MOMENTUM_CONSTANT = params_list[4]
        self.ERROR_CRITERION = params_list[5]

        # Set target values
        self.targets = targets_list

        # Initialize units
        self.input_units = []
        self.hidden_units = []
        self.output_units = []
        self.initialize_network()


    def initialize_network(self):
        """Create network of Neurons."""

        # Initialize input units
        inputs = []
        for _ in xrange(self.num_input_units):
            n = Neuron()
            inputs.append(n)

        self.input_units = inputs

        # Initialize hidden units
        hidden = []
        for _ in xrange(self.num_hidden_units):
            n = Neuron()
            hidden.append(n)

        self.hidden_units = hidden

        # Initialize output units
        output = []
        for _ in xrange(self.num_output_units):
            n = Neuron()
            output.append(n)

        self.output_units = output

        ## Set neighbors and weights
        # Connect each input unit with each hidden unit
        for i in self.input_units:
            for h in self.hidden_units:
                i.post_neighbors[h] = 0 # TODO initialize with small random value
                h.pre_neighbors[i] = 0 # TODO initialize with small random value

        # Connect each hidden unit with each output unit
        for h in self.hidden_units:
            for o in self.output_units:
                h.post_neighbors[o] = 0 # TODO initialize weights
                o.pre_neighbors[h] = 0 # TODO initialize weights


    def activate(self, val):
        """Activation function for a given unit."""
        pass


    def error(self, output, target):
        """Calculate error between output and target values."""
        pass


    def train(self):
        """Train neural network"""
        # Forward propagation

        # Backward propagation

        # Update weights

        pass


    def predict(self, input_pattern):
        """Given input pattern, predict output"""
        pass


    def __repr__(self):
        """Implement a nice way to print a BPNN object."""
        s = ''
        s += "Input units:\n"
        for i in self.input_units:
            s += i.__repr__()

        s += '\n'
        s += "Hidden units:\n"
        for h in self.hidden_units:
            s += i.__repr__()

        s += '\n'
        s += "Output units:\n"
        for o in self.output_units:
            s += i.__repr__()

        return s


if __name__ == "__main__":
    from parse import parse_params, parse_inputs, parse_targets
    print "Running backprop... \n"

    params = parse_params('param.txt')
    inputs =  parse_inputs('in.txt')
    targets = parse_targets('teach.txt')

    print 'params: ', params
    print 'inputs: ', inputs
    print 'targets:',  targets
    print ''

    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)
    print bpnn
