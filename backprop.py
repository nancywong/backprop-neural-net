"""Implementation of backpropgating neural network."""
import random, math


### Misc Helper Functions
def small_rand():
    """Returns a small random value for initializing weights."""
    init_rand = 0.0 # can adjust this value
    return random.uniform(-init_rand, init_rand)


class Neuron:
    """Units for neural network.
    Can be an input, hidden, or output neuron."""

    def __init__(self):
        self.value = 0 # output value


    def activate(self, net_input):
        """Sinoid activation function for a given unit."""
        activation = 1 / (1 + math.exp(-net_input))
        self.value = activation


    def __repr__(self):
        s = ('value:  ' + str(self.value) + '\n')
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

        # Set input and values
        self.input_patterns = inputs_list
        self.target_values = targets_list

        # Initialize units
        self.input_units = []
        self.hidden_units = []
        self.output_units = []
        self.weights = {} # map (Neuron1, Neuron2) -> weight
        self.initialize_network()


    def initialize_network(self):
        """Create network of Neurons."""

        # Initialize units
        self.input_units = [Neuron() for _ in xrange(self.num_input_units)]
        self.hidden_units = [Neuron() for _ in xrange(self.num_hidden_units)]
        self.output_units = [Neuron() for _ in xrange(self.num_output_units)]

        ## Set neighbors and weights with small random values
        # Connect each input unit with each hidden unit
        for i in self.input_units:
            for h in self.hidden_units:
                self.weights[(i,h)] = small_rand()

        # Connect each hidden unit with each output unit
        for h in self.hidden_units:
            for o in self.output_units:
                self.weights[(h,o)] = small_rand()


    def calculate_output_error(self, output, target):
        """Calculate error between output and target values."""
        diff_sum_squared = 0
        # TODO
        """ for each output:
        diff = target - output
        diff_sum_squred += (diff * diff)"""

        pattern_err = 0.5 * diff_sum_squared

        return pattern_err
        pass


    def calculate_hidden_error(self, hidden, target):
        pass


    def calculate_population_error(self, weight_changes):
        """Calculate population error."""
        # TODO
        sum_err = 0
        """ for each epoch (list of patterns):
            sum_err += error(output, target)"""

        n = self.num_output_units # number of output units
        k = len(self.input_patterns) # number of patterns

        pop_err = sum_err / n*k
        return pop_err
        pass


    def consolidate_err(self, batch_err, net_err):
        """Add net error mapping to batch error mapping.
        Return new batch error."""
        # TODO: add net to batch err

        return batch_err
        pass


    def train(self):
        """Train neural network"""

        num_epochs = 0
        population_err = float('inf')  # set to max
        weight_changes = {} # map (neuron1, neuron2) -> change value

        # Batch learning
        while self.ERROR_CRITERION < population_err:
            num_epochs += 1
            if num_epochs % 100 == 0:
                print num_epochs
                print population_err

            batch_err = {} # error per epoch, map (neuron1, neuron2) -> err val

            for pattern in self.input_patterns:
                self.set_input_units(pattern)

                # Forward propagation
                self.feed_forward()

                # Backward propagation
                net_err, weight_changes = self.back_propagate()

                # Update weights using Delta Rule
                self.update_weights(weight_changes)
                batch_err = self.consolidate_err(batch_err, net_err)


            # TODO: population_err += batch_err
            population_err = self.calculate_population_error(weight_changes)

        pass


    def feed_forward(self):
        """Propagate forward and calculate output activations for each neuron
        in the network. Set output units to output activation values."""

        # Calculate input -> hidden unit weights
        for i in self.input_units:
            net_input = i.value
            i.activate(net_input)

        # Calculate hidden -> output unit weights
        for h in self.hidden_units:
            net_input = 0
            for i in self.input_units:
                net_input += i.value * self.weights[(i,h)]

            h.activate(net_input)

        # Calculate output units
        for o in self.output_units:
            net_input = 0
            for h in self.hidden_units:
                net_input += h.value * self.weights[(h,o)]

            o.activate(net_input)


    def back_propagate(self):
        """Propagate backward, using target values to calculate error for
        weight changes for all output and hidden neurons.
        Return a map of (neuron1, neuron2) -> delta value."""
        net_err = {}  # maps (neuron1, neuron2) -> error value
        weight_changes = {}

        # TODO backward calculations
        return net_err, weight_changes


    def set_input_units(self, pattern):
        """Load input units with pattern values."""
        for i, n in enumerate(self.input_units):
            n.value = pattern[i]


    def update_weights(self, weight_changes):
        # TODO: for each entry in weight_changes, update_weight
        pass


    def update_weight(self, in_unit, out_unit, target):
        o = in_unit.post_neighbors[out_unit] # output weight, aka significance
        diff = target # - output?

        delta = self.LEARNING_CONSTANT * o * diff
        in_unit.post_neighbors[out_unit] += delta # update weight


    def predict(self, input_pattern):
        """Given input pattern, predict output by running it through the neural
        net."""
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
            s += h.__repr__()

        s += '\n'
        s += "Output units:\n"
        for o in self.output_units:
            s += o.__repr__()

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
    print ''

    print 'Feedforward'
    print 'Pattern 0:', bpnn.input_patterns[0]
    bpnn.set_input_units(bpnn.input_patterns[0])
    bpnn.feed_forward()
    print bpnn
    print ''

    print 'Feedforward'
    print 'Pattern 1:', bpnn.input_patterns[1]
    bpnn.set_input_units(bpnn.input_patterns[1])
    bpnn.feed_forward()
    print bpnn
    print ''

    print 'Feedforward'
    print 'Pattern 2:', bpnn.input_patterns[2]
    bpnn.set_input_units(bpnn.input_patterns[2])
    bpnn.feed_forward()
    print bpnn
    print ''

    print 'Feedforward'
    print 'Pattern 3:', bpnn.input_patterns[3]
    bpnn.set_input_units(bpnn.input_patterns[3])
    bpnn.feed_forward()
    print bpnn
    print ''

