"""Implementation of backpropgating neural network."""
import random

from neuron import Neuron


### Misc Helper Functions
def small_rand():
    """Returns a small random value for initializing weights."""
    init_rand = 1.0 # can adjust this value
    return random.uniform(-init_rand, init_rand)


def sum_squares(pattern_errors):
    """Given a list of pattern error values, return a sum of squares of
    differences."""
    s = 0.0
    for err in pattern_errors:
        s += err*err

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
        self.pattern_targets = {} # map pattern (list) -> target (list)
        self.initialize_network()

        # network settings
        self.population_err = float('inf') # gradient descent from max float
        self.curr_pattern = [] # initialize for stochasticity in online learning
        self.curr_target = []


    def initialize_network(self):
        """Create network of Neurons."""

        # Initialize units
        self.input_units = [Neuron() for _ in xrange(self.num_input_units)]
        self.hidden_units = [Neuron() for _ in xrange(self.num_hidden_units)]
        self.output_units = [Neuron() for _ in xrange(self.num_output_units)]

        # Set unit type
        for n in self.input_units:
            n.layer = 'Input'
        for h in self.hidden_units:
            h.layer = 'Hidden'
        for o in self.output_units:
            o.layer = 'Output'

        ## Set neighbors and weights with small random values
        # Connect each input unit with each hidden unit
        for i in self.input_units:
            for h in self.hidden_units:
                self.weights[(i,h)] = small_rand()

        # Connect each hidden unit with each output unit
        for h in self.hidden_units:
            for o in self.output_units:
                self.weights[(h,o)] = small_rand()

        # Map input patterns to target values
        for i, p in enumerate(self.input_patterns):
            t = tuple(p) # hash pattern as key to find matching target value
            self.pattern_targets[t] = self.target_values[i]


    def set_population_error(self, pattern_errors):
        """Given a list of pattern errors, calculate the population error."""
        error_summed = 0.0
        for pattern_error in pattern_errors:
            pattern_error_sum_sq = sum_squares(pattern_error)
            error_summed += pattern_error_sum_sq

        n = float(self.num_output_units) # number of output units
        k = float(len(self.input_patterns)) # number of patterns

        self.population_err = error_summed / (n*k)


    def train(self):
        """Train neural network"""

        num_epochs = 0

        # Batch learning
        while self.ERROR_CRITERION < self.population_err:
            # Print # of epochs and current population error every 100 epochs
            num_epochs += 1
            if num_epochs % 100 == 0:
                print '# epochs:', num_epochs
                print 'Pop err: ', self.population_err
                print ''
              #  print self, self.weights

            pattern_errors = []

            # For online learning, patterns must be presented in random order
            random_patterns = self.input_patterns[:] # make a copy
            random.shuffle(random_patterns)
            for pattern in random_patterns:
                self.curr_pattern = pattern
                self.set_input_units(pattern)
                self.curr_target = self.pattern_targets[tuple(pattern)]

                # Forward propagation
                self.feed_forward()

                # Backward propagation - calculate error and adjust weights
                pattern_error = self.back_propagate()
                pattern_errors.append(pattern_error)

            self.set_population_error(pattern_errors)


    def feed_forward(self):
        """Propagate forward and calculate output activations for each neuron
        in the network. Set output units to output activation values."""

        # Calculate input -> hidden unit weights
        for i in self.input_units:
            net_input = i.value
            i.activate(net_input)

        # Calculate hidden -> output unit weights
        for h in self.hidden_units:
            net_input = 0.0
            for i in self.input_units:
                net_input += i.value * self.weights[(i,h)]

            h.activate(net_input)

        # Calculate output units
        for o in self.output_units:
            net_input = 0.0
            for h in self.hidden_units:
                net_input += h.value * self.weights[(h,o)]

            o.activate(net_input)


    def back_propagate(self):
        """Propagate backward, using target values to calculate error for
        weight changes for all output and hidden neurons.
        Return a list of error values for the current pattern."""

        # 1. Calculate error
        pattern_error = []
        for i, o in enumerate(self.output_units):
            # Calculate error between output and targets
            target = float(self.curr_target[i])
            output = o.value
            net_error = target - output
            # print 'target, output', target, output
            # print 'net err', net_error

            pattern_error.append(net_error)
            delta = o.calculate_err(net_error)

        for h in self.hidden_units:
            # Calculate error between output and hidden units
            net_error = 0
            for o in self.output_units:
                net_error += o.err * self.weights[(h,o)]

            delta = h.calculate_err(net_error)

        for n in self.input_units:
            # Calculate error between hidden and input units
            net_error = 0
            for h in self.hidden_units:
                net_error += h.err * self.weights[(n,h)]

            delta = n.calculate_err(net_error)

        # 2. Update weights using Delta Rule
        for (pre, post) in self.weights:
            n = self.LEARNING_CONSTANT
            delta = post.err
            val = pre.value

            weight_change = n * delta * val
            self.weights[(pre,post)] += weight_change
            #print 'weight change'
            #print 'pre, post', pre, post
            #print 'n, delta, val', n, delta, val
            #print 'change', weight_change

        return pattern_error


    def set_input_units(self, pattern):
        """Load input units with pattern values."""
        for i, n in enumerate(self.input_units):
            n.value = pattern[i]


    def predict(self, input_pattern):
        """Given input pattern, predict output by running it through the neural
        net."""
        self.set_input_units(input_pattern)
        self.feed_forward()

        return self.output_units


    def __repr__(self):
        """Implement a nice way to print a BPNN object."""
        s = ''
        s += "Input units:\n"
        for i in self.input_units:
            s += i.__repr__()
            s += '\n'

        s += '\n'
        s += "Hidden units:\n"
        for h in self.hidden_units:
            s += h.__repr__()
            s += '\n'

        s += '\n'
        s += "Output units:\n"
        for o in self.output_units:
            s += o.__repr__()
            s += '\n'

        return s
