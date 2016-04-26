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

        # Initialize units, values, and network settings
        self.weights = {} # map (Neuron1, Neuron2) -> weight
        self.pattern_targets = {} # map pattern (list) -> target (list)
        self.momentum_values = {} # map (n1, n2) -> last weight change value

        self.population_err = float('inf') # gradient descent from max float
        self.curr_pattern = [] # initialize for stochasticity in online learning
        self.curr_target = []

        self.input_units = []
        self.hidden_units = []
        self.output_units = []
        self.bias_unit = Neuron('Bias')

        self.initialize_network()


    def initialize_network(self):
        """Create network of Neurons."""

        # Initialize units
        self.input_units = [Neuron('Input') for _ in xrange(self.num_input_units)]
        self.hidden_units = [Neuron('Hidden') for _ in xrange(self.num_hidden_units)]
        self.output_units = [Neuron('Output') for _ in xrange(self.num_output_units)]

        self.bias_unit.value = 1

        ## Set neighbors and weights with small random values
        # Connect each input unit with each hidden unit
        for i in self.input_units:
            for h in self.hidden_units:
                self.weights[(i,h)] = small_rand()
                self.momentum_values[(i,h)] = 0.0 # set initial momentum

        # Connect each hidden unit with each output unit
        for h in self.hidden_units:
            for o in self.output_units:
                self.weights[(h,o)] = small_rand()
                self.momentum_values[(h,o)] = 0.0 # set initial momentum

        # Connect bias unit to hidden and output units
        b = self.bias_unit
        for h in self.hidden_units:
            self.weights[(b,h)] = small_rand()
            self.momentum_values[(b,h)] = 0.0 # set initial momentum

        for o in self.output_units:
            self.weights[(b,o)] = small_rand()
            self.momentum_values[(b,o)] = 0.0 # set initial momentum


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

        b = self.bias_unit

        # Calculate input -> hidden unit weights
        for i in self.input_units:
            if i.layer is not 'Bias': # exclude bias unit in activation values
                net_input = i.value
                i.activate(net_input)

        # Calculate hidden -> output unit weights
        for h in self.hidden_units:
            net_input = 0.0
            for i in self.input_units:
                net_input += i.value * self.weights[(i,h)]

            net_input += b.value * self.weights[(b,h)]
            h.activate(net_input)

        # Calculate output units
        for o in self.output_units:
            net_input = 0.0
            for h in self.hidden_units:
                net_input += h.value * self.weights[(h,o)]

            net_input += b.value * self.weights[(b,o)]
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

            pattern_error.append(net_error)
            o.calculate_err(net_error)

        for h in self.hidden_units:
            # Calculate error between output and hidden units
            net_error = 0
            for o in self.output_units:
                net_error += o.err * self.weights[(h,o)]

            h.calculate_err(net_error)

        for n in self.input_units:
            # Calculate error between hidden and input units
            net_error = 0
            for h in self.hidden_units:
                net_error += h.err * self.weights[(n,h)]

            n.calculate_err(net_error)

        # Calculate error for bias unit
        b = self.bias_unit
        net_error = 0

        for h in self.hidden_units:
            net_error += h.err * self.weights[(b,h)]
        for o in self.output_units:
            net_error += o.err * self.weights[(b,o)]

        b.calculate_err(net_error)

        # 2. Update weights using Delta Rule
        for (pre, post) in self.weights:
            n = self.LEARNING_CONSTANT
            delta = post.err
            output = pre.value
            momentum = self.MOMENTUM_CONSTANT * self.momentum_values[(pre,post)]

            weight_change = n*delta*output + momentum
            self.weights[(pre,post)] += weight_change

            # Store previous weight change
            self.momentum_values[(pre,post)] = weight_change

        return pattern_error


    def set_input_units(self, pattern):
        """Load input units with pattern values."""
        for idx, val in enumerate(pattern):
            self.input_units[idx].value = val


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
