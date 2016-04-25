"""Implementation of backpropgating neural network."""
import random

from neuron import Neuron


### Misc Helper Functions
def small_rand():
    """Returns a small random value for initializing weights."""
    init_rand = 0.0 # can adjust this value
    return random.uniform(-init_rand, init_rand)


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
        self.targets = {} # map output_neuron -> target value
        self.initialize_network()

        # network settings
        self.curr_pattern = [] # initialize for stochasticity in online learning


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

        # Map output units to target values
        for i, o in enumerate(self.output_units):
            self.targets[o] = self.target_values[i]


    def calculate_population_error(self, pattern_error):
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


    def train(self):
        """Train neural network"""

        num_epochs = 0
        population_err = float('inf')  # set to max

        # Batch learning
        while self.ERROR_CRITERION < population_err:
            num_epochs += 1
            if num_epochs % 100 == 0:
                print num_epochs
                print population_err

            random_patterns = self.input_patterns[:] # make a copy
            random.shuffle(random_patterns)
            for pattern in random_patterns:
                self.curr_pattern = pattern
                self.set_input_units(pattern)

                # Forward propagation
                self.feed_forward()

                # Backward propagation
                # pattern_error = self.back_propagate()
                self.back_propagate()

            # TODO: population_err += pattern_error?
            #population_err = self.calculate_population_error(weight_changes)

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

        # 1. Calculate error
        pattern_error = 0
        for i, o in enumerate(self.output_units):
            target = self.targets[o][i]
            output = o.value
            net_error = target - output

            delta = o.calculate_err(net_error)
            pattern_error += delta

        for h in self.hidden_units:
            net_error = 0
            for o in self.output_units:
                net_error += o.err * self.weights[(h,o)]

            delta = h.calculate_err(net_error)
            # TODO: do pattern errors account for hidden units?

        # 2. Update weights using Delta Rule
        for (pre, post) in self.weights:
            n = self.LEARNING_CONSTANT
            # TODO: double check pre and post
            delta = post.err
            val = pre.value

            weight_change = n * delta * val
            self.weights[(pre,post)] += weight_change

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

    for i in xrange(4):
        print 'Feedforward'
        print 'Pattern ' + str(i), bpnn.input_patterns[i]
        bpnn.set_input_units(bpnn.input_patterns[i])
        bpnn.feed_forward()
        print bpnn
        print ''

        print 'Backprop...'
        bpnn.back_propagate()
        print bpnn
        print ''
