"""Implementation of backpropgating neural network."""

class BackPropagationNeuralNetwork:
    """Generalized delta rule network"""

    def __init__(self, params_list, inputs_list, targets_list):
        # Set parameters
        self.num_input_units = params_list[0]
        self.num_hidden_units = params_list[1]
        self.num_output_units = params_list[2]

        self.LEARNING_CONSTANT = params_list[3]
        self.MOMENTUM_CONSTANT = params_list[4]
        self.ERROR_CRITERION = params_list[5]

        # Set input units
        self.inputs = inputs_list

        # Set target values
        self.targets = targets_list


    def train(self):
        """Train neural network"""
        # Forward propagation

        # Backward propagation

        # Update weights

        pass


    def predict(self, input_pattern):
        """Given input pattern, predict output"""
        pass


if __name__ == "__main__":
    from parse import parse_params, parse_inputs, parse_targets
    print "running backprop..."

    params = parse_params('param.txt')
    inputs =  parse_inputs('in.txt')
    targets = parse_targets('teach.txt')

    print 'params: ', params
    print 'inputs: ', inputs
    print 'targets:',  targets

    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)
