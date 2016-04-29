"""Run tests on sample data sets."""
from parse import parse_params, parse_inputs, parse_targets
from backprop import BackPropagationNeuralNetwork


def prop_one_epoch(bpnn):
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


def print_weights(bpnn):
    for (n1, n2) in bpnn.weights:
        print 'N1:', n1
        print 'N2:', n2
        print 'Weight:', bpnn.weights[(n1,n2)]
        print ''


def test_xor():
    print 'XOR:'
    params = parse_params('param.txt')
    inputs =  parse_inputs('in.txt')
    targets = parse_targets('teach.txt')

    res = test(params, inputs, targets)
    return res


def test_3bit():
    print '3bit:'

    params = parse_params('data/3bit-parity-param.txt')
    inputs = parse_inputs('data/3bit-parity-in.txt')
    targets = parse_targets('data/3bit-parity-teach.txt')

    res = test(params, inputs, targets)
    return res


def test_4bit():
    print '4bit:'

    params = parse_params('data/4bit-parity-param.txt')
    inputs = parse_inputs('data/4bit-parity-in.txt')
    targets = parse_targets('data/4bit-parity-teach.txt')

    res = test(params, inputs, targets)
    return res


def test_encoder():
    print 'encoder:'

    params = parse_params('data/encoder-param.txt')
    inputs = parse_inputs('data/encoder-in.txt')
    targets = parse_targets('data/encoder-teach.txt')

    res = test(params, inputs, targets)
    return res


def batch_test_encoder():
    print 'Batch testing encoder:'

    params = parse_params('data/encoder-param.txt')
    inputs = parse_inputs('data/encoder-in.txt')
    targets = parse_targets('data/encoder-teach.txt')

    weights = batch_test(params, inputs, targets, 3)
    learning = batch_test(params, inputs, targets, 1)
    momentum = batch_test(params, inputs, targets, 2)

    return learning, momentum, weights


def test_five_three_five():
    print '5-3-5 network'

    params = parse_params('data/5-3-5-param.txt')
    inputs = parse_inputs('data/5-3-5-in.txt')
    targets = parse_targets('data/5-3-5-teach.txt')

    res = test(params, inputs, targets)
    return res


def batch_test_five_three_five():
    print 'Batch testing 5-3-5 network'

    params = parse_params('data/5-3-5-param.txt')
    inputs = parse_inputs('data/5-3-5-in.txt')
    targets = parse_targets('data/5-3-5-teach.txt')

    learning = batch_test(params, inputs, targets, 1)
    weights = batch_test(params, inputs, targets, 3)
    momentum = batch_test(params, inputs, targets, 2)

    return learning, momentum, weights


def test_iris():
    print 'iris network'

    params = parse_params('data/iris-param.txt')
    inputs = parse_inputs('data/iris-in.txt')
    targets = parse_targets('data/iris-teach.txt')

    res = test(params, inputs, targets)
    return res


def batch_test_iris():
    print 'Batch testing iris network'

    params = parse_params('data/iris-param.txt')
    inputs = parse_inputs('data/iris-in.txt')
    targets = parse_targets('data/iris-teach.txt')

    weights = batch_test(params, inputs, targets, 3)
    learning = batch_test(params, inputs, targets, 1)
    momentum = batch_test(params, inputs, targets, 2)

    return learning, momentum, weights


def test(params, inputs, targets):
    """Given parameters, input and target values, return the number of epochs
    it took for the network to learn."""
    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)

    print 'Training...'
    s = bpnn.train()
    while not s:
        s = bpnn.train

    print 'Predicting...'
    for idx, n in enumerate(inputs):
        outputs = bpnn.predict(n)

        print 'Target:', targets[idx]

        out = []
        for o in outputs:
            out.append(o.value)
        print 'Actual:', out

    print 'Settings:'
    print 'Learning Constant:', bpnn.LEARNING_CONSTANT
    print 'Momentum Constant:', bpnn.MOMENTUM_CONSTANT
    print 'Initial Weight range:', bpnn.INITIAL_WEIGHT_RANGE
    print 'Num epochs:', bpnn.num_epochs

    return bpnn.num_epochs


def batch_test(params, inputs, targets, variable):
    """Given parameters, input and target values, return the number of epochs
    it took for the network to learn, contingent on varying values for
    learning rate, momentum constant, and initial weight ranges."""

    # Values: 0.1 to 1.0
    values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    epochs = []

    # Set custom re-runs
    if variable == 1: # learning
        values = []
    if variable == 2: # momentum
        values = []
    if variable == 3: # weights
        values = []

    for v in values:
        run_epochs = []

        for i in xrange(30):
            bpnn = BackPropagationNeuralNetwork(params, inputs, targets)
            if variable == 1: # learning rate
                bpnn.LEARNING_CONSTANT = v
                print 'Testing learning rates...', i
            elif variable == 2: # momentum
                bpnn.MOMENTUM_CONSTANT = v
                print 'Testing momentum constants...', i
            elif variable == 3: # initial weight range
                bpnn.INITIAL_WEIGHT_RANGE = v
                print 'Testing initial weight ranges...', i
            else:
                print 'Failure. Variable value must be 1, 2, or 3.'
                return []

            s = bpnn.train()
            while not s:
                s = bpnn.train

            run_epochs.append(bpnn.num_epochs)
            print 'num epochs', bpnn.num_epochs

        epochs.append(run_epochs)

        print 'Settings:'
        print 'Learning Constant:', bpnn.LEARNING_CONSTANT
        print 'Momentum Constant:', bpnn.MOMENTUM_CONSTANT
        print 'Initial Weight range:', bpnn.INITIAL_WEIGHT_RANGE
        print ''

    return epochs


def print_num_epochs(test_function):
    print "Running backprop... \n"

    num_epochs = []
    for i in xrange(30):
        ne = test_function()
        num_epochs.append(ne)
        print 'Run', i, ':', ne

    s = 0.0
    for n in num_epochs:
        s += n
    s = s / len(num_epochs)

    print 'num epochs:'
    print num_epochs
    print 'average num epochs:'
    print s


def batch_test_generalization_iris():
    num_epochs = []
    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    for r in ratios:
        ne = test_generalization_iris(r)
        num_epochs.append(ne)

    return num_epochs


def test_generalization_iris(training_set_ratio):
    """Given a ratio k for number of training sets, test how well the network
    generalizes on the last 10 samples."""
    print 'Testing ratio:', training_set_ratio

    params = parse_params('data/iris-param.txt')
    inputs = parse_inputs('data/iris-in.txt')
    targets = parse_targets('data/iris-teach.txt')

    # Split up data sets for each iris species
    SPECIES_SET_SIZE = len(inputs) / 3

    c = training_set_ratio * SPECIES_SET_SIZE
    c = int(c)

    gen_inputs_species1 = inputs[:c]
    gen_inputs_species2 = inputs[50:50+c]
    gen_inputs_species3 = inputs[100:100+c]

    gen_targets_species1 = targets[:c]
    gen_targets_species2 = targets[50:50+c]
    gen_targets_species3 = targets[100:100+c]

    test_inputs = gen_inputs_species1[-10:] + \
                  gen_inputs_species2[-10:] + \
                  gen_inputs_species3[-10:]

    test_targets = gen_targets_species1[-10:] + \
                   gen_targets_species2[-10:] + \
                   gen_targets_species3[-10:]

    gen_inputs = gen_inputs_species1 + \
                 gen_inputs_species2 + \
                 gen_inputs_species3

    gen_targets = gen_targets_species1 + \
                  gen_targets_species2 + \
                  gen_targets_species3

    # Train network on subset
    bpnn = BackPropagationNeuralNetwork(params, gen_inputs, gen_targets)
    bpnn.train()

    # Test network on unseen test set
    errors = []
    for i, ti in enumerate(test_inputs):
        prediction = bpnn.predict(ti)
        error = check_prediction(prediction, test_targets[i])
        errors.append(error)

    print 'Errors:', errors
    return errors


def check_prediction(prediction, target):
    """Returned sum of differences squared as error measure."""

    error_summed = 0.0
    print 'prediction', prediction
    for idx, val in enumerate(prediction):
        err = target[idx] - val
        squared = err*err
        error_summed += squared

    return error_summed


if __name__ == '__main__':
    print_num_epochs(test_xor)
    print_num_epochs(test_five_three_five)
    print_num_epochs(test_encoder)
    print_num_epochs(test_iris)

    batch_test_generalization_iris()

    easy_epochs = batch_test_five_three_five()
    encoder_epochs = batch_test_encoder()
    iris_epochs = batch_test_iris()

    print 'Easy network:'
    print easy_epochs
    print ''

    print 'Encoder network:'
    print encoder_epochs
    print ''

    print 'Iris network:'
    print iris_epochs
    print ''

