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

    test(params, inputs, targets)


def test_3bit():
    print '3bit:'

    params = parse_params('data/3bit-parity-param.txt')
    inputs = parse_inputs('data/3bit-parity-in.txt')
    targets = parse_targets('data/3bit-parity-teach.txt')

    test(params, inputs, targets)


def test_4bit():
    print '4bit:'

    params = parse_params('data/4bit-parity-param.txt')
    inputs = parse_inputs('data/4bit-parity-in.txt')
    targets = parse_targets('data/4bit-parity-teach.txt')

    test(params, inputs, targets)


def test_encoder():
    print 'encoder:'

    params = parse_params('data/encoder-param.txt')
    inputs = parse_inputs('data/encoder-in.txt')
    targets = parse_targets('data/encoder-teach.txt')

    test(params, inputs, targets)


def test_five_three_five():
    print '5-3-5 network'

    params = parse_params('data/5-3-5-param.txt')
    inputs = parse_inputs('data/5-3-5-in.txt')
    targets = parse_targets('data/5-3-5-teach.txt')

    test(params, inputs, targets)


def test_iris():
    print 'iris network'

    params = parse_params('data/iris-param.txt')
    inputs = parse_inputs('data/iris-in.txt')
    targets = parse_targets('data/iris-teach.txt')

    test(params, inputs, targets)


def test(params, inputs, targets):
    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)

    print 'Training...'
    bpnn.train()

    print 'Predicting...'
    for idx, n in enumerate(inputs):
        outputs = bpnn.predict(n)

        print 'Target:', targets[idx]

        out = []
        for o in outputs:
            out.append(o.value)
        print 'Actual:  ', out


if __name__ == '__main__':
    print "Running backprop... \n"

    test_xor()
    # test_3bit()

