"""Main interface."""
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

    print 'params: ', params
    print 'inputs: ', inputs
    print 'targets:',  targets
    print ''

    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)
    print bpnn
    print ''

    bpnn.train()
    print bpnn

    # Predict
    print 'Predicting 0,0; 0,1; 1,0; 1,1'
    print bpnn.predict([0.0,0.0]) # 0
    print bpnn.predict([0.0,1.0]) # 1
    print bpnn.predict([1.0,0.0]) # 1
    print bpnn.predict([1.0,1.0]) # 0


def test_3bit():
    print '3bit:'

    params = parse_params('data/3bit-parity-param.txt')
    inputs = parse_inputs('data/3bit-parity-in.txt')
    targets = parse_targets('data/3bit-parity-teach.txt')

    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)
    print bpnn
    print ''

    bpnn.train()
    print bpnn
    print 'Predicting...'
    print bpnn.predict([1,1,1])
    print bpnn.predict([1,1,0])
    print bpnn.predict([1,0,1])

    print bpnn


if __name__ == '__main__':
    print "Running backprop... \n"

    test_xor()
    # test_3bit()

