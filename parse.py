"""
Parse data.

Sample input files: param.txt, in.txt, teach.txt
"""

def parse_params(param_file):
    """Parses file containing parameters.
    i.e. param.txt:
        containing 6 lines, each with a single value. First 3 lines
        (integers) specify the number of input, hidden, and output units
        respectively. The next 3 lines (real numbers) should specify the
        learning constant, the momentum constant, and the learning/error
        criterion.

    Arguments: name of the parameter file, a string
    Returns: an array of parameter values, array of ints or doubles"""
    params = []

    with open(param_file, 'r') as f:
        for line in f:
            l = line.splitlines()[0] # Remove \n at end of line
            params.append(float(l)) # Convert string to float

    return params

def parse_inputs(input_file):
    """Parses file containing input patterns.
    i.e. in.txt:
        containing the input patterns, one pattern per line. each pattern
        is a sequence of values (assume only 0 or 1) separated by single
        spaces.

    Arguments: name of the input patterns file, a string
    Returns: an array of input patterns (an array of values), an array of arrays"""
    input_patterns = []

    with open(input_file, 'r') as f:
        for line in f:
            # Convert string to list of floats
            pattern = [float(p) for p in line.split()]
            input_patterns.append(pattern)

    return input_patterns

def parse_targets(target_file):
    """Parses file containing teacher/target values.
    i.e. teach.txt:
        containing the teaching patterns. one pattern per line.

    Arguments: name of the teacher file, a string
    Returns: an array of teacher values, array of ints or doubles"""
    targets = []

    with open(target_file, 'r') as f:
        for line in f:
            # Convert string to list of floats
            pattern = [float(p) for p in line.split()]
            targets.append(pattern)

    return targets


if __name__ == "__main__":
    ## Test parsing
    # Test XOR
    params = parse_params('param.txt')
    inputs =  parse_inputs('in.txt')
    targets = parse_targets('teach.txt')

    print 'params: ', params
    print 'inputs: ', inputs
    print 'targets:',  targets
    print ''


    # Test Random Easy data set parsing
    params2 = parse_params('data/5-3-5-param.txt')
    inputs2 =  parse_inputs('data/5-3-5-in.txt')
    targets2 = parse_targets('data/5-3-5-teach.txt')

    print 'params: ', params2
    print 'inputs: ', inputs2
    print 'targets:',  targets2
    print ''


    # Test 3bit parity data set parsing
    params3 = parse_params('data/3bit-parity-param.txt')
    inputs3 =  parse_inputs('data/3bit-parity-in.txt')
    targets3 = parse_targets('data/3bit-parity-teach.txt')

    print 'params: ', params3
    print 'inputs: ', inputs3
    print 'targets:',  targets3
    print ''


    # Test 4bit parity data set parsing
    params4 = parse_params('data/4bit-parity-param.txt')
    inputs4 =  parse_inputs('data/4bit-parity-in.txt')
    targets4 = parse_targets('data/4bit-parity-teach.txt')

    print 'params: ', params4
    print 'inputs: ', inputs4
    print 'targets:',  targets4
    print ''


    # Test encoder data set parsing
    params5 = parse_params('data/encoder-param.txt')
    inputs5 =  parse_inputs('data/encoder-in.txt')
    targets5 = parse_targets('data/encoder-teach.txt')

    print 'params: ', params5
    print 'inputs: ', inputs5
    print 'targets:',  targets5
    print ''


    # Test iris data set parsing
    params6 = parse_params('data/iris-param.txt')
    inputs6 =  parse_inputs('data/iris-in.txt')
    targets6 = parse_targets('data/iris-teach.txt')

    print 'params: ', params6
    print 'inputs: ', inputs6
    print 'targets:',  targets6
    print ''
