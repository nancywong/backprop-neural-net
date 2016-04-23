"""
Parse data.

Sample input files: param.txt, in.txt, teach.txt
"""

def parse_params(param_file):
    """Parses file containing parameters.
    e.g. param.txt:
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
            params.append(line.splitlines()) # remove \n at end of line

    return params

def parse_inputs(input_file):
    """Parses file containing input patterns.
    e.g. in.txt:
        containing the input patterns, one pattern per line. each pattern
        is a sequence of values (assume only 0 or 1) separated by single
        spaces.

    Arguments: name of the input patterns file, a string
    Returns: an array of input patterns (an array of values), an array of arrays"""
    input_patterns = []

    with open(input_file, 'r') as f:
        for line in f:
            pattern = line.split()
            input_patterns.append(pattern)

    return input_patterns

def parse_targets(target_file):
    """Parses file containing teacher/target values.
    e.g. teach.txt:
        containing the teaching patterns. one pattern per line.

    Arguments: name of the teacher file, a string
    Returns: an array of teacher values, array of ints or doubles"""
    targets = []

    with open(target_file, 'r') as f:
        for line in f:
            targets.append(line.splitlines()) # remove \n at end of line

    return targets


