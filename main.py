"""Main interface."""
from parse import parse_params, parse_inputs, parse_targets
from backprop import BackPropagationNeuralNetwork

def print_intro():
    print 'Backpropagating Neural Network using Generalized Delta Rule.'
    print 'Paramater, input, and target files should be in the current directory and named, respectively: \n \
            param.txt \n \
            in.txt \n \
            teach.txt'


def print_menu():
    print 'Please input your selection and hit Enter.'
    print '1 - Train network with current settings.'
    print '2 - Give the network a pattern and see the predicted output.'
    print '3 - View network settings.'
    print '4 - Change network settings.'
    print '5 - Exit.'


def predict(bpnn):
    prompt = 'Please input ' + str(bpnn.num_input_units) + ' numbers, separated by a space.'
    print prompt
    print 'e.g. if there are 4 input units, type in \'0.918000  0.416000  0.949000  0.831000\''

    pattern = []
    usr_input = raw_input('Pattern: ')

    try:
        str_pattern = usr_input.split()
        while len(str_pattern) != bpnn.num_input_units:
            print prompt
            usr_input = raw_input('Pattern: ')
            str_pattern = usr_input.split()

        pattern = map(float, pattern) # map strings to int values

    except ValueError:
        print prompt
        usr_input = raw_input('Pattern: ')
        str_pattern = usr_input.split()

    print 'Predicting...'
    output = bpnn.predict(pattern)

    print 'Output:'
    for o in output:
        print o.value


def view_settings(bpnn):
    print 'The learning constant is currently set to', bpnn.LEARNING_CONSTANT
    print 'The momentum constant is currently set to', bpnn.MOMENTUM_CONSTANT
    print 'The error criterion is currently set to', bpnn.ERROR_CRITERION
    print 'The initial weight range is currently set to +=', bpnn.INITIAL_WEIGHT_RANGE


def change_settings(bpnn):
    print 'The learning constant is currently set to', bpnn.LEARNING_CONSTANT
    print 'What do you want to set the learning constant to?'
    usr_input = raw_input('Learning constant: ')
    bpnn.LEARNING_CONSTANT = float(usr_input)

    print 'The momentum constant is currently set to', bpnn.MOMENTUM_CONSTANT
    print 'What do you want to set the momentum constant to?'
    usr_input = raw_input('Momentum constant: ')
    bpnn.MOMENTUM_CONSTANT = float(usr_input)

    print 'The error criterion is currently set to', bpnn.ERROR_CRITERION
    print 'What do you want to set the error criterion to?'
    usr_input = raw_input('Error criterion: ')
    bpnn.ERROR_CRITERION = float(usr_input)

    print 'The initial weight range is currently set to +=', bpnn.INITIAL_WEIGHT_RANGE
    print 'What do you want to set the initial weight range to?'
    usr_input = raw_input('Initial weight range: +-')
    bpnn.ERROR_CRITERION = float(usr_input)

    print 'Settings changed. If you want to change the number of input, hidden, or output units, please do so in the first three lines of the param.txt file.'


if __name__ == '__main__':
    print_intro()

    selection = 0
    trained = False

    # Initialize network
    params = parse_params('param.txt')
    inputs =  parse_inputs('in.txt')
    targets = parse_targets('teach.txt')

    bpnn = BackPropagationNeuralNetwork(params, inputs, targets)

    QUIT_VAL = 5
    # User input
    while selection != QUIT_VAL:
        print ''
        print_menu()
        usr_input = raw_input('Enter a number: ')

        try:
            selection = int(usr_input)
        except ValueError:
            print 'Please enter a number between 1 to', QUIT_VAL

        if selection == 1:
            # Train network
            print 'Training network...'
            bpnn.train()
            trained = True

        elif selection == 2:
            # Give network a pattern to predict
            if not trained:
                print 'Please train the network first.'
            else:
                predict(bpnn)

        elif selection == 3:
            # View network settings
            view_settings(bpnn)

        elif selection == 4:
            # Change network settings
            change_settings(bpnn)

        elif selection == QUIT_VAL:
            break

        else:
            print 'Please enter a number between 1 to', QUIT_VAL

    print 'Bye!'
