"""Main interface."""
from parse import parse_params, parse_inputs, parse_targets
from backprop import BackPropagationNeuralNetwork

def print_intro():
    print 'Backpropagating Neural Network using Generalized Delta Rule.'
    print 'Paramater, input, and target files should be in the current \n\
            directory and named, respectively: \n \
            param.txt \n \
            in.txt \n \
            teach.txt'


def print_menu():
    print 'Please input your selection and hit Enter.'
    print '1 - Train network with current settings.'
    print '2 - Give the network a pattern and see the predicted output.'
    print '3 - Change network settings.'
    print '4 - Exit.'


def predict(bpnn):
    prompt = 'Please input' + str(bpnn.num_input_units) + 'numbers, separated by a space.'
    print prompt
    print 'e.g. if there are 2 input units, type in \'0 0\''

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


def change_settings(bpnn):
    print 'What do you want to set the learning constant to?'
    usr_input = raw_input('Learning constant: ')
    bpnn.LEARNING_CONSTANT = float(usr_input)

    print 'What do you want to set the momentum constant to?'
    usr_input = raw_input('Momentum constant: ')
    bpnn.MOMENTUM_CONSTANT = float(usr_input)

    print 'What do you want to set the error criterion to?'
    usr_input = raw_input('Error criterion: ')
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

    # User input
    while selection != 4:
        print ''
        print_menu()
        usr_input = raw_input('Enter a number: ')

        try:
            selection = int(usr_input)
        except ValueError:
            print 'Please enter a number between 1 to 4.'

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
            # Change network settings
            change_settings(bpnn)

        elif selection == 4:
            break

        else:
            print 'Please enter a number between 1 to 4.'

    print 'Bye!'
