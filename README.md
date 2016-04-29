# Backpropagating Neural Network with Generalized Delta Learning Rule

Implementation of a back propagating neural network algorithm using the generalized delta rule. Assumes a simple fully connected feedforward network with 1 hidden layer.


### Setup
Before running the network, ensure your parameters and data sets are in place. Place the following files in the main directory:
- `param.txt`
- `in.txt`
- `teach.txt`

`param.txt` should contain 6 lines, each with a single value. The first 3 lines respectively specify, in integers, the number of input, hidden, and output units. The next 3 lines respectively specify, as real values, the learning constant, the momentum constant, and the error criterion.

`in.txt` should contain the input patterns, with one pattern per line. Each pattern should be a sequence of values, separated by single spaces.

`teach.txt` should contain the teaching patterns. Each pattern should be a sequence of values, separated by single spaces.

By default, the paramaters and data sets used are for the iris data set.


### Run
To run the neural network, navigate to the main directory and run the following command: 
	`python main.py`

Alternatively, run:
	`make`
(will clean up *.pyc files before running the main program)

This will launch the text-based user interface with the following options:
```
	1 - Train network with current settings.
	2 - Give the network a pattern and see the predicted output.
	3 – View network settings.
	4 - Change network settings.
	5 – Exit.
```

Enter the number for the option you want. Note: The network cannot predict output without being trained first.


### File Descriptions
Necessary files for running the main interface:
- `main.py` 	contains the code for the text-based user interface
- `backprop.py`	contains the code for the backpropagating neural network algorithm implementation
- `neuron.py`	contains the code for the implementation of individual neurons
- `parse.py`	contains the code for parsing text files containing network parameters, input patterns, and target values

- `param.txt` 	see Setup
- `in.txt` 	see Setup
- `teach.txt` 	see Setup

Helpful but not strictly necessary:
- `test.py` 	contains the code for testing various parameter values or data sets
- `Makefile` 	contains Make rules for convenient cleanup, running, and testing of the network
- `data/*` 	directory containing sample data sets, necessary for running tests in test.py
