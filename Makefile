.PHONY: clean run

run: clean
	python backprop.py

parse: clean
	python parse.py

clean:
	find . -name '*.pyc' -delete
