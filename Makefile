.PHONY: clean run

run: clean
	python backprop.py

clean:
	find . -name '*.pyc' -delete
