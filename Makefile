.PHONY: clean run

run: clean
	python main.py

test: clean
	python test.py

parse: clean
	python parse.py

clean:
	find . -name '*.pyc' -delete
