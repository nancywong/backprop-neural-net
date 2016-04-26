.PHONY: clean run

run: clean
	python main.py

parse: clean
	python parse.py

clean:
	find . -name '*.pyc' -delete
