# Makefile for the dslr project
.PHONY:  all clean run re

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

all: $(VENV)/bin/python

$(VENV)/bin/python: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -rf $(VENV)

re : clean all

run:
	$(PYTHON) -m init
