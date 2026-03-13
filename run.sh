#!/bin/bash

source .venv/bin/activate
python src/collection/visa.py && python src/collection/encounter.py && python src/processing/parse.py