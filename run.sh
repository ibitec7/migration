#!/bin/bash

source .venv/bin/activate
python src/collection/visa.py && python src/collection/encounter.py && bash download_trends.sh && python src/processing/parse.py