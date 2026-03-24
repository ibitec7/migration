#!/bin/bash

source .venv/bin/activate
hf download sdsc2005-migration/flan-t5-tensorrt-int8_wo-engine --local-dir src/models/tensor-rt
python src/collection/visa.py && python src/collection/encounter.py && bash download_trends.sh && python src/processing/parse.py