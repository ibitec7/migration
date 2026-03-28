#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${PROJECT_ROOT}/.venv/bin/python"

if [ ! -x "${VENV_PYTHON}" ]; then
	echo "Virtual environment Python not found at ${VENV_PYTHON}" >&2
	echo "Run: uv sync" >&2
	exit 1
fi

cd "${PROJECT_ROOT}"

"${VENV_PYTHON}" -m src.main bootstrap --org sdsc2005-migration
"${VENV_PYTHON}" -m src.collection.visa
"${VENV_PYTHON}" -m src.collection.encounter
bash download_trends.sh
"${VENV_PYTHON}" -m src.processing.parse