#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${PYPI_TOKEN:-}" ]]; then
  echo "PyPI publish skipped: PYPI_TOKEN is unset"
  exit 0
fi
python3 -m pip install -q build twine
python3 -m build
python3 -m twine upload dist/*
