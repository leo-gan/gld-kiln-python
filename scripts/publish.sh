#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYPIRC_FILE="${PYPIRC_FILE:-$(cd "$(dirname "$0")/../.." && pwd)/anonymizer/.pypirc}"

python3 -m pip install -q build twine
rm -rf dist
python3 -m build

if [[ -n "${PYPI_TOKEN:-}" ]]; then
  python3 -m twine upload dist/* -u __token__ -p "$PYPI_TOKEN"
elif [[ -f "$PYPIRC_FILE" ]]; then
  echo "Uploading to PyPI with ${PYPIRC_FILE}"
  python3 -m twine upload --config-file "$PYPIRC_FILE" -r pypi dist/*
else
  echo "PyPI publish skipped: PYPI_TOKEN is unset and ${PYPIRC_FILE} is missing"
  exit 0
fi
