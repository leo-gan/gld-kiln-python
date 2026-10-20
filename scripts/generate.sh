#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p tests/generated
python3 -m kiln.codegen testdata/schema/point.json -o tests/generated/point.py
python3 -m kiln.codegen testdata/schema/message.json -o tests/generated/message.py
