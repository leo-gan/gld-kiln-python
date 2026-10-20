#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
python3 -m kiln.codegen testdata/schema/point.json -o "$tmpdir/point.py"
python3 -m kiln.codegen testdata/schema/point.json -o "$tmpdir/point2.py"
python3 -m kiln.codegen testdata/schema/message.json -o "$tmpdir/message.py"
python3 -m kiln.codegen testdata/schema/message.json -o "$tmpdir/message2.py"
cmp "$tmpdir/point.py" "$tmpdir/point2.py"
cmp "$tmpdir/message.py" "$tmpdir/message2.py"
