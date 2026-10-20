# Instructions

Python 3.10 or newer.

```bash
pip install gld-kiln
```

From a git checkout:

```bash
pip install -e .
python -m pytest
gld-kilngen-python testdata/schema/point.json -o point_gen.py
```

The import name is `kiln`. `encode` and `decode` take a schema tuple and a value. `Writer` reuses one buffer across calls. `schema_from_json` reads the JSON Schema subset: booleans, integers with an optional `format`, numbers, strings, binary strings, arrays, objects, `$ref` of the form `#/$defs/Name`, and `additionalProperties` as a dict. A property omitted from `required` is optional. `gld-kilngen-python` writes one Python module of dataclasses. A `tensors` field is not produced from that JSON subset. Build a tensor directory with `encode_tensors` and read it with `decode_tensors`.
