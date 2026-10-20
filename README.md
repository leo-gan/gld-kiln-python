# gld-kiln-python

A from-scratch Kiln codec for Python. The runtime and the code generator are Python. They do not wrap, link, or vendor a C, C++, or Rust Kiln library.

`testdata/reference/kiln_ref.py` is a test oracle for Frozen values. It is not used at run time.

This repository is a standalone library.

Documentation: [Why Kiln](https://leo-gan.github.io/gld-kiln-python/why-kiln/),
[Instructions](https://leo-gan.github.io/gld-kiln-python/instructions/),
[Examples](https://leo-gan.github.io/gld-kiln-python/examples/),
[Techniques](https://leo-gan.github.io/gld-kiln-python/techniques/),
[Test data](https://leo-gan.github.io/gld-kiln-python/test-data/).

## Install

```bash
pip install gld-kiln
```

The import name is `kiln`. The generator command is `gld-kilngen-python`.

```python
from kiln import Writer, decode, encode

point = ("struct", [("x", ("i32",)), ("y", ("i32",))])
writer = Writer()
raw = bytes(writer.encode(point, {"x": 9, "y": -4}))
print(decode(point, raw))
```

## License

MIT. Copyright (c) 2026 Leonid Ganeline.
