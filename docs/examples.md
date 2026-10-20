# Examples

`Point` has two `i32` fields. Kiln writes them in that order. Eight little-endian bytes, and no names.

```python
from kiln import Writer, decode, encode

point = ("struct", [("x", ("i32",)), ("y", ("i32",))])
writer = Writer(64)
raw = writer.encode(point, {"x": 9, "y": -4})
print(decode(point, raw), len(raw))
```

The program prints `{'x': 9, 'y': -4} 8`.

A tensor directory keeps the numeric payload as the array. The view's `data` field is a slice of the input buffer.

```python
from kiln import decode_tensors, encode_tensors
import struct

block = encode_tensors([
    {"name": None, "dtype": "f32", "shape": [2], "data": struct.pack("<ff", 1.0, -2.0)},
])
directory, length = decode_tensors(block)
payload = directory["tensors"][0]["data"]
print(length, bytes(payload).hex())
```
