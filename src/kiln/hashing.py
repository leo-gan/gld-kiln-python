"""FNV-1a 64 schema hash. Each token is mixed, then a 0x00 step."""

from __future__ import annotations

FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK = 0xFFFFFFFFFFFFFFFF

_SCALARS = {
    "bool",
    "i8",
    "i16",
    "i32",
    "i64",
    "int",
    "u8",
    "u16",
    "u32",
    "u64",
    "uint",
    "f32",
    "f64",
    "str",
    "bytes",
    "tensors",
}


def _mix(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * FNV_PRIME) & MASK
    return (hash_value * FNV_PRIME) & MASK


def _hash_schema(hash_value: int, schema: tuple) -> int:
    kind = schema[0]
    if kind in _SCALARS:
        return _mix(hash_value, kind.encode("ascii"))
    if kind == "list":
        hash_value = _mix(hash_value, b"list<")
        hash_value = _hash_schema(hash_value, schema[1])
        return _mix(hash_value, b">")
    if kind == "opt":
        hash_value = _mix(hash_value, b"opt<")
        hash_value = _hash_schema(hash_value, schema[1])
        return _mix(hash_value, b">")
    if kind == "dict":
        hash_value = _mix(hash_value, b"dict<str,")
        hash_value = _hash_schema(hash_value, schema[1])
        return _mix(hash_value, b">")
    if kind == "struct":
        hash_value = _mix(hash_value, b"struct{")
        for name, field in schema[1]:
            hash_value = _mix(hash_value, name.encode("utf-8"))
            hash_value = _mix(hash_value, b":")
            hash_value = _hash_schema(hash_value, field)
            hash_value = _mix(hash_value, b",")
        return _mix(hash_value, b"}")
    raise ValueError("unknown schema " + str(kind))


def type_hash(schema: tuple) -> int:
    return _hash_schema(FNV_OFFSET, schema)
