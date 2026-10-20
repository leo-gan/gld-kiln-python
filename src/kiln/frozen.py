"""Frozen positional values. The caller passes the schema."""

from __future__ import annotations

import array
import struct
import sys
import threading
from typing import Any

from kiln.error import KilnError
from kiln.hashing import type_hash
from kiln.tensor import CAP, HEADER, decode_tensors, encode_tensors

_WIDTH = {
    "bool": 1,
    "i8": 1,
    "u8": 1,
    "i16": 2,
    "u16": 2,
    "i32": 4,
    "u32": 4,
    "f32": 4,
    "i64": 8,
    "u64": 8,
    "int": 8,
    "uint": 8,
    "f64": 8,
}
_PACK = {
    "i8": "<b",
    "i16": "<h",
    "i32": "<i",
    "i64": "<q",
    "int": "<q",
    "u8": "<B",
    "u16": "<H",
    "u32": "<I",
    "u64": "<Q",
    "uint": "<Q",
    "f32": "<f",
    "f64": "<d",
}
_STRUCT = {kind: struct.Struct(fmt) for kind, fmt in _PACK.items()}
_ARRAY_CODE = {
    "i8": "b",
    "i16": "h",
    "i32": "i",
    "i64": "q",
    "int": "q",
    "u16": "H",
    "u32": "I",
    "u64": "Q",
    "uint": "Q",
    "f32": "f",
    "f64": "d",
}
_ARRAY = {}
if sys.byteorder == "little":
    for _kind, _code in _ARRAY_CODE.items():
        if array.array(_code).itemsize == _WIDTH[_kind]:
            _ARRAY[_kind] = _code
_LIST_STRUCT: dict[tuple[str, int], struct.Struct] = {}


def _list_struct(kind: str, count: int) -> struct.Struct:
    key = (kind, count)
    found = _LIST_STRUCT.get(key)
    if found is None:
        found = struct.Struct("<" + (_PACK[kind][1] * count))
        if len(_LIST_STRUCT) > 32:
            _LIST_STRUCT.clear()
        _LIST_STRUCT[key] = found
    return found


def min_size(schema: tuple) -> int:
    kind = schema[0]
    if kind in _WIDTH:
        return _WIDTH[kind]
    if kind == "str" or kind == "bytes":
        return 4
    if kind == "tensors":
        return HEADER
    if kind == "opt":
        return 1
    if kind == "list" or kind == "dict":
        return 4
    if kind == "struct":
        return sum(min_size(field) for _, field in schema[1])
    raise KilnError("TYPE")


def _fit(count: int, each: int, unread: int, cap: int) -> None:
    if count > cap:
        raise KilnError("LIMIT")
    if each > 0 and count > (2**64 - 1) // each:
        raise KilnError("LIMIT")
    if count * each > unread:
        raise KilnError("EOF")


class Writer:
    """Reusable encode buffer. `encode` resets the length and keeps the storage."""

    def __init__(self, capacity: int = 64) -> None:
        self.buf = bytearray(capacity)
        self.end = 0

    def _reserve(self, extra: int) -> None:
        need = self.end + extra
        if need > len(self.buf):
            grown = bytearray(max(need, len(self.buf) * 2))
            grown[: self.end] = self.buf[: self.end]
            self.buf = grown

    def encode(self, schema: tuple, value: Any, *, cap: int = CAP) -> memoryview:
        self.end = 0
        self._write(schema, value, cap, 0)
        return memoryview(self.buf)[: self.end]

    def encode_framed(self, schema: tuple, value: Any, *, cap: int = CAP) -> memoryview:
        body = encode(schema, value, cap=cap)
        self.end = 0
        self._reserve(16 + len(body))
        self.buf[0:8] = b"KLN0\x00\x00\x00\x00"
        struct.pack_into("<Q", self.buf, 8, type_hash(schema))
        self.buf[16 : 16 + len(body)] = body
        self.end = 16 + len(body)
        return memoryview(self.buf)[: self.end]

    def _write(self, schema: tuple, value: Any, cap: int, depth: int) -> None:
        kind = schema[0]
        if depth >= 64 and kind in ("list", "opt", "dict", "struct", "tensors"):
            raise KilnError("LIMIT")
        if kind == "bool":
            self._reserve(1)
            self.buf[self.end] = 1 if value else 0
            self.end += 1
            return
        if kind in _PACK:
            width = _WIDTH[kind]
            self._reserve(width)
            _STRUCT[kind].pack_into(self.buf, self.end, value)
            self.end += width
            return
        if kind == "str":
            raw = value.encode("utf-8")
            self._bytes(raw, cap)
            return
        if kind == "bytes":
            self._bytes(bytes(value), cap)
            return
        if kind == "tensors":
            raw = encode_tensors(value if isinstance(value, list) else value["tensors"])
            self._reserve(len(raw))
            self.buf[self.end : self.end + len(raw)] = raw
            self.end += len(raw)
            return
        if kind == "opt":
            self._reserve(1)
            if value is None:
                self.buf[self.end] = 0
                self.end += 1
                return
            self.buf[self.end] = 1
            self.end += 1
            self._write(schema[1], value, cap, depth + 1)
            return
        if kind == "list":
            elem = schema[1]
            count = len(value)
            _fit(count, _elem_width(elem), 2**64 - 1, cap)
            self._u32(count)
            if elem[0] in ("u8", "i8"):
                if elem[0] == "u8":
                    raw = bytes(value)
                else:
                    raw = struct.pack("<" + "b" * count, *value) if count else b""
                if len(raw) > cap:
                    raise KilnError("LIMIT")
                self._reserve(len(raw))
                self.buf[self.end : self.end + len(raw)] = raw
                self.end += len(raw)
                return
            if elem[0] in _PACK and elem[0] not in ("i8",):
                if count:
                    packed = _list_struct(elem[0], count)
                    self._reserve(packed.size)
                    packed.pack_into(self.buf, self.end, *value)
                    self.end += packed.size
                return
            for item in value:
                self._write(elem, item, cap, depth + 1)
            return
        if kind == "dict":
            items = sorted(value.items(), key=lambda pair: pair[0].encode("utf-8"))
            _fit(len(items), 4, 2**64 - 1, cap)
            self._u32(len(items))
            for key, item in items:
                self._write(("str",), key, cap, depth + 1)
                self._write(schema[1], item, cap, depth + 1)
            return
        if kind == "struct":
            for name, field in schema[1]:
                fk = field[0]
                item = value[name]
                if fk in _STRUCT:
                    width = _WIDTH[fk]
                    self._reserve(width)
                    _STRUCT[fk].pack_into(self.buf, self.end, item)
                    self.end += width
                elif fk == "bool":
                    self._reserve(1)
                    self.buf[self.end] = 1 if item else 0
                    self.end += 1
                elif fk == "str":
                    self._bytes(item.encode("utf-8"), cap)
                elif fk == "bytes":
                    self._bytes(bytes(item), cap)
                else:
                    self._write(field, item, cap, depth + 1)
            return
        raise KilnError("TYPE")

    def _u32(self, value: int) -> None:
        self._reserve(4)
        struct.pack_into("<I", self.buf, self.end, value)
        self.end += 4

    def _bytes(self, raw: bytes, cap: int) -> None:
        if len(raw) > cap:
            raise KilnError("LIMIT")
        self._u32(len(raw))
        self._reserve(len(raw))
        self.buf[self.end : self.end + len(raw)] = raw
        self.end += len(raw)


def _elem_width(schema: tuple) -> int:
    if schema[0] in ("u8", "i8"):
        return 1
    return min_size(schema)


_writers = threading.local()


def _writer() -> Writer:
    writer = getattr(_writers, "writer", None)
    if writer is None:
        writer = Writer(256)
        _writers.writer = writer
    return writer


class Reader:
    """Unchecked reads for a buffer this process already bounded.

    `str_fast` does not apply the count cap. Pure Python still rejects
    invalid UTF-8 when it builds a `str`. A length past the buffer raises
    `KilnError("EOF")` instead of reading off the end of the view.
    """

    def __init__(self, buf: bytes | memoryview) -> None:
        self.buf = buf if isinstance(buf, memoryview) else memoryview(buf)
        self.i = 0

    def u8_fast(self) -> int:
        value = self.buf[self.i]
        self.i += 1
        return value

    def u32_fast(self) -> int:
        value = _STRUCT["u32"].unpack_from(self.buf, self.i)[0]
        self.i += 4
        return value

    def u64_fast(self) -> int:
        value = _STRUCT["u64"].unpack_from(self.buf, self.i)[0]
        self.i += 8
        return value

    def str_fast(self) -> str:
        length = self.u32_fast()
        raw = self.buf[self.i : self.i + length]
        if len(raw) != length:
            raise KilnError("EOF")
        self.i += length
        return str(raw, "utf-8")


def encode(schema: tuple, value: Any, *, cap: int = CAP) -> bytes:
    return bytes(_writer().encode(schema, value, cap=cap))


def encode_framed(schema: tuple, value: Any, *, cap: int = CAP) -> bytes:
    return bytes(_writer().encode_framed(schema, value, cap=cap))


def decode(schema: tuple, buf: bytes | memoryview, *, cap: int = CAP, standalone: bool = True) -> Any:
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    value, end = _read(schema, view, 0, cap, 0, standalone=False)
    if standalone and end != len(view):
        raise KilnError("TRAILING")
    return value


def decode_framed(schema: tuple, buf: bytes | memoryview, *, cap: int = CAP) -> Any:
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    if len(view) < 16:
        raise KilnError("EOF")
    if bytes(view[:4]) != b"KLN0" or any(view[4:8]):
        raise KilnError("FRAME")
    got = struct.unpack_from("<Q", view, 8)[0]
    if got != type_hash(schema):
        raise KilnError("SCHEMA")
    value, end = _read(schema, view, 16, cap, 0, standalone=False)
    if end != len(view):
        raise KilnError("TRAILING")
    return value


def _read(schema: tuple, buf: memoryview, i: int, cap: int, depth: int, *, standalone: bool) -> tuple[Any, int]:
    kind = schema[0]
    if depth >= 64 and kind in ("list", "opt", "dict", "struct", "tensors"):
        raise KilnError("LIMIT")
    if kind == "bool":
        if i >= len(buf):
            raise KilnError("EOF")
        byte = buf[i]
        if byte > 1:
            raise KilnError("TYPE")
        return bool(byte), i + 1
    if kind in _PACK:
        width = _WIDTH[kind]
        if i + width > len(buf):
            raise KilnError("EOF")
        return _STRUCT[kind].unpack_from(buf, i)[0], i + width
    if kind in ("str", "bytes"):
        if i + 4 > len(buf):
            raise KilnError("EOF")
        length = struct.unpack_from("<I", buf, i)[0]
        if length > cap:
            raise KilnError("LIMIT")
        start = i + 4
        end = start + length
        if end > len(buf):
            raise KilnError("EOF")
        raw = bytes(buf[start:end])
        if kind == "bytes":
            return raw, end
        try:
            return raw.decode("utf-8"), end
        except UnicodeDecodeError as exc:
            raise KilnError("UTF8") from exc
    if kind == "tensors":
        directory, block = decode_tensors(buf[i:], standalone=False, cap=cap)
        if standalone and i + block != len(buf):
            raise KilnError("TRAILING")
        return directory, i + block
    if kind == "opt":
        if i >= len(buf):
            raise KilnError("EOF")
        tag = buf[i]
        if tag == 0:
            return None, i + 1
        if tag != 1:
            raise KilnError("TYPE")
        return _read(schema[1], buf, i + 1, cap, depth + 1, standalone=False)
    if kind == "list":
        if i + 4 > len(buf):
            raise KilnError("EOF")
        count = struct.unpack_from("<I", buf, i)[0]
        elem = schema[1]
        _fit(count, _elem_width(elem), len(buf) - (i + 4), cap)
        start = i + 4
        if elem[0] in ("u8", "i8") or elem[0] in _PACK:
            width = _WIDTH[elem[0]]
            end = start + count * width
            raw = buf[start:end]
            if elem[0] == "u8":
                return raw.tobytes(), end
            code = _ARRAY.get(elem[0])
            if code is not None and count:
                values = array.array(code)
                values.frombytes(raw)
                return values.tolist(), end
            fmt = "<" + _PACK[elem[0]][1] * count
            return list(struct.unpack_from(fmt, raw)) if count else [], end
        out = []
        cursor = start
        for _ in range(count):
            item, cursor = _read(elem, buf, cursor, cap, depth + 1, standalone=False)
            out.append(item)
        return out, cursor
    if kind == "dict":
        if i + 4 > len(buf):
            raise KilnError("EOF")
        count = struct.unpack_from("<I", buf, i)[0]
        _fit(count, 4, len(buf) - (i + 4), cap)
        cursor = i + 4
        out: dict[str, Any] = {}
        prev = b""
        for _ in range(count):
            key, cursor = _read(("str",), buf, cursor, cap, depth + 1, standalone=False)
            raw_key = key.encode("utf-8")
            if raw_key < prev:
                raise KilnError("TYPE")
            prev = raw_key
            item, cursor = _read(schema[1], buf, cursor, cap, depth + 1, standalone=False)
            out[key] = item
        return out, cursor
    if kind == "struct":
        out = {}
        cursor = i
        for name, field in schema[1]:
            fk = field[0]
            if fk in _STRUCT:
                width = _WIDTH[fk]
                if cursor + width > len(buf):
                    raise KilnError("EOF")
                out[name] = _STRUCT[fk].unpack_from(buf, cursor)[0]
                cursor += width
            elif fk == "bool":
                if cursor >= len(buf):
                    raise KilnError("EOF")
                byte = buf[cursor]
                if byte > 1:
                    raise KilnError("TYPE")
                out[name] = bool(byte)
                cursor += 1
            else:
                out[name], cursor = _read(field, buf, cursor, cap, depth + 1, standalone=False)
        return out, cursor
    raise KilnError("TYPE")
