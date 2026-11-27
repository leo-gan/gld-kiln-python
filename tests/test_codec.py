"""Tests for compiled pure-Python Kiln codecs."""

from __future__ import annotations

import array
import math
import struct
import pytest

from kiln import Codec, KilnError, Writer, compile as compile_codec, decode, encode


DOCUMENT = (
    "struct",
    [
        ("id", ("str",)),
        ("status", ("i32",)),
        ("meta", ("struct", [("region", ("str",)), ("version", ("i32",))])),
        (
            "items",
            ("list", ("struct", [("sku", ("str",)), ("qty", ("i32",)), ("price_minor", ("i64",))])),
        ),
    ],
)

TELEMETRY = (
    "struct",
    [
        ("source", ("str",)),
        ("ts", ("i64",)),
        ("tags", ("list", ("str",))),
        ("values", ("list", ("f64",))),
    ],
)


def test_compiled_codec_matches_writer_wire_bytes():
    doc = {
        "id": "doc-001",
        "status": 200,
        "meta": {"region": "us-east-1", "version": 3},
        "items": [
            {"sku": "sku-a", "qty": 4, "price_minor": 1999},
            {"sku": "café-🚀", "qty": -2, "price_minor": -999999999999},
        ],
    }
    codec = compile_codec(DOCUMENT)
    assert isinstance(codec, Codec)
    raw = codec.encode(doc)
    ref = bytes(Writer().encode(DOCUMENT, doc))
    assert raw == ref
    assert codec.decode(raw) == doc
    assert codec.decode(bytearray(raw)) == doc
    assert codec.decode(memoryview(raw)) == doc


def test_compiled_codec_pod_arrays_and_float_bits():
    tel = {
        "source": "sensor-1",
        "ts": 1700000000,
        "tags": ["", "a", "us-west-2", "long-tag-over-sixteen-chars"],
        "values": array.array("d", [-0.0, 0.0, 1.5, -1e100]),
    }
    codec_arr = compile_codec(TELEMETRY, pod_arrays=True)
    codec_list = compile_codec(TELEMETRY, pod_arrays=False)

    raw = codec_arr.encode(tel)
    assert raw == encode(TELEMETRY, {**tel, "values": list(tel["values"])})

    got_arr = codec_arr.decode(raw)
    assert isinstance(got_arr["values"], array.array)
    assert got_arr["values"].typecode == "d"
    assert math.copysign(1.0, got_arr["values"][0]) < 0
    assert list(got_arr["values"]) == list(tel["values"])

    got_list = codec_list.decode(raw)
    assert isinstance(got_list["values"], list)
    assert got_list["values"] == list(tel["values"])


def test_compiled_codec_rejects_errors():
    str_codec = compile_codec(("str",))
    with pytest.raises(KilnError) as caught:
        str_codec.decode(struct.pack("<I", 2) + b"\xff\xfe")
    assert caught.value.code == "UTF8"

    with pytest.raises(KilnError) as caught:
        str_codec.decode(struct.pack("<I", 1) + b"a\x00")
    assert caught.value.code == "TRAILING"

    bool_codec = compile_codec(("struct", [("b", ("bool",)), ("x", ("i32",))]))
    with pytest.raises(KilnError) as caught:
        bool_codec.decode(b"\x02\x00\x00\x00\x00")
    assert caught.value.code == "TYPE"

    list_codec = compile_codec(("list", ("i64",)))
    with pytest.raises(KilnError) as caught:
        list_codec.decode(bytes.fromhex("00000001"))
    assert caught.value.code == "EOF"
    with pytest.raises(KilnError) as caught:
        list_codec.decode(bytes.fromhex("01000001"))
    assert caught.value.code == "LIMIT"

    opt_codec = compile_codec(("opt", ("str",)))
    assert opt_codec.decode(opt_codec.encode(None)) is None
    assert opt_codec.decode(opt_codec.encode("ok")) == "ok"
    with pytest.raises(KilnError) as caught:
        opt_codec.decode(b"\x02")
    assert caught.value.code == "TYPE"
