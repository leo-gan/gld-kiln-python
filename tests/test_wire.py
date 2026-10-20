"""The 46 Kiln wire v2 cases, plus Frozen bytes from the reference codec."""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "testdata" / "reference"))

from kiln import KilnError, decode, decode_framed, decode_tensors, encode, encode_framed, encode_tensors, type_hash
from kiln.schema import schema_from_case
import kiln_ref

CASES = json.loads((ROOT / "testdata" / "kiln-v2-cases.json").read_text())["cases"]


def _project(value):
    if isinstance(value, dict) and "tensors" in value and "block_length" in value:
        return {
            "version": value["version"],
            "flags": value["flags"],
            "tensors": [
                {
                    "name": item["name"],
                    "dtype": item["dtype"],
                    "shape": item["shape"],
                    "quant_scheme": item["quant_scheme"],
                    "data_hex": bytes(item["data"]).hex(),
                }
                for item in value["tensors"]
            ],
            "block_length": value["block_length"],
        }
    if isinstance(value, dict):
        return {key: _project(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_project(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return value
        return value
    return value


def _same(got, want) -> bool:
    if isinstance(want, float) and isinstance(got, float):
        if math.isnan(want) and math.isnan(got):
            return True
        return got == want
    if isinstance(want, dict) and isinstance(got, dict):
        return set(want) <= set(got) and all(_same(got[key], item) for key, item in want.items())
    if isinstance(want, list) and isinstance(got, list):
        return len(want) == len(got) and all(_same(a, b) for a, b in zip(got, want))
    return got == want


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_v2_case(case):
    raw = bytes.fromhex(case["input"])
    schema = schema_from_case(case["schema"])
    framed = case["schema"] == "frame"
    if case["expect"] == "reject":
        with pytest.raises(KilnError) as caught:
            if framed:
                decode_framed(schema, raw)
            else:
                decode(schema, raw)
        assert caught.value.code == case["error"]
        return
    value = decode_framed(schema, raw) if framed else decode(schema, raw)
    if "decoded" in case:
        want = case["decoded"]
        if framed:
            assert want["hash"] == type_hash(schema)
            want = {"id": want["id"], "batch": want["batch"]}
        assert _same(_project(value), want)
    again = encode_framed(schema, value) if framed else encode(schema, value)
    assert again == raw


def test_reference_point_and_message():
    point = kiln_ref.POINT
    message = kiln_ref.MESSAGE
    point_value = {"x": 9, "y": -4}
    message_value = {
        "f_bool": True,
        "f_int32": -2,
        "f_int64": 99,
        "f_float64": 1.5,
        "f_string": "hello",
        "f_bool_2": False,
        "f_int32_2": 7,
        "f_string_2": "world",
    }
    for schema, value in ((point, point_value), (message, message_value)):
        raw = kiln_ref.encode(schema, value)
        assert decode(schema, raw) == value
        assert encode(schema, value) == raw
        framed = kiln_ref.encode_framed(schema, value)
        assert decode_framed(schema, framed) == value
        assert encode_framed(schema, value) == framed


def test_negative_zero_and_nan_payload():
    schema = ("list", ("f64",))
    payload = struct.pack("<dd", -0.0, struct.unpack("<d", bytes.fromhex("010000000000f87f"))[0])
    raw = struct.pack("<I", 2) + payload
    got = decode(schema, raw)
    assert math.copysign(1.0, got[0]) < 0
    assert math.isnan(got[1])
    assert struct.pack("<d", got[1]) == bytes.fromhex("010000000000f87f")
    assert encode(schema, got) == raw


def test_dict_key_order_and_opt():
    schema = ("dict", ("i32",))
    raw = encode(schema, {"b": 2, "a": 1})
    assert decode(schema, raw) == {"a": 1, "b": 2}
    opt = ("opt", ("str",))
    assert decode(opt, encode(opt, None)) is None
    assert decode(opt, encode(opt, "z")) == "z"


def test_frame_hash_mismatch():
    schema = ("struct", [("id", ("i64",))])
    raw = encode_framed(schema, {"id": 1})
    other = ("struct", [("id", ("i32",))])
    with pytest.raises(KilnError) as caught:
        decode_framed(other, raw)
    assert caught.value.code == "SCHEMA"


def test_nonzero_alignment_gap_is_type():
    raw = bytearray(encode_tensors([
        {"dtype": "f8e4m3", "shape": [1], "data": bytes([0x38])},
        {"dtype": "f8e5m2", "shape": [1], "data": bytes([0x3C])},
    ]))
    # The second payload starts at offset 64. The byte before it is padding.
    raw[320 + 63] = 1
    with pytest.raises(KilnError) as caught:
        decode_tensors(raw)
    assert caught.value.code == "TYPE"


def test_reader_skips_the_cap():
    from kiln import Reader

    raw = encode(("str",), "z" * 20, cap=100)
    assert Reader(raw).str_fast() == "z" * 20
    with pytest.raises(KilnError) as caught:
        decode(("str",), raw, cap=10)
    assert caught.value.code == "LIMIT"


def test_depth_limit():
    schema = ("i32",)
    value = 1
    for _ in range(64):
        schema = ("list", schema)
        value = [value]
    encode(schema, value)
    schema = ("list", schema)
    value = [value]
    with pytest.raises(KilnError) as caught:
        encode(schema, value)
    assert caught.value.code == "LIMIT"


def test_signed_byte_list():
    schema = ("list", ("i8",))
    value = [0, -1, 127, -128]
    assert decode(schema, encode(schema, value)) == value


def test_hostile_list_does_not_allocate():
    schema = schema_from_case({"type": "list", "element": "struct{a:i64}"})
    with pytest.raises(KilnError) as caught:
        decode(schema, bytes.fromhex("00000001"))
    assert caught.value.code == "EOF"
