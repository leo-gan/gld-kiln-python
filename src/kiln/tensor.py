"""Version-2 tensor directory. Payloads are slices of the input buffer."""

from __future__ import annotations

import struct
import threading
from typing import Any

from kiln.error import KilnError

HEADER = 64
DESC = 128
CAP = 16_777_216
MASK = 0xFFFFFFFFFFFFFFFF
FLAG_NAME = 0x01
FLAG_LE = 0x02

DTYPE = {
    "f64": 1,
    "f32": 2,
    "f16": 3,
    "bf16": 4,
    "f8e4m3": 5,
    "f8e5m2": 6,
    "bool": 7,
    "i8": 8,
    "u8": 9,
    "i16": 10,
    "u16": 11,
    "i32": 12,
    "u32": 13,
    "i64": 14,
    "u64": 15,
    "i4": 16,
    "u4": 17,
    "e2m1": 18,
    "e8m0": 19,
}
DTYPE_NAME = {code: name for name, code in DTYPE.items()}
WHOLE = {1: 8, 2: 4, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 12: 4, 13: 4, 14: 8, 15: 8, 19: 1}
PACKED = {16, 17, 18}
OPAQUE_MIN = 4096
OPAQUE_MAX = 32767
_ZERO16 = bytes(16)
_ZERO32 = bytes(32)
_HDR = struct.Struct("<4sBBHIIQQ32s")
_DESC = struct.Struct("<HBBHHHHIHHIIIQQ8Q16s")
_scratch = threading.local()


def _scratch_buf(need: int) -> bytearray:
    block = getattr(_scratch, "buf", None)
    if block is None or len(block) < need:
        block = bytearray(need)
        _scratch.buf = block
    return block


def align64(n: int) -> int:
    return (n + 63) & ~63


def _add(a: int, b: int) -> int:
    total = a + b
    if total > MASK:
        raise KilnError("LIMIT")
    return total


def _mul(a: int, b: int) -> int:
    if a != 0 and b > MASK // a:
        raise KilnError("LIMIT")
    return a * b


def _u16(buf: bytes | memoryview, i: int) -> int:
    return struct.unpack_from("<H", buf, i)[0]


def _u32(buf: bytes | memoryview, i: int) -> int:
    return struct.unpack_from("<I", buf, i)[0]


def _u64(buf: bytes | memoryview, i: int) -> int:
    return struct.unpack_from("<Q", buf, i)[0]


def element_count(rank: int, shape: list[int]) -> int:
    if rank == 0:
        return 1
    count = 1
    for dim in range(rank):
        count = _mul(count, shape[dim])
    return count


def expected_bytes(dtype: int, count: int) -> int:
    if dtype in WHOLE:
        return _mul(count, WHOLE[dtype])
    if dtype in PACKED:
        return (count + 1) // 2
    raise KilnError("TYPE")


def _stamp(tensors: list[dict[str, Any]]) -> None:
    heap = 0
    for tensor in tensors:
        name = tensor.get("name") or b""
        if isinstance(name, str):
            name = name.encode("utf-8")
            tensor["name"] = name
        if name:
            tensor["name_off"] = heap
            tensor["name_len"] = len(name)
            heap += len(name)
        else:
            tensor["name"] = b""
            tensor["name_off"] = 0xFFFF
            tensor["name_len"] = 0
    for tensor in tensors:
        layout = tensor.get("layout") or b""
        if isinstance(layout, str):
            layout = layout.encode("utf-8")
            tensor["layout"] = layout
        if layout:
            tensor["layout_off"] = heap
            tensor["layout_len"] = len(layout)
            heap += len(layout)
        else:
            tensor["layout"] = b""
            tensor["layout_off"] = 0xFFFF
            tensor["layout_len"] = 0


def _encode_plain(tensor: dict[str, Any]) -> bytes | None:
    """One unnamed tensor with no layout and no quant metadata."""
    if (
        tensor.get("name")
        or tensor.get("layout")
        or tensor.get("quant_scheme")
        or tensor.get("block_size")
        or tensor.get("group_size")
        or tensor.get("quant_axis") not in (None, 0xFFFF)
        or tensor.get("scale_index") not in (None, 0xFFFFFFFF)
        or tensor.get("outer_index") not in (None, 0xFFFFFFFF)
    ):
        return None
    dtype = tensor["dtype"]
    code = dtype if isinstance(dtype, int) else DTYPE.get(dtype)
    if code is None or code >= OPAQUE_MIN:
        return None
    shape = tensor["shape"]
    rank = len(shape)
    if rank > 8:
        return None
    data = tensor["data"]
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    slots = (0, 0, 0, 0, 0, 0, 0, 0)
    if rank:
        slots = tuple(shape) + slots[rank:]
    length = len(data)
    need = 192 + length
    block = _scratch_buf(need)
    _HDR.pack_into(block, 0, b"KLNT", 2, FLAG_LE, 0, 1, 0, length, DESC, _ZERO32)
    _DESC.pack_into(
        block, HEADER,
        code, rank, 0, 0xFFFF, 0, 0xFFFF, 0, 0, 0, 0xFFFF, 0,
        0xFFFFFFFF, 0xFFFFFFFF, 0, length,
        slots[0], slots[1], slots[2], slots[3], slots[4], slots[5], slots[6], slots[7],
        _ZERO16,
    )
    block[192 : 192 + length] = data
    return bytes(block[:need])


def _encode_plains(tensors: list[dict[str, Any]]) -> bytes | None:
    """Unnamed tensors with no layout and no quant metadata."""
    prepared = []
    end = 0
    for tensor in tensors:
        if (
            tensor.get("name")
            or tensor.get("layout")
            or tensor.get("quant_scheme")
            or tensor.get("block_size")
            or tensor.get("group_size")
            or tensor.get("quant_axis") not in (None, 0xFFFF)
            or tensor.get("scale_index") not in (None, 0xFFFFFFFF)
            or tensor.get("outer_index") not in (None, 0xFFFFFFFF)
        ):
            return None
        dtype = tensor["dtype"]
        code = dtype if isinstance(dtype, int) else DTYPE.get(dtype)
        if code is None or code == 0 or code >= OPAQUE_MIN:
            return None
        shape = tensor["shape"]
        rank = len(shape)
        if rank > 8:
            return None
        data = tensor["data"]
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        offset = align64(end)
        end = offset + len(data)
        prepared.append((code, rank, shape, data, offset))
    count = len(prepared)
    directory_bytes = count * DESC
    payload_base = HEADER + directory_bytes
    need = payload_base + end
    block = _scratch_buf(need)
    _HDR.pack_into(block, 0, b"KLNT", 2, FLAG_LE, 0, count, 0, end, directory_bytes, _ZERO32)
    for index, (code, rank, shape, data, offset) in enumerate(prepared):
        full = [0, 0, 0, 0, 0, 0, 0, 0]
        for dim in range(rank):
            full[dim] = shape[dim]
        _DESC.pack_into(
            block, HEADER + index * DESC,
            code, rank, 0, 0xFFFF, 0, 0xFFFF, 0, 0, 0, 0xFFFF, 0,
            0xFFFFFFFF, 0xFFFFFFFF, offset, len(data),
            full[0], full[1], full[2], full[3], full[4], full[5], full[6], full[7],
            _ZERO16,
        )
        if offset > (0 if index == 0 else prepared[index - 1][4] + len(prepared[index - 1][3])):
            gap_from = 0 if index == 0 else prepared[index - 1][4] + len(prepared[index - 1][3])
            block[payload_base + gap_from : payload_base + offset] = b"\x00" * (offset - gap_from)
        dest = payload_base + offset
        block[dest : dest + len(data)] = data
    return bytes(memoryview(block)[:need])


def encode_tensors(tensors: list[dict[str, Any]]) -> bytes:
    """Write one canonical directory. `tensors` is in caller order."""
    if len(tensors) == 1:
        one = _encode_plain(tensors[0])
        if one is not None:
            return one
    elif len(tensors) != 1:
        plain = _encode_plains(tensors)
        if plain is not None:
            return plain
    prepared: list[dict[str, Any]] = []
    for tensor in tensors:
        dtype = tensor["dtype"]
        code = dtype if isinstance(dtype, int) else DTYPE[dtype]
        shape = tensor["shape"]
        rank = len(shape)
        data = tensor["data"]
        if not isinstance(data, (bytes, bytearray, memoryview)):
            data = bytes(data)
        full = [0, 0, 0, 0, 0, 0, 0, 0]
        for dim in range(rank):
            full[dim] = shape[dim]
        axis = tensor.get("quant_axis")
        scale = tensor.get("scale_index")
        outer = tensor.get("outer_index")
        prepared.append(
            {
                "dtype": code,
                "rank": rank,
                "shape": full,
                "data": data,
                "name": tensor.get("name") or b"",
                "layout": tensor.get("layout") or b"",
                "block_size": int(tensor.get("block_size") or 0),
                "quant_scheme": int(tensor.get("quant_scheme") or 0),
                "quant_axis": 0xFFFF if axis is None else int(axis),
                "group_size": int(tensor.get("group_size") or 0),
                "scale_index": 0xFFFFFFFF if scale is None else int(scale),
                "outer_index": 0xFFFFFFFF if outer is None else int(outer),
                "byte_length": len(data),
            }
        )
    _stamp(prepared)
    end = 0
    heap_len = 0
    named = False
    for tensor in prepared:
        tensor["byte_offset"] = align64(end)
        end = tensor["byte_offset"] + tensor["byte_length"]
        heap_len += tensor["name_len"] + tensor["layout_len"]
        if tensor["name_len"]:
            named = True
    count = len(prepared)
    directory_bytes = count * DESC + heap_len
    payload_base = align64(HEADER + directory_bytes)
    need = payload_base + end
    block = _scratch_buf(need)
    pad = payload_base - (HEADER + directory_bytes)
    if pad:
        block[HEADER + directory_bytes : payload_base] = b"\x00" * pad
    _HDR.pack_into(
        block, 0, b"KLNT", 2, FLAG_LE | (FLAG_NAME if named else 0), 0,
        count, heap_len, end, directory_bytes, _ZERO32,
    )
    for index, tensor in enumerate(prepared):
        shape = tensor["shape"]
        _DESC.pack_into(
            block, HEADER + index * DESC,
            tensor["dtype"], tensor["rank"], 0,
            tensor["name_off"], tensor["name_len"], tensor["layout_off"], tensor["layout_len"],
            tensor["block_size"], tensor["quant_scheme"], tensor["quant_axis"], tensor["group_size"],
            tensor["scale_index"], tensor["outer_index"], tensor["byte_offset"], tensor["byte_length"],
            shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7],
            _ZERO16,
        )
    cursor = HEADER + count * DESC
    for tensor in prepared:
        name = tensor["name"]
        if name:
            nxt = cursor + len(name)
            block[cursor:nxt] = name
            cursor = nxt
    for tensor in prepared:
        layout = tensor["layout"]
        if layout:
            nxt = cursor + len(layout)
            block[cursor:nxt] = layout
            cursor = nxt
    cursor = 0
    for tensor in prepared:
        start = tensor["byte_offset"]
        if start > cursor:
            block[payload_base + cursor : payload_base + start] = b"\x00" * (start - cursor)
        off = payload_base + start
        block[off : off + tensor["byte_length"]] = tensor["data"]
        cursor = start + tensor["byte_length"]
    return bytes(block[:need])


def _check_tensor(item: dict[str, Any]) -> None:
    dtype = item["dtype"]
    rank = item["rank"]
    if rank > 8 or any(item["shape"][rank:]):
        raise KilnError("TYPE")
    if dtype == 0 or (20 <= dtype < OPAQUE_MIN) or dtype > OPAQUE_MAX:
        raise KilnError("TYPE")
    count = element_count(rank, item["shape"])
    if dtype >= OPAQUE_MIN:
        if not item["layout"] or item["block_size"] == 0:
            raise KilnError("TYPE")
        if item["byte_length"] % item["block_size"] != 0:
            raise KilnError("TYPE")
        return
    if item["layout"] or item["block_size"] != 0:
        raise KilnError("TYPE")
    if item["byte_length"] != expected_bytes(dtype, count):
        raise KilnError("TYPE")


def _check_quant(tensors: list[dict[str, Any]]) -> None:
    for index, item in enumerate(tensors):
        scheme = item["quant_scheme"]
        if scheme == 0:
            if (
                item["group_size"] != 0
                or item["quant_axis"] != 0xFFFF
                or item["scale_index"] != 0xFFFFFFFF
                or item["outer_index"] != 0xFFFFFFFF
            ):
                raise KilnError("TYPE")
            continue
        if scheme not in (1, 2):
            raise KilnError("TYPE")
        if item["rank"] == 0 or item["quant_axis"] >= item["rank"] or item["group_size"] == 0:
            raise KilnError("TYPE")
        scale = item["scale_index"]
        if scale == index or scale >= len(tensors):
            raise KilnError("TYPE")
        other = tensors[scale]
        if other["quant_scheme"] != 0:
            raise KilnError("TYPE")
        groups = (item["shape"][item["quant_axis"]] + item["group_size"] - 1) // item["group_size"]
        other_dims = 1
        for dim in range(item["rank"]):
            if dim != item["quant_axis"]:
                other_dims = _mul(other_dims, item["shape"][dim])
        if element_count(other["rank"], other["shape"]) != _mul(groups, other_dims):
            raise KilnError("TYPE")
        if scheme == 1:
            if item["outer_index"] != 0xFFFFFFFF:
                raise KilnError("TYPE")
            continue
        outer = item["outer_index"]
        if outer == index or outer == scale or outer >= len(tensors):
            raise KilnError("TYPE")
        outer_item = tensors[outer]
        if (
            outer_item["quant_scheme"] != 0
            or outer_item["dtype"] != DTYPE["f32"]
            or element_count(outer_item["rank"], outer_item["shape"]) != 1
        ):
            raise KilnError("TYPE")


def _finish_single(view: memoryview, block_end: int, payload_bytes: int, standalone: bool) -> tuple[dict[str, Any], int]:
    (
        dtype, rank, reserved, name_off, name_len, layout_off, layout_len, block_size,
        quant_scheme, quant_axis, group_size, scale_index, outer_index,
        byte_offset, byte_length, s0, s1, s2, s3, s4, s5, s6, s7, reserved_desc,
    ) = _DESC.unpack_from(view, HEADER)
    if reserved_desc != _ZERO16 or reserved != 0:
        raise KilnError("FRAME")
    if (
        name_len
        or name_off != 0xFFFF
        or layout_len
        or layout_off != 0xFFFF
        or quant_scheme
        or group_size
        or block_size
        or quant_axis != 0xFFFF
        or scale_index != 0xFFFFFFFF
        or outer_index != 0xFFFFFFFF
        or byte_offset
        or byte_length != payload_bytes
        or rank > 8
    ):
        raise KilnError("TYPE")
    shape = (s0, s1, s2, s3, s4, s5, s6, s7)
    width = WHOLE.get(dtype)
    if width is None:
        out, off, length = _one_descriptor(view, HEADER, view[192 : 192 + payload_bytes])
        if off or length != payload_bytes:
            raise KilnError("TYPE")
    else:
        for dim in range(rank, 8):
            if shape[dim]:
                raise KilnError("TYPE")
        count = 1
        for dim in range(rank):
            count = _mul(count, shape[dim])
        if byte_length != _mul(count, width):
            raise KilnError("TYPE")
        region = view[192 : 192 + byte_length]
        if dtype == 7 and any(byte > 1 for byte in region):
            raise KilnError("TYPE")
        out = {
            "name": None,
            "dtype": DTYPE_NAME[dtype],
            "shape": list(shape[:rank]),
            "data": region,
            "quant_scheme": 0,
            "quant_axis": 0xFFFF,
            "group_size": 0,
            "scale_index": 0xFFFFFFFF,
            "outer_index": 0xFFFFFFFF,
            "layout": None,
            "block_size": 0,
        }
    if standalone and block_end != len(view):
        raise KilnError("TRAILING")
    return {"version": 2, "flags": ["le"], "tensors": [out], "block_length": block_end}, block_end


def _finish_unnamed(
    view: memoryview,
    count: int,
    payload_base: int,
    block_end: int,
    payload_bytes: int,
    standalone: bool,
) -> tuple[dict[str, Any], int]:
    payload = view[payload_base : payload_base + payload_bytes]
    tensors = []
    cursor = 0
    for index in range(count):
        out, byte_offset, byte_length = _one_descriptor(view, HEADER + index * DESC, payload)
        if byte_offset != align64(cursor):
            raise KilnError("TYPE")
        gap = payload[cursor:byte_offset]
        if gap and any(gap):
            raise KilnError("TYPE")
        cursor = byte_offset + byte_length
        tensors.append(out)
    if cursor != payload_bytes:
        raise KilnError("TYPE")
    if standalone and block_end != len(view):
        raise KilnError("TRAILING")
    return {
        "version": 2,
        "flags": ["le"],
        "tensors": tensors,
        "block_length": block_end,
    }, block_end


def _one_descriptor(view: memoryview, base: int, payload: memoryview) -> tuple[dict[str, Any], int, int]:
    (
        dtype, rank, reserved, name_off, name_len, layout_off, layout_len, block_size,
        quant_scheme, quant_axis, group_size, scale_index, outer_index,
        byte_offset, byte_length, *shape, reserved_desc,
    ) = _DESC.unpack_from(view, base)
    if reserved_desc != _ZERO16 or reserved != 0:
        raise KilnError("FRAME")
    if name_len != 0 or name_off != 0xFFFF or layout_len != 0 or layout_off != 0xFFFF:
        raise KilnError("TYPE")
    if quant_scheme != 0 or group_size != 0 or quant_axis != 0xFFFF or scale_index != 0xFFFFFFFF or outer_index != 0xFFFFFFFF or block_size != 0:
        raise KilnError("TYPE")
    if _add(byte_offset, byte_length) > len(payload):
        raise KilnError("TYPE")
    width = WHOLE.get(dtype)
    if width is None or rank > 8:
        item = {
            "dtype": dtype,
            "rank": rank,
            "layout": b"",
            "block_size": 0,
            "quant_scheme": 0,
            "quant_axis": 0xFFFF,
            "group_size": 0,
            "scale_index": 0xFFFFFFFF,
            "outer_index": 0xFFFFFFFF,
            "byte_length": byte_length,
            "shape": list(shape),
        }
        _check_tensor(item)
        shape_out = item["shape"][:rank]
    else:
        for dim in range(rank, 8):
            if shape[dim]:
                raise KilnError("TYPE")
        count = 1
        for dim in range(rank):
            count = _mul(count, shape[dim])
        if byte_length != _mul(count, width):
            raise KilnError("TYPE")
        shape_out = shape[:rank]
    region = payload[byte_offset : byte_offset + byte_length]
    if dtype in PACKED:
        count_elems = 1 if rank == 0 else shape[0]
        for dim in range(1, rank):
            count_elems = _mul(count_elems, shape[dim])
        if count_elems % 2 == 1 and region and (region[-1] >> 4) != 0:
            raise KilnError("TYPE")
    elif dtype == 7 and any(byte > 1 for byte in region):
        raise KilnError("TYPE")
    out = {
        "name": None,
        "dtype": DTYPE_NAME.get(dtype, dtype),
        "shape": shape_out,
        "data": region,
        "quant_scheme": 0,
        "quant_axis": 0xFFFF,
        "group_size": 0,
        "scale_index": 0xFFFFFFFF,
        "outer_index": 0xFFFFFFFF,
        "layout": None,
        "block_size": 0,
    }
    return out, byte_offset, byte_length


def decode_tensors(buf: bytes | memoryview, *, standalone: bool = True, cap: int = CAP) -> tuple[dict[str, Any], int]:
    """Decode one directory. The second value is the block length."""
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    if len(view) < 4:
        raise KilnError("EOF")
    if view[0] != 0x4B or view[1] != 0x4C or view[2] != 0x4E or view[3] != 0x54:
        raise KilnError("FRAME")
    if len(view) < HEADER:
        raise KilnError("EOF")
    magic, version, flags, reserved16, count, name_bytes, payload_bytes, directory_bytes, reserved_tail = _HDR.unpack_from(view, 0)
    if magic != b"KLNT" or version != 2 or (flags & ~0x03) or (flags & FLAG_LE) == 0:
        raise KilnError("FRAME")
    if reserved16 != 0 or reserved_tail != _ZERO32:
        raise KilnError("FRAME")
    if count > cap or name_bytes > cap:
        raise KilnError("LIMIT")
    if count == 1 and name_bytes == 0 and flags == FLAG_LE and directory_bytes == DESC:
        block_end = _add(192, payload_bytes)
        if block_end > len(view):
            raise KilnError("EOF")
        return _finish_single(view, block_end, payload_bytes, standalone)
    if name_bytes == 0 and flags == FLAG_LE and directory_bytes == count * DESC:
        payload_base = HEADER + count * DESC
        block_end = _add(payload_base, payload_bytes)
        if block_end > len(view):
            raise KilnError("EOF")
        plain = True
        if count != 1:
            for index in range(count):
                base = HEADER + index * DESC
                if _u32(view, base + 12) != 0 or _u16(view, base + 16) != 0:
                    plain = False
                    break
        if plain:
            return _finish_unnamed(view, count, payload_base, block_end, payload_bytes, standalone)
    expect_dir = _add(_mul(count, DESC), name_bytes)
    if directory_bytes != expect_dir:
        raise KilnError("TYPE")
    directory_end = _add(HEADER, directory_bytes)
    if directory_end > len(view):
        raise KilnError("EOF")
    payload_base = align64(directory_end)
    block_end = _add(payload_base, payload_bytes)
    if block_end > len(view):
        raise KilnError("EOF")
    if any(view[directory_end:payload_base]):
        raise KilnError("TYPE")
    raw: list[dict[str, Any]] = []
    for index in range(count):
        base = HEADER + index * DESC
        (
            dtype, rank, reserved, name_off, name_len, layout_off, layout_len, block_size,
            quant_scheme, quant_axis, group_size, scale_index, outer_index,
            byte_offset, byte_length, *shape, reserved_desc,
        ) = _DESC.unpack_from(view, base)
        if reserved_desc != _ZERO16:
            raise KilnError("FRAME")
        item = {
            "dtype": dtype,
            "rank": rank,
            "reserved": reserved,
            "name_off": name_off,
            "name_len": name_len,
            "layout_off": layout_off,
            "layout_len": layout_len,
            "block_size": block_size,
            "quant_scheme": quant_scheme,
            "quant_axis": quant_axis,
            "group_size": group_size,
            "scale_index": scale_index,
            "outer_index": outer_index,
            "byte_offset": byte_offset,
            "byte_length": byte_length,
            "shape": list(shape),
        }
        raw.append(item)
    heap = view[HEADER + count * DESC : directory_end]
    names_flag = bool(view[5] & FLAG_NAME)
    has_name = any(item["name_len"] for item in raw)
    if names_flag != has_name or (names_flag and any(item["name_len"] == 0 for item in raw)):
        raise KilnError("TYPE")
    parts: list[bytes] = []
    layouts: list[bytes] = []
    seen: set[bytes] = set()
    for item in raw:
        if item["reserved"] != 0:
            raise KilnError("FRAME")
        if item["name_len"] == 0:
            if item["name_off"] != 0xFFFF:
                raise KilnError("TYPE")
            item["name"] = b""
        else:
            if item["name_off"] == 0xFFFF:
                raise KilnError("TYPE")
            end = _add(item["name_off"], item["name_len"])
            if end > name_bytes:
                raise KilnError("TYPE")
            name = bytes(heap[item["name_off"] : end])
            try:
                name.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise KilnError("UTF8") from exc
            if name in seen:
                raise KilnError("TYPE")
            seen.add(name)
            parts.append(name)
            item["name"] = name
        if item["layout_len"] == 0:
            if item["layout_off"] != 0xFFFF:
                raise KilnError("TYPE")
            item["layout"] = b""
        else:
            end = _add(item["layout_off"], item["layout_len"])
            if end > name_bytes:
                raise KilnError("TYPE")
            layout = bytes(heap[item["layout_off"] : end])
            try:
                layout.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise KilnError("UTF8") from exc
            layouts.append(layout)
            item["layout"] = layout
    if bytes(heap) != b"".join(parts) + b"".join(layouts):
        raise KilnError("TYPE")
    ranges: list[tuple[int, int]] = []
    for item in raw:
        item_end = _add(item["byte_offset"], item["byte_length"])
        if item_end > payload_bytes:
            raise KilnError("TYPE")
        ranges.append((item["byte_offset"], item_end))
        _check_tensor(item)
    if len(ranges) > 1:
        for left in range(len(ranges)):
            a0, a1 = ranges[left]
            for right in range(left + 1, len(ranges)):
                b0, b1 = ranges[right]
                if a0 == a1 or b0 == b1:
                    continue
                if a0 < b1 and b0 < a1:
                    raise KilnError("TYPE")
    end = 0
    for item in raw:
        if item["byte_offset"] != align64(end):
            raise KilnError("TYPE")
        end = item["byte_offset"] + item["byte_length"]
    if end != payload_bytes:
        raise KilnError("TYPE")
    payload = view[payload_base:block_end]
    for item in raw:
        region = payload[item["byte_offset"] : item["byte_offset"] + item["byte_length"]]
        if item["dtype"] in PACKED:
            count_elems = element_count(item["rank"], item["shape"])
            if count_elems % 2 == 1 and region and (region[-1] >> 4) != 0:
                raise KilnError("TYPE")
        if item["dtype"] == DTYPE["bool"] and any(byte > 1 for byte in region):
            raise KilnError("TYPE")
        item["data"] = region
    cursor = 0
    for item in raw:
        gap = payload[cursor : item["byte_offset"]]
        if gap and any(gap):
            raise KilnError("TYPE")
        cursor = item["byte_offset"] + item["byte_length"]
    tail = payload[cursor:]
    if tail and any(tail):
        raise KilnError("TYPE")
    _check_quant(raw)
    if standalone and block_end != len(view):
        raise KilnError("TRAILING")
    tensors = []
    for item in raw:
        dtype = item["dtype"]
        tensors.append(
            {
                "name": item["name"].decode("utf-8") if item["name"] else None,
                "dtype": DTYPE_NAME.get(dtype, dtype),
                "shape": item["shape"][: item["rank"]],
                "data": item["data"],
                "quant_scheme": item["quant_scheme"],
                "quant_axis": item["quant_axis"],
                "group_size": item["group_size"],
                "scale_index": item["scale_index"],
                "outer_index": item["outer_index"],
                "layout": item["layout"].decode("utf-8") if item["layout"] else None,
                "block_size": item["block_size"],
            }
        )
    directory = {
        "version": 2,
        "flags": ["names", "le"] if names_flag else ["le"],
        "tensors": tensors,
        "block_length": block_end,
    }
    return directory, block_end
