"""Compiled pure-Python encode/decode plans for Kiln schemas."""

from __future__ import annotations

import array
import struct
import sys
from typing import Any, Callable

from kiln.error import KilnError
from kiln.tensor import CAP

_LE = sys.byteorder == "little"

_WIDTH: dict[str, int] = {
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

# Struct format character per scalar. For decode, bool reads unsigned byte ("B")
# so values > 1 are rejected with KilnError("TYPE").
_ENC_FMT: dict[str, str] = {
    "bool": "?",
    "i8": "b",
    "u8": "B",
    "i16": "h",
    "u16": "H",
    "i32": "i",
    "u32": "I",
    "f32": "f",
    "i64": "q",
    "u64": "Q",
    "int": "q",
    "uint": "Q",
    "f64": "d",
}

_DEC_FMT: dict[str, str] = {
    **_ENC_FMT,
    "bool": "B",
}

_ARRAY_CODE_RAW: dict[str, str] = {
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

_ARRAY_CODE: dict[str, str] = {}
if _LE:
    for _k, _c in _ARRAY_CODE_RAW.items():
        if array.array(_c).itemsize == _WIDTH[_k]:
            _ARRAY_CODE[_k] = _c

_PACK_I = struct.Struct("<I").pack
_UNPACK_I = struct.Struct("<I").unpack_from
_HDRS = tuple(_PACK_I(i) for i in range(257))


def _min_size(schema: tuple) -> int:
    kind = schema[0]
    w = _WIDTH.get(kind)
    if w is not None:
        return w
    if kind in ("str", "bytes", "list"):
        return 4
    if kind == "opt":
        return 1
    if kind == "struct":
        return sum(_min_size(field) for _, field in schema[1])
    return -1


def _can_compile(schema: tuple, depth: int = 0) -> bool:
    kind = schema[0]
    if depth >= 12 and kind in ("list", "opt", "struct"):
        return False
    if kind in _WIDTH or kind in ("str", "bytes"):
        return True
    if kind in ("opt", "list"):
        return _can_compile(schema[1], depth + 1)
    if kind == "struct":
        return all(_can_compile(field, depth + 1) for _, field in schema[1])
    return False


def compile_encode(schema: tuple) -> Callable[[Any, int], bytes] | None:
    """Compile a specialized pure-Python encoder for `schema`, or None if unsupported."""
    if not _can_compile(schema, 0):
        return None

    env: dict[str, Any] = {
        "_KE": KilnError,
        "_pI": _PACK_I,
        "_H": _HDRS,
        "_array": array.array,
        "_bytes": bytes,
        "_len": len,
        "_map": map,
        "_senc": str.encode,
        "_struct": struct,
    }
    structs: dict[str, str] = {}

    def get_pack(fmt: str) -> str:
        name = structs.get(fmt)
        if name is None:
            name = f"_pk{len(structs)}"
            env[name] = struct.Struct("<" + fmt).pack
            structs[fmt] = name
        return name

    # Fast single-call path for pure fixed-width top-level struct or scalar.
    if schema[0] in _WIDTH:
        pk = get_pack(_ENC_FMT[schema[0]])
        ns: dict[str, Any] = {}
        exec(f"def _enc(val, cap):\n    return {pk}(val)\n", env, ns)
        return ns["_enc"]

    lines: list[str] = [
        "def _enc(val, cap):",
        "    parts = []",
        "    ap = parts.append",
        "    H = _H",
        "    pI = _pI",
        "    ln = _len",
    ]
    vid = 0

    def fresh(prefix: str) -> str:
        nonlocal vid
        vid += 1
        return f"_{prefix}{vid}"

    def emit(node: tuple, expr: str, indent: str) -> None:
        kind = node[0]
        if kind in _WIDTH:
            pk = get_pack(_ENC_FMT[kind])
            lines.append(f"{indent}ap({pk}({expr}))")
            return
        if kind == "str":
            b = fresh("b")
            n = fresh("n")
            lines.append(f"{indent}{b} = {expr}.encode()")
            lines.append(f"{indent}{n} = ln({b})")
            lines.append(f"{indent}if {n} > cap: raise _KE('LIMIT')")
            lines.append(f"{indent}ap(H[{n}] if {n} <= 256 else pI({n}))")
            lines.append(f"{indent}ap({b})")
            return
        if kind == "bytes":
            b = fresh("b")
            n = fresh("n")
            lines.append(f"{indent}{b} = {expr}")
            lines.append(f"{indent}if type({b}) is not _bytes: {b} = _bytes({b})")
            lines.append(f"{indent}{n} = ln({b})")
            lines.append(f"{indent}if {n} > cap: raise _KE('LIMIT')")
            lines.append(f"{indent}ap(H[{n}] if {n} <= 256 else pI({n}))")
            lines.append(f"{indent}ap({b})")
            return
        if kind == "opt":
            ov = fresh("o")
            lines.append(f"{indent}{ov} = {expr}")
            lines.append(f"{indent}if {ov} is None:")
            lines.append(f"{indent}    ap(b'\\x00')")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    ap(b'\\x01')")
            emit(node[1], ov, indent + "    ")
            return
        if kind == "struct":
            sv = fresh("s")
            fields = node[1]
            lines.append(f"{indent}{sv} = {expr}")
            run_fmt: list[str] = []
            run_args: list[str] = []

            def flush_run() -> None:
                if not run_fmt:
                    return
                pk = get_pack("".join(run_fmt))
                lines.append(f"{indent}ap({pk}({', '.join(run_args)}))")
                run_fmt.clear()
                run_args.clear()

            for fname, fschema in fields:
                fk = fschema[0]
                if fk in _WIDTH:
                    run_fmt.append(_ENC_FMT[fk])
                    run_args.append(f"{sv}[{fname!r}]")
                else:
                    flush_run()
                    emit(fschema, f"{sv}[{fname!r}]", indent)
            flush_run()
            return
        if kind == "list":
            elem = node[1]
            ek = elem[0]
            lv = fresh("l")
            cv = fresh("c")
            lines.append(f"{indent}{lv} = {expr}")
            lines.append(f"{indent}{cv} = ln({lv})")
            lines.append(f"{indent}if {cv} > cap: raise _KE('LIMIT')")
            lines.append(f"{indent}ap(H[{cv}] if {cv} <= 256 else pI({cv}))")
            if ek == "u8":
                lines.append(f"{indent}if {cv}: ap(_bytes({lv}))")
                return
            if ek in _ARRAY_CODE:
                tc = _ARRAY_CODE[ek]
                lines.append(f"{indent}if {cv}:")
                lines.append(f"{indent}    if type({lv}) is _array and {lv}.typecode == {tc!r}:")
                lines.append(f"{indent}        ap({lv})")
                lines.append(f"{indent}    else:")
                lines.append(f"{indent}        ap(_array({tc!r}, {lv}))")
                return
            if ek in _WIDTH:
                ch = _ENC_FMT[ek]
                lines.append(f"{indent}if {cv}: ap(_struct.pack('<' + {ch!r} * {cv}, *{lv}))")
                return
            if ek == "str":
                bv = fresh("b")
                nv = fresh("n")
                lines.append(f"{indent}for {bv} in _map(_senc, {lv}):")
                lines.append(f"{indent}    {nv} = ln({bv})")
                lines.append(f"{indent}    if {nv} > cap: raise _KE('LIMIT')")
                lines.append(f"{indent}    ap(H[{nv}] if {nv} <= 256 else pI({nv}))")
                lines.append(f"{indent}    ap({bv})")
                return
            iv = fresh("it")
            lines.append(f"{indent}for {iv} in {lv}:")
            emit(elem, iv, indent + "    ")
            return

    emit(schema, "val", "    ")
    lines.append("    return b''.join(parts)")

    ns = {}
    exec("\n".join(lines), env, ns)
    return ns["_enc"]


def compile_decode(
    schema: tuple, *, pod_arrays: bool = False
) -> Callable[[bytes | bytearray | memoryview, int, bool], Any] | None:
    """Compile a specialized pure-Python decoder for `schema`, or None if unsupported."""
    if not _can_compile(schema, 0):
        return None

    env: dict[str, Any] = {
        "_KE": KilnError,
        "_uI": _UNPACK_I,
        "_array": array.array,
        "_bytes": bytes,
        "_mv": memoryview,
        "_bdec": bytes.decode,
        "_len": len,
        "_range": range,
        "_list": list,
        "_struct": struct,
        "_UDE": UnicodeDecodeError,
        "_SE": struct.error,
    }
    structs: dict[str, tuple[str, int]] = {}

    def get_unpack(fmt: str) -> tuple[str, int]:
        hit = structs.get(fmt)
        if hit is None:
            st = struct.Struct("<" + fmt)
            name = f"_up{len(structs)}"
            env[name] = st.unpack_from
            hit = (name, st.size)
            structs[fmt] = hit
        return hit

    lines: list[str] = [
        "def _dec(buf, cap, standalone):",
        "    if type(buf) is not _bytes:",
        "        buf = _bytes(buf)",
        "    blen = _len(buf)",
        "    uI = _uI",
        "    bdec = _bdec",
        "    o = 0",
        "    try:",
    ]
    vid = 0

    def fresh(prefix: str) -> str:
        nonlocal vid
        vid += 1
        return f"_{prefix}{vid}"

    def emit(node: tuple, dst: str, indent: str) -> None:
        kind = node[0]
        if kind in _WIDTH:
            up, sz = get_unpack(_DEC_FMT[kind])
            if kind == "bool":
                bv = fresh("b")
                lines.append(f"{indent}{bv}, = {up}(buf, o)")
                lines.append(f"{indent}if {bv} > 1: raise _KE('TYPE', o)")
                lines.append(f"{indent}{dst} = {bv} == 1")
            else:
                lines.append(f"{indent}{dst}, = {up}(buf, o)")
            lines.append(f"{indent}o += {sz}")
            return
        if kind == "str":
            sl = fresh("sl")
            se = fresh("se")
            lines.append(f"{indent}{sl}, = uI(buf, o)")
            lines.append(f"{indent}if {sl} > cap: raise _KE('LIMIT', o)")
            lines.append(f"{indent}o += 4")
            lines.append(f"{indent}{se} = o + {sl}")
            lines.append(f"{indent}if {se} > blen: raise _KE('EOF', o)")
            lines.append(f"{indent}{dst} = bdec(buf[o:{se}])")
            lines.append(f"{indent}o = {se}")
            return
        if kind == "bytes":
            bl = fresh("bl")
            be = fresh("be")
            lines.append(f"{indent}{bl}, = uI(buf, o)")
            lines.append(f"{indent}if {bl} > cap: raise _KE('LIMIT', o)")
            lines.append(f"{indent}o += 4")
            lines.append(f"{indent}{be} = o + {bl}")
            lines.append(f"{indent}if {be} > blen: raise _KE('EOF', o)")
            lines.append(f"{indent}{dst} = buf[o:{be}]")
            lines.append(f"{indent}o = {be}")
            return
        if kind == "opt":
            tg = fresh("tg")
            lines.append(f"{indent}if o >= blen: raise _KE('EOF', o)")
            lines.append(f"{indent}{tg} = buf[o]")
            lines.append(f"{indent}o += 1")
            lines.append(f"{indent}if {tg} == 0:")
            lines.append(f"{indent}    {dst} = None")
            lines.append(f"{indent}elif {tg} == 1:")
            emit(node[1], dst, indent + "    ")
            lines.append(f"{indent}else:")
            lines.append(f"{indent}    raise _KE('TYPE', o - 1)")
            return
        if kind == "struct":
            fields = node[1]
            fvars = [fresh("f") for _ in fields]
            run_fmt: list[str] = []
            run_vars: list[tuple[str, bool]] = []

            def flush_run() -> None:
                if not run_fmt:
                    return
                up, sz = get_unpack("".join(run_fmt))
                lhs = ", ".join(v for v, _ in run_vars)
                if len(run_vars) == 1:
                    lhs += ","
                lines.append(f"{indent}{lhs} = {up}(buf, o)")
                for v, is_bool in run_vars:
                    if is_bool:
                        lines.append(f"{indent}if {v} > 1: raise _KE('TYPE', o)")
                        lines.append(f"{indent}{v} = {v} == 1")
                lines.append(f"{indent}o += {sz}")
                run_fmt.clear()
                run_vars.clear()

            for (fname, fschema), fv in zip(fields, fvars):
                fk = fschema[0]
                if fk in _WIDTH:
                    run_fmt.append(_DEC_FMT[fk])
                    run_vars.append((fv, fk == "bool"))
                else:
                    flush_run()
                    emit(fschema, fv, indent)
            flush_run()
            pairs = ", ".join(f"{fname!r}: {fv}" for (fname, _), fv in zip(fields, fvars))
            lines.append(f"{indent}{dst} = {{{pairs}}}")
            return
        if kind == "list":
            elem = node[1]
            ek = elem[0]
            cv = fresh("c")
            emin = 1 if ek in ("u8", "i8") else _min_size(elem)
            lines.append(f"{indent}{cv}, = uI(buf, o)")
            lines.append(f"{indent}if {cv} > cap: raise _KE('LIMIT', o)")
            lines.append(f"{indent}o += 4")
            if emin > 0:
                lines.append(f"{indent}if {cv} * {emin} > blen - o: raise _KE('EOF', o)")
            if ek == "u8":
                lines.append(f"{indent}{dst} = buf[o : o + {cv}]")
                lines.append(f"{indent}o += {cv}")
                return
            if ek in _ARRAY_CODE:
                tc = _ARRAY_CODE[ek]
                w = _WIDTH[ek]
                av = fresh("a")
                nb = fresh("nb")
                lines.append(f"{indent}{av} = _array({tc!r})")
                lines.append(f"{indent}if {cv}:")
                lines.append(f"{indent}    {nb} = {cv} * {w}")
                lines.append(
                    f"{indent}    {av}.frombytes(_mv(buf)[o : o + {nb}] if {nb} > 4096 else buf[o : o + {nb}])"
                )
                lines.append(f"{indent}    o += {nb}")
                if pod_arrays:
                    lines.append(f"{indent}{dst} = {av}")
                else:
                    lines.append(f"{indent}{dst} = {av}.tolist()")
                return
            if ek in _WIDTH and ek != "bool":
                ch = _DEC_FMT[ek]
                w = _WIDTH[ek]
                nb = fresh("nb")
                lines.append(f"{indent}if {cv}:")
                lines.append(f"{indent}    {nb} = {cv} * {w}")
                lines.append(f"{indent}    {dst} = _list(_struct.unpack_from('<' + {ch!r} * {cv}, buf, o))")
                lines.append(f"{indent}    o += {nb}")
                lines.append(f"{indent}else:")
                lines.append(f"{indent}    {dst} = []")
                return
            outv = fresh("ls")
            iv = fresh("k")
            if ek == "str":
                sl = fresh("sl")
                se = fresh("se")
                lines.append(f"{indent}{outv} = [None] * {cv}")
                lines.append(f"{indent}for {iv} in _range({cv}):")
                lines.append(f"{indent}    {sl}, = uI(buf, o)")
                lines.append(f"{indent}    if {sl} > cap: raise _KE('LIMIT', o)")
                lines.append(f"{indent}    o += 4")
                lines.append(f"{indent}    {se} = o + {sl}")
                lines.append(f"{indent}    if {se} > blen: raise _KE('EOF', o)")
                lines.append(f"{indent}    {outv}[{iv}] = bdec(buf[o:{se}])")
                lines.append(f"{indent}    o = {se}")
                lines.append(f"{indent}{dst} = {outv}")
                return
            lines.append(f"{indent}{outv} = [None] * {cv}")
            lines.append(f"{indent}for {iv} in _range({cv}):")
            emit(elem, f"{outv}[{iv}]", indent + "    ")
            lines.append(f"{indent}{dst} = {outv}")
            return

    emit(schema, "root", "        ")
    lines.extend([
        "    except _SE as _exc:",
        "        raise _KE('EOF', o) from _exc",
        "    except _UDE as _exc:",
        "        raise _KE('UTF8', o) from _exc",
        "    if standalone:",
        "        if o != blen: raise _KE('TRAILING', o)",
        "        return root",
        "    return root, o",
    ])

    ns: dict[str, Any] = {}
    exec("\n".join(lines), env, ns)
    return ns["_dec"]
