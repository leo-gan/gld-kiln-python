"""Emit Python structs from the JSON Schema subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from kiln.schema import schema_from_json

_PY = {
    "bool": "bool",
    "i8": "int",
    "i16": "int",
    "i32": "int",
    "i64": "int",
    "u8": "int",
    "u16": "int",
    "u32": "int",
    "u64": "int",
    "f32": "float",
    "f64": "float",
    "str": "str",
    "bytes": "bytes",
}


def _py_type(schema: tuple) -> str:
    kind = schema[0]
    if kind in _PY:
        return _PY[kind]
    if kind == "list":
        return "list[" + _py_type(schema[1]) + "]"
    if kind == "opt":
        return _py_type(schema[1]) + " | None"
    if kind == "dict":
        return "dict[str, " + _py_type(schema[1]) + "]"
    if kind == "struct":
        return schema[2] if len(schema) == 3 else "dict"
    return "object"


def _named(schema: tuple, name: str) -> tuple:
    if schema[0] != "struct":
        return schema
    return ("struct", schema[1], name)


def _collect(schema: tuple, out: list[tuple]) -> None:
    kind = schema[0]
    if kind in ("list", "opt"):
        _collect(schema[1], out)
        return
    if kind == "dict":
        _collect(schema[1], out)
        return
    if kind != "struct":
        return
    for _, field in schema[1]:
        _collect(field, out)
    out.append(schema)


def _default(schema: tuple) -> str:
    kind = schema[0]
    if kind == "bool":
        return "False"
    if kind == "str":
        return '""'
    if kind == "bytes":
        return 'b""'
    if kind in ("f32", "f64"):
        return "0.0"
    if kind in ("list",):
        return "field(default_factory=list)"
    if kind == "dict":
        return "field(default_factory=dict)"
    if kind == "opt":
        return "None"
    if kind == "struct":
        return "field(default_factory=" + _py_type(schema) + ")"
    return "0"


def emit(text: str) -> str:
    title, root = schema_from_json(text)
    if root[0] != "struct":
        raise SystemExit("root must be an object")
    root = ("struct", root[1], title)
    structs: list[tuple] = []
    _collect(root, structs)
    lines = [
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
        "",
        "from kiln import decode, encode",
        "",
    ]
    for schema in structs:
        name = schema[2]
        lines.append("@dataclass")
        lines.append(f"class {name}:")
        for fname, fschema in schema[1]:
            lines.append(f"    {fname}: {_py_type(fschema)} = {_default(fschema)}")
        lines.append("")
        lines.append("    def encode(self) -> bytes:")
        lines.append(f"        return encode({name}._schema(), self.to_value())")
        lines.append("")
        lines.append("    @staticmethod")
        lines.append(f"    def decode(buf: bytes) -> {name}:")
        lines.append(f"        return {name}.from_value(decode({name}._schema(), buf))")
        lines.append("")
        lines.append("    @staticmethod")
        lines.append("    def encoded_size(value: dict) -> int:")
        lines.append(f"        return len(encode({name}._schema(), value))")
        lines.append("")
        lines.append("    def to_value(self) -> dict:")
        lines.append("        return {")
        for fname, fschema in schema[1]:
            if fschema[0] == "struct":
                lines.append(f'            "{fname}": self.{fname}.to_value(),')
            elif fschema[0] == "opt" and fschema[1][0] == "struct":
                lines.append(
                    f'            "{fname}": None if self.{fname} is None else self.{fname}.to_value(),'
                )
            elif fschema[0] == "list" and fschema[1][0] == "struct":
                lines.append(f'            "{fname}": [item.to_value() for item in self.{fname}],')
            else:
                lines.append(f'            "{fname}": self.{fname},')
        lines.append("        }")
        lines.append("")
        lines.append("    @staticmethod")
        lines.append(f"    def from_value(value: dict) -> {name}:")
        args = []
        for fname, fschema in schema[1]:
            if fschema[0] == "struct":
                child = fschema[2]
                args.append(f'{fname}={child}.from_value(value["{fname}"])')
            elif fschema[0] == "opt" and fschema[1][0] == "struct":
                child = fschema[1][2]
                args.append(
                    f'{fname}=None if value["{fname}"] is None else {child}.from_value(value["{fname}"])'
                )
            elif fschema[0] == "list" and fschema[1][0] == "struct":
                child = fschema[1][2]
                args.append(
                    f'{fname}=[{child}.from_value(item) for item in value["{fname}"]]'
                )
            else:
                args.append(f'{fname}=value["{fname}"]')
        lines.append(f"        return {name}({', '.join(args)})")
        lines.append("")
        lines.append("    @staticmethod")
        lines.append("    def _schema() -> tuple:")
        lines.append(f"        return {schema[0:2]!r}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gld-kilngen-python")
    parser.add_argument("schema")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    text = Path(args.schema).read_text()
    Path(args.output).write_text(emit(text))


if __name__ == "__main__":
    main()
