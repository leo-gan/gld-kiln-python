"""JSON Schema subset used by the generator. See the Kiln wire spec."""

from __future__ import annotations

import json
from typing import Any

from kiln.error import KilnError

_INT = {
    "int8": "i8",
    "int16": "i16",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
}


def parse_type_text(text: str) -> tuple:
    """Parse a case type such as `f32` or `struct{a:i64}`."""
    text = text.strip()
    if text.startswith("struct{") and text.endswith("}"):
        body = text[len("struct{") : -1]
        fields = []
        if body:
            for part in body.split(","):
                if not part:
                    continue
                name, typ = part.split(":")
                fields.append((name, parse_type_text(typ)))
        return ("struct", fields)
    if text.startswith("list<") and text.endswith(">"):
        return ("list", parse_type_text(text[5:-1]))
    if text.startswith("opt<") and text.endswith(">"):
        return ("opt", parse_type_text(text[4:-1]))
    return (text,)


def schema_from_case(schema: Any) -> tuple:
    if schema == "tensor-v2":
        return ("tensors",)
    if schema == "frame":
        return ("struct", [("id", ("i64",)), ("batch", ("tensors",))])
    if isinstance(schema, dict) and schema.get("type") == "list":
        return ("list", parse_type_text(schema["element"]))
    if isinstance(schema, dict) and schema.get("type") == "struct":
        fields = []
        for item in schema["fields"]:
            name, typ = item.split(":", 1)
            fields.append((name, parse_type_text(typ)))
        return ("struct", fields)
    raise KilnError("SCHEMA")


def schema_from_json(text: str) -> tuple:
    """Return `(struct_name, schema_tuple)` for a JSON Schema object."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise KilnError("SCHEMA") from exc
    if not isinstance(doc, dict):
        raise KilnError("SCHEMA")
    defs = doc.get("$defs") or {}
    cache: dict[str, tuple] = {}
    building: set[str] = set()

    def convert(node: dict, label: str) -> tuple:
        if "$ref" in node:
            ref = node["$ref"]
            prefix = "#/$defs/"
            if not isinstance(ref, str) or not ref.startswith(prefix):
                raise KilnError("SCHEMA")
            return resolve(ref[len(prefix) :])
        kind = node.get("type")
        if kind == "boolean":
            return ("bool",)
        if kind == "string":
            if node.get("format") == "binary":
                return ("bytes",)
            return ("str",)
        if kind == "integer":
            return (_INT.get(node.get("format") or "", "i64"),)
        if kind == "number":
            if node.get("format") == "float":
                return ("f32",)
            return ("f64",)
        if kind == "array":
            items = node.get("items")
            if not isinstance(items, dict):
                raise KilnError("SCHEMA")
            return ("list", convert(items, label + "Item"))
        if kind == "object":
            props = node.get("properties") or {}
            extra = node.get("additionalProperties")
            if isinstance(extra, dict) and not props:
                return ("dict", convert(extra, label + "Value"))
            return object_schema(node, label)
        raise KilnError("SCHEMA")

    def resolve(name: str) -> tuple:
        if name in cache:
            return cache[name]
        if name in building or name not in defs:
            raise KilnError("SCHEMA")
        building.add(name)
        built = convert(defs[name], name)
        building.remove(name)
        cache[name] = built
        return built

    def object_schema(node: dict, label: str) -> tuple:
        props = node.get("properties") or {}
        required = node.get("required")
        fields = []
        for fname, body in props.items():
            field = convert(body, label + "_" + fname)
            if isinstance(required, list) and fname not in required:
                field = ("opt", field)
            fields.append((fname, field))
        return ("struct", fields, label)

    title = doc.get("title") or "Root"
    root = convert(doc, title)
    return title, root
