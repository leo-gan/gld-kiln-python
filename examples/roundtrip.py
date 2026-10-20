from kiln import Writer, decode

point = ("struct", [("x", ("i32",)), ("y", ("i32",))])
writer = Writer(64)
raw = writer.encode(point, {"x": 9, "y": -4})
print(decode(point, raw), len(raw))
