"""Small-record and tensor microbench. Prints nanoseconds per call."""

import struct
import time

from kiln import Writer, decode, decode_tensors, encode, encode_tensors

POINT = ("struct", [("x", ("i32",)), ("y", ("i32",)), ("label", ("str",))])
VALUE = {"x": 9, "y": -4, "label": "item"}
TENSOR = [{"dtype": "f32", "shape": [1024], "data": struct.pack("<1024f", *([1.0] * 1024))}]


def bench(fn, n=2000, warm=100):
    for _ in range(warm):
        fn()
    start = time.perf_counter_ns()
    for _ in range(n):
        fn()
    return (time.perf_counter_ns() - start) // n


def main() -> None:
    writer = Writer(256)
    block = encode_tensors(TENSOR)

    def enc():
        writer.encode(POINT, VALUE)

    view = memoryview(writer.buf)

    def dec():
        decode(POINT, view[: writer.end])

    writer.encode(POINT, VALUE)
    print("struct_encode_ns", bench(enc))
    print("struct_decode_ns", bench(dec))
    print("struct_bytes", writer.end)
    print("tensor_encode_ns", bench(lambda: encode_tensors(TENSOR), n=200, warm=20))
    print("tensor_decode_ns", bench(lambda: decode_tensors(block), n=200, warm=20))
    print("tensor_bytes", len(block))
    column = ("list", ("f32",))
    values = [1.0] * 1024
    print("list_f32_encode_ns", bench(lambda: encode(column, values), n=400, warm=20))
    raw_list = encode(column, values)
    print("list_f32_decode_ns", bench(lambda: decode(column, raw_list), n=400, warm=20))


if __name__ == "__main__":
    main()
