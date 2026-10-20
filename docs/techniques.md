# Techniques

Frozen values are positional. The schema stays in the program. Dict keys are sorted by their UTF-8 bytes so two encodes of the same mapping match. Floats keep every bit, including the sign of zero and the payload of a NaN.

The tensor directory is one Frozen value. Its magic is `KLNT` and its version byte is 2. Descriptors are 128 bytes. Payloads start at a multiple of 64 from the start of the directory. A borrowed read returns a `memoryview` of the caller's buffer. An owned copy is `bytes(view)`.

A count is checked against the unread bytes before Kiln allocates storage for the elements. The default cap is 16,777,216 counts and 16 MiB for one string or byte run. Pass `cap` to raise it.

A frame is sixteen bytes: `KLN0`, four zero bytes, and the FNV-1a 64 hash of the canonical type text, then one Frozen value. A hash that does not match the schema the caller passed is `SCHEMA`.

`Writer.encode` resets the length and keeps the existing `bytearray`. Call it in a loop when the same process encodes many records. `encode_tensors` keeps one scratch block per thread and returns a copy of the finished bytes, so two threads do not share that block.
