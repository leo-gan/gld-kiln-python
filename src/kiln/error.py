"""Kiln error codes. The names match the wire specification."""


class KilnError(Exception):
    """A decode or encode failure.

    `code` is one of FRAME, LIMIT, EOF, TYPE, UTF8, TRAILING, SCHEMA.
    """

    def __init__(self, code: str, offset: int = 0) -> None:
        super().__init__(code)
        self.code = code
        self.offset = offset
