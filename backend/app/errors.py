"""Stable public errors omit local paths and speech content."""


class ServiceError(Exception):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        self.code = code
        self.message = message
        self.status = status
        super().__init__(code)
