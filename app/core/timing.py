"""Small helpers for consistent performance logging."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator


def _format_value(value: object, max_len: int = 120) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


@contextmanager
def timed_stage(logger: logging.Logger, stage: str, **fields: object) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        detail = " ".join(
            f"{key}={_format_value(value)}"
            for key, value in fields.items()
            if value is not None
        )
        suffix = f" {detail}" if detail else ""
        logger.info("[PERF] stage=%s elapsed_ms=%.2f%s", stage, elapsed_ms, suffix)
