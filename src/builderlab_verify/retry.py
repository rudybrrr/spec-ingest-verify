"""Bounded retries for transient Gemini API responses."""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable


DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY_SECONDS = 0.5
DEFAULT_MAX_DELAY_SECONDS = 8.0


@dataclass
class RetryInfo:
    """Attempt history persisted alongside a provider stage."""

    attempts: int = 0
    retry_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "retry_count": self.retry_count,
            "errors": self.errors,
        }


class RetryFailure(RuntimeError):
    """The final provider error after a retry policy was applied."""

    def __init__(self, error: Exception, info: RetryInfo) -> None:
        super().__init__(str(error))
        self.error = error
        self.info = info


def http_status_code(error: Exception) -> int | None:
    """Return an HTTP status exposed by the SDK or its string representation."""

    for candidate in (
        getattr(error, "status_code", None),
        getattr(getattr(error, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    match = re.search(r"\b(?:HTTP\s*)?(\d{3})\b", str(error), re.IGNORECASE)
    return int(match.group(1)) if match else None


def is_transient_gemini_error(error: Exception) -> bool:
    """Retry only rate limits and server-side HTTP failures."""

    status_code = http_status_code(error)
    if status_code == 429 or (status_code is not None and 500 <= status_code < 600):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    error_name = type(error).__name__.lower()
    return any(token in error_name for token in ("apiconnectionerror", "apitimeouterror", "connecterror"))


def call_with_retry(
    operation: Callable[[], Any],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
    max_delay_seconds: float = DEFAULT_MAX_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    random_value: Callable[[], float] = random.random,
) -> tuple[Any, RetryInfo]:
    """Run an operation with bounded exponential backoff and multiplicative jitter."""

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    info = RetryInfo()
    for attempt in range(1, max_retries + 2):
        info.attempts = attempt
        try:
            return operation(), info
        except Exception as error:
            status_code = http_status_code(error)
            info.errors.append({
                "attempt": attempt,
                "status_code": status_code,
                "error": str(error),
                "transient": is_transient_gemini_error(error),
            })
            if not is_transient_gemini_error(error) or attempt > max_retries:
                raise RetryFailure(error, info) from error
            info.retry_count += 1
            delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
            delay *= 0.5 + random_value()
            sleep(delay)

    raise AssertionError("retry loop exited unexpectedly")
