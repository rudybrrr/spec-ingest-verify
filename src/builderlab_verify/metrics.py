"""Lightweight, filesystem-backed operational metrics for pipeline runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from builderlab_verify.models import TokenUsage

# Rates are USD per million tokens. Override with MODEL_INPUT_RATE_USD and
# MODEL_OUTPUT_RATE_USD when the provider/model contract changes.
DEFAULT_INPUT_RATE = 0.10
DEFAULT_OUTPUT_RATE = 0.40


def _rate(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def estimate_cost(usage: TokenUsage | dict[str, Any] | None) -> float | None:
    if usage is None:
        return None
    data = usage if isinstance(usage, dict) else usage.model_dump()
    prompt, candidates = data.get("prompt"), data.get("candidates")
    if prompt is None and candidates is None:
        return None
    return round(
        (prompt or 0) / 1_000_000 * _rate("MODEL_INPUT_RATE_USD", DEFAULT_INPUT_RATE)
        + (candidates or 0) / 1_000_000 * _rate("MODEL_OUTPUT_RATE_USD", DEFAULT_OUTPUT_RATE),
        8,
    )


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def record_stage(path: Path, stage: str, *, model: str, usage: TokenUsage | dict[str, Any] | None,
                 latency_ms: float, error_category: str | None = None,
                 error: str | None = None, attempts: int = 1, retry_count: int = 0,
                 errors: list[dict[str, Any]] | None = None) -> None:
    payload = read_metrics(path)
    payload.setdefault("stages", {})[stage] = {
        "model": model,
        "usage": usage if isinstance(usage, dict) else (usage.model_dump() if usage else None),
        "cost_usd": estimate_cost(usage),
        "latency_ms": round(latency_ms, 2),
        "error_category": error_category,
        "error": error,
        "attempts": attempts,
        "retry_count": retry_count,
        "errors": errors or [],
    }
    write_metrics(path, payload)
