"""Aggregate a batch's filesystem artifacts into a portable evaluation report."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from builderlab_verify.metrics import read_metrics
from builderlab_verify.schemas import CategorySchema


def build_batch_report(batch_root: Path, filenames: list[str], category: CategorySchema) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    statuses = Counter()
    escalation_fields = Counter()
    escalation_reasons = Counter()
    failures = Counter()
    stage_usage: dict[str, Counter] = {}
    stage_cost: Counter = Counter()
    stage_latency: Counter = Counter()
    latencies: list[float] = []
    provenance_statuses = Counter()
    provenance_rescued: list[str] = []
    provenance_exact_matches = 0
    for filename in filenames:
        manifest = json.loads((batch_root / "manifest.json").read_text(encoding="utf-8"))["documents"].get(filename, {})
        run_id = manifest.get("run_id")
        run = batch_root / "runs" / run_id if run_id else None
        metrics = read_metrics(run / "metrics.json") if run else {"stages": {}}
        status = manifest.get("status", "not_started")
        if status in ("verified", "human_review", "failed"):
            statuses[status] += 1
        latency = metrics.get("latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        for stage, data in metrics.get("stages", {}).items():
            usage = data.get("usage") or {}
            bucket = stage_usage.setdefault(stage, Counter())
            for key in ("prompt", "candidates", "thoughts", "total"):
                if usage.get(key) is not None:
                    bucket[key] += usage[key]
            if data.get("cost_usd") is not None:
                stage_cost[stage] += data["cost_usd"]
            if data.get("latency_ms") is not None:
                stage_latency[stage] += data["latency_ms"]
        if status == "human_review" and run:
            verification_path = run / "verification.json"
            if verification_path.exists():
                verification = json.loads(verification_path.read_text(encoding="utf-8"))
                for field, comparison in verification.get("fields", {}).items():
                    if comparison.get("status") != "match":
                        escalation_fields[field] += 1
                        escalation_reasons[comparison.get("status", "unknown")] += 1
                        if verification.get("notes", {}).get(field):
                            escalation_reasons[verification["notes"][field]] += 1
        if status == "failed":
            failures[metrics.get("failure_category", "api")] += 1
        provenance = {}
        if run:
            verification_path = run / "verification.json"
            if verification_path.exists():
                verification = json.loads(verification_path.read_text(encoding="utf-8"))
                provenance = verification.get("external_provenance", {})
        if provenance:
            provenance_statuses[provenance.get("status", "unknown")] += 1
            if provenance.get("accepted"):
                provenance_exact_matches += 1
            if provenance.get("rescued"):
                provenance_rescued.append(filename)
        rows.append({"filename": filename, "run_id": run_id, "status": status, "latency_ms": latency,
                     "external_provenance_status": provenance.get("status") if provenance else None})
    total = len(filenames)
    def pct(value: int) -> float:
        return round(value * 100 / total, 2) if total else 0.0
    return {
        "selection": {"count": total, "filenames": filenames, "schema": category.model_dump()},
        "counts": {key: {"count": statuses[key], "percentage": pct(statuses[key])} for key in ("verified", "human_review", "failed")},
        "not_started_count": total - sum(statuses.values()),
        "escalation_fields": escalation_fields.most_common(),
        "escalation_reasons": escalation_reasons.most_common(),
        "failure_taxonomy": dict(failures),
        "processing_time_ms": {
            "average": round(statistics.mean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "total": round(sum(latencies), 2),
        },
        "token_usage_by_stage": {stage: dict(values) for stage, values in stage_usage.items()},
        "cost_usd_by_stage": {stage: round(value, 8) for stage, value in stage_cost.items()},
        "latency_ms_by_stage": {stage: round(value, 2) for stage, value in stage_latency.items()},
        "total_api_cost_usd": round(sum(stage_cost.values()), 8),
        "average_api_cost_usd": round(sum(stage_cost.values()) / total, 8) if total else None,
        "documents": rows,
        "external_provenance": {
            "status_counts": dict(provenance_statuses),
            "accepted_exact_hash_match_count": provenance_exact_matches,
            "rescued_count": len(provenance_rescued),
            "rescued_documents": provenance_rescued,
        },
    }


def write_batch_report(batch_root: Path, filenames: list[str], category: CategorySchema) -> Path:
    path = batch_root / "report.json"
    path.write_text(json.dumps(build_batch_report(batch_root, filenames, category), indent=2) + "\n", encoding="utf-8")
    return path
