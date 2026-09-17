"""Minimal Streamlit dashboard for one spec-sheet verification run."""

from __future__ import annotations

import html
import hashlib
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from builderlab_verify.dashboard import (  # noqa: E402
    build_final_result,
    parse_category_json,
    parse_json_scalar,
    parse_schema_mapping_json,
    write_final_result,
)
from builderlab_verify.batch import BatchRunner, DocumentStatus  # noqa: E402
from builderlab_verify.ingestion import ingest_pdf  # noqa: E402
from builderlab_verify.models import FieldProvenance, RunStatus, VerificationResult  # noqa: E402
from builderlab_verify.ocr import extract_ocr_to_json  # noqa: E402
from builderlab_verify.storage import RunDirectory  # noqa: E402
from builderlab_verify.verification import verify_ocr_vision  # noqa: E402
from builderlab_verify.vision import extract_vision_to_json  # noqa: E402


RUNS_DIR = ROOT / ".runs"
DEFAULT_SCHEMA = json.dumps(
    {
        "name": "spec_sheet",
        "fields": [
            {"name": "part_number", "description": "Manufacturer part number", "required": True},
            {"name": "description", "description": "Short product description", "required": False},
        ],
    },
    indent=2,
)


def _display_value(field: FieldProvenance) -> str:
    if field.value is None:
        return "—"
    value = str(field.value)
    return f"{value} {field.unit}".strip() if field.unit else value


def _status_badge(status: str) -> str:
    colors = {"match": "#16803c", "mismatch": "#c62828", "uncertain": "#a15c00"}
    color = colors.get(status, "#555")
    return (
        f'<span style="background:{color};color:white;padding:0.2rem 0.55rem;'
        f'border-radius:0.8rem;font-weight:600">{html.escape(status)}</span>'
    )


def _render_verification(run: RunDirectory, category) -> None:
    verification = VerificationResult.model_validate_json(run.verification_json.read_text(encoding="utf-8"))
    st.subheader("Verification result")
    if verification.status is RunStatus.VERIFIED:
        st.success("Overall status: verified")
    else:
        st.warning("Overall status: human_review")

    st.markdown("| Field | OCR | Vision | Status | Note |\n|---|---|---|---|---|")
    for field in category.fields:
        comparison = verification.fields[field.name]
        note = verification.notes.get(field.name, "")
        st.markdown(
            f"| `{html.escape(field.name)}` | {html.escape(_display_value(comparison.ocr))} | "
            f"{html.escape(_display_value(comparison.vision))} | {_status_badge(comparison.status.value)} | "
            f"{html.escape(note)} |",
            unsafe_allow_html=True,
        )

    if verification.status is RunStatus.VERIFIED:
        if st.button("Write final.json", key="write_verified"):
            fields = {name: comparison.ocr for name, comparison in ((n, verification.fields[n]) for n in verification.fields)}
            result = build_final_result(verification, fields)
            write_final_result(run.final_json, result)
            st.success(f"Wrote {run.final_json}")
        return

    st.subheader("Human resolution")
    st.caption("Enter a JSON scalar or plain text for each escalated field. Matches remain unchanged.")
    resolution_errors = False
    with st.form("resolution_form"):
        reviewer = st.text_input("Reviewer")
        resolved_fields: dict[str, FieldProvenance] = {}
        for field in category.fields:
            comparison = verification.fields[field.name]
            if comparison.status.value == "match":
                resolved_fields[field.name] = comparison.ocr
                continue
            current = "" if comparison.ocr.value is None else json.dumps(comparison.ocr.value)
            value = st.text_input(field.name, value=current, key=f"resolved_{field.name}")
            unit = st.text_input(f"{field.name} unit", value=comparison.ocr.unit or comparison.vision.unit or "", key=f"unit_{field.name}")
            page = st.number_input(f"{field.name} page", min_value=1, value=comparison.ocr.page or comparison.vision.page or 1, step=1, key=f"page_{field.name}")
            try:
                parsed = parse_json_scalar(value)
                resolved_fields[field.name] = FieldProvenance(value=parsed, unit=unit or None, page=int(page))
            except ValueError as error:
                resolution_errors = True
                st.error(f"{field.name}: {error}")
        note = st.text_area("Resolution note (optional)")
        submitted = st.form_submit_button("Confirm and write final.json")

    if submitted and resolution_errors:
        st.error("Fix the highlighted field values before writing final.json.")
    elif submitted:
        try:
            result = build_final_result(verification, resolved_fields, reviewer=reviewer, note=note)
            write_final_result(run.final_json, result)
        except ValueError as error:
            st.error(str(error))
        else:
            st.success(f"Wrote {run.final_json}")


def main() -> None:
    st.set_page_config(page_title="Spec-sheet verification", layout="wide")
    st.title("Spec-sheet batch verification")
    st.caption("Select a bounded set of PDFs, run the existing per-document pipeline, and review escalations.")

    input_dir_text = st.text_input("Input folder", placeholder=r"C:\\path\\to\\specsheets")
    filenames_text = st.text_area(
        "PDF filenames (one per line)",
        placeholder="Enter explicit filenames; the app will not scan or process the whole folder.",
        height=120,
    )
    schema_text = st.text_area("One explicit category schema for this batch", value=DEFAULT_SCHEMA, height=220)
    mapping_text = st.text_area(
        "Optional explicit filename-to-schema mapping JSON (for mixed categories)",
        value="",
        height=120,
        placeholder='{"motor.pdf": {"name": "motor", "fields": [...]}}',
    )
    run_clicked = st.button("Run selected batch", type="primary", disabled=not input_dir_text.strip() or not filenames_text.strip())

    if run_clicked:
        try:
            category = parse_category_json(schema_text)
            filenames = [line.strip() for line in filenames_text.splitlines() if line.strip()]
            schema_mapping = parse_schema_mapping_json(mapping_text) if mapping_text.strip() else None
            batch_key = json.dumps(
                {"input_dir": str(Path(input_dir_text.strip()).resolve()), "filenames": filenames},
                sort_keys=True,
            ).encode("utf-8")
            batch_id = f"batch-{hashlib.sha256(batch_key).hexdigest()[:12]}"
            runner = BatchRunner(RUNS_DIR / batch_id)
            with st.status("Running selected batch…", expanded=True) as progress:
                results = runner.run(Path(input_dir_text.strip()), filenames, category, schema_by_filename=schema_mapping)
                progress.update(label="Batch complete", state="complete")
            st.session_state["batch_root"] = str(runner.batch_root)
            st.session_state["batch_results"] = [result.__dict__ for result in results]
            st.session_state["batch_category"] = category.model_dump_json()
        except Exception as error:
            st.error(str(error))

    batch_root = st.session_state.get("batch_root")
    batch_category_json = st.session_state.get("batch_category")
    batch_results = st.session_state.get("batch_results", [])
    if not batch_root or not batch_category_json:
        st.info("Provide an input folder, explicit PDF filenames, and a category schema to run the batch.")
        return

    category = parse_category_json(batch_category_json)
    counts = {status.value: 0 for status in DocumentStatus}
    for result in batch_results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    columns = st.columns(5)
    for column, label, value in zip(
        columns,
        ("Processed", "Verified", "Human review", "Failed", "Total"),
        (len(batch_results), counts[DocumentStatus.VERIFIED.value], counts[DocumentStatus.HUMAN_REVIEW.value], counts[DocumentStatus.FAILED.value], len(batch_results)),
    ):
        column.metric(label, value)
    st.subheader("Batch documents")
    st.dataframe(
        [{"PDF": result["filename"], "Status": result["status"], "Error": result.get("error") or ""} for result in batch_results],
        use_container_width=True,
        hide_index=True,
    )
    options = [result["filename"] for result in batch_results]
    selected = st.selectbox("Open document", options) if options else None
    if not selected:
        return
    selected_result = next(result for result in batch_results if result["filename"] == selected)
    run = RunDirectory(Path(batch_root) / "runs" / selected_result["run_id"])
    if selected_result["status"] == DocumentStatus.FAILED.value:
        st.error(selected_result.get("error") or "Document failed without an error message.")
        return
    st.caption(f"Run: `{run.root.name}` · source: `{run.source_pdf}`")
    if run.source_pdf.exists():
        st.download_button("Download source PDF", data=run.source_pdf.read_bytes(), file_name=run.source_pdf.name, mime="application/pdf")
    _render_verification(run, category)


if __name__ == "__main__":
    main()
