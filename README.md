# BuilderLab spec-sheet verification

This repository contains the frozen v0 BuilderLab spec-sheet ingestion and
verification pipeline. It accepts an explicitly selected set of local PDFs,
extracts the requested category fields through independent paths, compares the
results, and surfaces mismatches for human review.

## Setup

Requirements:

- Python 3.11 or newer
- A local Tesseract installation
- A Gemini API key with access to the configured models

Create a virtual environment, install the project, and create a local `.env`:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set these values in `.env`:

```text
GEMINI_API_KEY=your-key
TESSERACT_CMD=tesseract
TESSERACT_PSM=3
```

`GEMINI_API_KEY` is loaded from `.env` and is never written to pipeline
artifacts or logs. If Tesseract is not on `PATH`, set `TESSERACT_CMD` to its
executable path, for example:

```text
TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
```

`TESSERACT_PSM` defaults to `3` and controls the page segmentation mode passed
to Tesseract.

## Launch

From the repository root:

```powershell
python -m streamlit run streamlit_app.py
```

In the dashboard, enter an input folder, list the exact PDF filenames to
process, and provide the category schema JSON. For a mixed-category batch,
provide an explicit filename-to-schema mapping. The app does not scan the
folder, infer categories, or process unspecified files.

## Pipeline flow

Each selected PDF is processed sequentially:

1. Validate and copy the PDF into a local run directory.
2. Render each page and run Tesseract OCR, writing `ocr.txt`.
3. Send the OCR text and supplied schema to `gemini-3.5-flash-lite` with
   minimal thinking, writing structured `ocr.json`.
4. Send the original PDF and supplied schema to `gemini-3.8-flash` with low
   thinking, writing structured `vision.json`.
5. Compare corresponding fields with `gemini-3.5-flash-lite` using minimal
   thinking, writing `verification.json`.
6. When needed, lazily extract native PDF text as an additional evidence path.
7. Verify upstream expected-SKU claims without weakening the field semantics.
8. Apply persisted exact-hash external provenance rescue when the model
   branches are unavailable and the evidence is authoritative.
9. Show verified results or human-review fields in the dashboard. A reviewer
   can confirm or resolve a result, which writes `final.json`.

The verifier preserves both extraction snapshots. Missing values are handled
deterministically: a null/non-null difference is a mismatch; both-null
optional fields match; both-null required fields are uncertain and go to human
review. For two non-null values, the checker returns only `match` or `mismatch`
and an optional note. It does not correct, normalize, select, or invent a
value. In particular, description mismatches remain mismatches for review.

Run artifacts are stored under `.runs/` while the app is running. PDFs,
generated runs, smoke-test outputs, local test fixtures, and Streamlit local
state are ignored by Git because they may contain source documents or derived
content.

## Verify

```powershell
python -m pytest
```

The final v0 validation is documented in
[`reports/v0/README.md`](reports/v0/README.md): 77 verified, 23 human review,
0 failed on the clean 100-PDF end-to-end evaluation, with 110 automated tests
passing.

## Release evidence

The version-controlled v0 report directory contains the clean-run report and
manifest, the live provider-recovery report, and the concise release summary.
The raw PDFs and transient `.runs/` trees remain local-only.

## Current v0 limitations

- Local filesystem and one Streamlit process only; no database or durable job
  queue.
- Selected PDFs are processed sequentially and must be named explicitly.
- Category inference, authentication, deployment configuration, Docker, API
  serving, and V2 integration are out of scope.
- Tesseract and Gemini access are required for an end-to-end run.
- Human-review results require an explicit reviewer confirmation before
  `final.json` is written.
