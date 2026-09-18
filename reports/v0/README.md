# BuilderLab Spec Sheet Verification v0

## Purpose

BuilderLab Spec Sheet Verification v0 is a standalone pipeline for extracting
specified fields from an explicitly selected set of local PDFs, comparing
independent extraction paths, and sending insufficient or contradictory
evidence to human review. Verification semantics and thresholds remain
conservative: the system does not invent, normalize away, or silently select a
value when the evidence is insufficient.

## Final architecture

```text
PDF
├─ Tesseract OCR → Gemini structured extraction
├─ native PDF → Gemini Vision structured extraction
├─ semantic comparison
├─ lazy native-text fallback when needed
├─ upstream expected-SKU claim verification
├─ persisted exact-hash external provenance rescue
└─ verified / human review
```

## Results

| Snapshot | Verified | Human review | Failed |
| --- | ---: | ---: | ---: |
| Initial evaluation | 6 | 94 | 0 |
| Final clean end-to-end v0 result | **77** | **23** | **0** |

The clean 100-PDF run used 100 raw PDFs and cost **$0.2720724** in API calls,
approximately **$0.00272 per PDF**. Its average processing latency was
**27,896.99 ms/PDF** (median **8,530.51 ms/PDF**; total **2,789,699.47 ms**).
The complete automated test suite has **110 passing tests**.

The earlier **78/22/0** result is retained as useful historical evidence from
the frozen-artifact evaluation. It is not the release headline: **77/23/0** is
the final clean end-to-end v0 result after provider recovery.

## Improvements that worked

- Added bounded retry/backoff for transient Gemini rate-limit and server
  failures.
- Added lazy native-PDF text fallback when the primary evidence branches need
  more support.
- Added conservative upstream expected-SKU claim verification.
- Added persisted exact-hash manufacturer provenance that can rescue a
  document when model branches are unavailable, without loosening verification
  thresholds or field semantics.
- Preserved stage metrics, retry history, evidence provenance, and resumable
  batch artifacts for auditability.

## Experiments rejected for v0

Broad OCR/layout substitutions, filename-derived answers, aggressive value
normalization, and other candidate extraction branches were not promoted when
they failed to provide sufficiently reliable product evidence or would have
changed the conservative verification contract. The shipped path keeps the
independent OCR/Vision comparison and adds only evidence-backed fallbacks and
rescue paths.

## Remaining limitation

The 23 human-review documents deliberately remain human review because the
available evidence is insufficient, ambiguous, or contradictory for a safe
automatic decision. Human review is an intended v0 outcome, not a failed
verification threshold.

## Version-controlled evidence

- [`clean-run-report.json`](clean-run-report.json) — clean 100-PDF report.
- [`clean-run-manifest.json`](clean-run-manifest.json) — clean run document
  manifest; the local input path is redacted from the release copy.
- [`provider-recovery-live-report.json`](provider-recovery-live-report.json) —
  live provider-recovery report.
- [`BuilderLab_Spec_Sheet_Verification_v0_Final_Report.pdf`](BuilderLab_Spec_Sheet_Verification_v0_Final_Report.pdf) —
  final report supplied locally and copied into the release evidence folder.

The raw PDF corpus, virtual environment, transient downloaded provenance PDFs,
and `.runs/` trees remain excluded from version control.
