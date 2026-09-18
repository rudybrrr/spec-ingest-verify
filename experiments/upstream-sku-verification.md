# Upstream SKU verification experiment

Metadata source: `input/index.csv` (`SKU, Category, File, SizeKB, SourcePath`).
Production code changed: **no**.

The source contains 8,668 rows; all 100 selected evaluation PDFs have a populated, unique SKU row (100/100 coverage).
The frozen evaluation has 45 human-review PDFs. Results: supported **10**, contradicted **1**, insufficient evidence **34**.
Safe additional auto-verifications under this conservative rule: **10**.

A case is supported by an explicit part-number/order-code label in PDF evidence or by matching evidence from at least two independent families. Filenames and the index itself are never PDF evidence.
Contradicted requires an explicitly labeled alternative or the same non-matching structured identifier from at least two branches; all other cases abstain.

## Cases

| PDF | Expected SKU | Result | Evidence |
|---|---|---|---|
| `00551106.pdf` | `00551106` | insufficient evidence | none |
| `00571102.pdf` | `00571102` | insufficient evidence | none |
| `00591104.pdf` | `00591104` | insufficient evidence | none |
| `00601102.pdf` | `00601102` | insufficient evidence | none |
| `00681101.pdf` | `00681101` | insufficient evidence | none |
| `00701101.pdf` | `00701101` | insufficient evidence | none |
| `00741200.pdf` | `00741200` | insufficient evidence | none |
| `00751200.pdf` | `00751200` | insufficient evidence | none |
| `01.02.120.pdf` | `01.02.120` | insufficient evidence | none |
| `02.02.120.pdf` | `02.02.120` | insufficient evidence | none |
| `03.02.120.pdf` | `03.02.120` | insufficient evidence | none |
| `04A-B01.pdf` | `04A-B01` | supported | docling, evidence_candidates, native |
| `04J-AP-T01.pdf` | `04J-AP-T01` | insufficient evidence | none |
| `04J-AS-T01.pdf` | `04J-AS-T01` | insufficient evidence | none |
| `05.02.120.pdf` | `05.02.120` | insufficient evidence | none |
| `05.02.80A-05.01.71A.pdf` | `05.02.80A-05.01.71A` | insufficient evidence | none |
| `05.02.81A-05.01.72A.pdf` | `05.02.81A-05.01.72A` | insufficient evidence | none |
| `06.02.120.pdf` | `06.02.120` | insufficient evidence | none |
| `06513.0-00.pdf` | `06513.0-00` | supported | docling, evidence_candidates, native, tesseract, vision |
| `06520.0-00.pdf` | `06520.0-00` | supported | docling, evidence_candidates, native, tesseract, vision |
| `07.02.120.pdf` | `07.02.120` | insufficient evidence | none |
| `08.02.120.pdf` | `08.02.120` | insufficient evidence | none |
| `09.02.120.pdf` | `09.02.120` | insufficient evidence | none |
| `1-1414631-0.pdf` | `1-1414631-0` | supported | docling, evidence_candidates, native, tesseract, vision |
| `1-2186527-1.pdf` | `1-2186527-1` | supported | evidence_candidates, native, tesseract |
| `1-2186528-1.pdf` | `1-2186528-1` | supported | evidence_candidates, native, tesseract |
| `1-2186576-1.pdf` | `1-2186576-1` | supported | docling, evidence_candidates, native |
| `1-2186577-1.pdf` | `1-2186577-1` | supported | docling, evidence_candidates, native |
| `1-350777-1.pdf` | `1-350777-1` | insufficient evidence | evidence_candidates |
| `1-480700-0.pdf` | `1-480700-0` | supported | tesseract |
| `1-794616-2.pdf` | `1-794616-2` | insufficient evidence | docling, evidence_candidates, native |
| `10-AT3.pdf` | `10-AT3` | insufficient evidence | none |
| `10-E1126TA035M7-L859F73Z.pdf` | `10-E1126TA035M7-L859F73Z` | supported | docling, evidence_candidates, native, tesseract, vision |
| `10-MXL.pdf` | `10-MXL` | insufficient evidence | native |
| `10-T5.pdf` | `10-T5` | insufficient evidence | none |
| `10.02.120.pdf` | `10.02.120` | insufficient evidence | none |
| `100-AT10.pdf` | `100-AT10` | insufficient evidence | none |
| `100-AT5.pdf` | `100-AT5` | insufficient evidence | none |
| `100-ATN10K6.pdf` | `100-ATN10K6` | insufficient evidence | none |
| `100-T10.pdf` | `100-T10` | insufficient evidence | none |
| `100-T20.pdf` | `100-T20` | insufficient evidence | tesseract |
| `100818033524001.pdf` | `100818033524001` | contradicted | evidence_candidates, native, tesseract, vision |
| `100_AT3_900_SFX.pdf` | `100_AT3_900_SFX` | insufficient evidence | none |
| `100_T5_990_BFX.pdf` | `100_T5_990_BFX` | insufficient evidence | none |
| `101.6-XL.pdf` | `101.6-XL` | insufficient evidence | none |

## Interpretation

This framing materially outperforms identity discovery when judged by the review burden: it converts the task from selecting one correct identifier among many PDF strings into checking one known upstream claim. The exact gain is the supported count above; it should not be promoted to production without adjudicating the supported cases for false positives, especially multi-product sheets.

Recommendation: preserve the upstream SKU as an input contract and add a separate claim-verification path. Do not replace the existing identity-discovery path universally until the supported cases are human-audited and the upstream manifest contract is formalized.
