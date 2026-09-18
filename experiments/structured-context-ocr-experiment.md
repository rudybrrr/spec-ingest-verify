# Structured local-context OCR experiment

OCR-only experiment for `part_number`; the frozen Vision result is reference-only and is consulted only for comparison after extraction. The prior bounded result is also comparison-only.

## Aggregate

- PDFs tested: 20
- Anchors: `{'no_anchor': 12, 'anchor_found': 8}`
- Improvement over full-page OCR: 1
- Improvement over prior bounded experiment: 2
- Regressions vs full-page OCR: 1
- Regressions vs prior bounded experiment: 1
- Still ambiguous: 5

## Per-PDF results

| PDF | anchors | full-page OCR | structured context | prior bounded | status |
|---|---|---|---|---|---|
| `00551106.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `00701101.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `00741200.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `01.02.120.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `04A-B01.pdf` | part_number (p1), part_number (p1) | `None` | `04A.` | `None` | anchor_found |
| `04J-AP-T01.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `05.02.80A-05.01.71A.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `06513.0-00.pdf` | manufacturer_part_number (p1), part_number (p1) | `06513.0-00` | `06513.0-00` | `None` | anchor_found |
| `06524.0-00.pdf` | manufacturer_part_number (p1), part_number (p1) | `06524.0-00` | `06524.0-00` | `None` | anchor_found |
| `1-2186527-1.pdf` | part_number (p3), part_number (p5), part_number (p5), part_number (p5), part_number (p5) | `1-2186527-1` | `None` | `None` | anchor_found |
| `1-2186576-1.pdf` | part_number (p2) | `None` | `None` | `None` | anchor_found |
| `1-350777-1.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `1-480700-0.pdf` | part_number (p1), part_number (p2) | `None` | `None` | `None` | anchor_found |
| `1-794616-2.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `10-AT3.pdf` | part_number (p2) | `None` | `None` | `None` | anchor_found |
| `10-E1126TA035M7-L859F73Z.pdf` | no_anchor | `10-E1126TA035M7-L859F73Z` | `None` | `None` | no_anchor |
| `10-MXL.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `100_AT3_900_SFX.pdf` | part_number (p7), part_number (p167), part_number (p168), part_number (p169), part_number (p170), part_number (p171), part_number (p172), part_number (p173), part_number (p174), part_number (p175), part_number (p176), part_number (p177), part_number (p178), part_number (p179) | `None` | `None` | `None` | anchor_found |
| `100-ATN10K6.pdf` | no_anchor | `None` | `None` | `None` | no_anchor |
| `101DOR010-S10.pdf` | no_anchor | `101DORO010-S10` | `None` | `None` | no_anchor |

## Representative examples

- `04A-B01.pdf`: anchor(s) found; full-page `None`, structured `04A.`, prior bounded `None`, reference `None`.
- `06513.0-00.pdf`: anchor(s) found; full-page `06513.0-00`, structured `06513.0-00`, prior bounded `None`, reference `06513.0-00`.
- `06524.0-00.pdf`: anchor(s) found; full-page `06524.0-00`, structured `06524.0-00`, prior bounded `None`, reference `06524.0-00`.

## Interpretation

On this cohort, structured local context materially outperforms simple bounded extraction by the reference-only comparison: 2 improvement(s) versus 1 regression(s). There are 5 genuinely ambiguous anchor-found cases and 12 explicit `no_anchor` cases. Recommendation: do not integrate based on this cohort. Inspect `tsv-evidence.json`, `anchors.json`, and `structured-prompt.txt` under the run artifacts for exact coordinate evidence. No production code or Vision behavior is changed by this experiment.
