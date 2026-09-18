# Anchor-based bounded OCR experiment

## Scope and method

Tested 47 human-review PDFs in the frozen 53/47 evaluation snapshot; the representative subset contains 20 PDFs.
The bounded path uses Tesseract TSV geometry and semantic anchors. `anchor_found` calls the bounded prompt only; `no_anchor` is recorded without bounded extraction. Existing full-page OCR is the independent baseline. Filenames are not extraction evidence.

## Aggregate results

- Anchor status: part_number `{'no_anchor': 12, 'anchor_found': 8}`; description `{'anchor_found': 11, 'no_anchor': 9}`.
- Would align with the existing vision result: part_number `2`; description `0`.
- Bounded worse than a baseline that already aligned with vision: part_number `0`; description `3`.
- Layout signals covered: 15 page-size patterns and 5 anchor-label sets.

## Per-document results

`vision_reference` is used only as the existing comparison target for estimating whether a previous human-review mismatch would become aligned; it is not changed or treated as newly verified ground truth.

| PDF | anchors | part number: full-page -> bounded | description: full-page -> bounded |
|---|---|---|---|
| `00551106.pdf` | description | `None` -> `None` (no_anchor) | `Verdin iIMX8M Mini Datasheet` -> `None` (anchor_found) |
| `00701101.pdf` | description | `None` -> `None` (no_anchor) | `Verdin iMX8M Plus V1.1 HW Datasheet` -> `description` (anchor_found) |
| `00741200.pdf` | description | `None` -> `None` (no_anchor) | `Verdin System on Module (SoM) family based on the Texas Instruments AM62x Sitara family of System on Chips (SoCs)` -> `description.` (anchor_found) |
| `01.02.120.pdf` | none | `None` -> `None` (no_anchor) | `None` -> `None` (no_anchor) |
| `04A-B01.pdf` | description, part_number | `None` -> `04A` (anchor_found) | `None` -> `04A Basic Switch 04A-KO1 Switch with Arrowhead Keycap 04A-KO2 Switch with Keycap Chi` (anchor_found) |
| `04J-AP-T01.pdf` | description | `None` -> `None` (no_anchor) | `Series 04J is a single pole, 4-way toggle switch with a pushbutton in the center position.` -> `DESCRIPTION
FEATURES` (anchor_found) |
| `05.02.80A-05.01.71A.pdf` | none | `None` -> `None` (no_anchor) | `dPOFLEX Industrial Peristaltic Pump` -> `None` (no_anchor) |
| `06513.0-00.pdf` | manufacturer_part_number, part_number | `06513.0-00` -> `06513.0-00` (anchor_found) | `Stego enclosure heater, 100OW heating capacity, touch-safe PTC resistor heating element, 120-240 VAC/VDC operating voltage, 50/60 Hz, fixed thermostat, N.C. (open on rise) 41 °F (5 °C) switch- on, 35mm DIN rail mount.` -> `None` (no_anchor) |
| `06524.0-00.pdf` | manufacturer_part_number, part_number | `06524.0-00` -> `06524.0-00` (anchor_found) | `Stego enclosure heater, 150W heating capacity, touch-safe PTC resistor heating element, 120-240 VAC/VDC operating voltage` -> `None` (no_anchor) |
| `1-2186527-1.pdf` | description, part_number | `1-2186527-1` -> `PART NUMBER` (anchor_found) | `BUDGET THERMAL TRANSFER PRINTER (300dpi)` -> `DESCRIPTION
PART NUMBER` (anchor_found) |
| `1-2186576-1.pdf` | description, part_number | `None` -> `T7112DS` (anchor_found) | `High quality, 300 dpi double side thermal transfer printer.` -> `T7112DS-PRINTER` (anchor_found) |
| `1-350777-1.pdf` | description | `None` -> `None` (no_anchor) | `PLUG, UNIVERSAL MATE-N-LOK(TM)` -> `DESCRIPTION` (anchor_found) |
| `1-480700-0.pdf` | description, part_number | `None` -> `PLUG, 3 CIRCUIT` (anchor_found) | `PLUG, 3 CIRCUIT, UNIVERSAL MATE-N-LOCK` -> `None` (anchor_found) |
| `1-794616-2.pdf` | description | `None` -> `None` (no_anchor) | `PLUG HOUSING, 2 to 24 POSITION DUAL ROW, FREE HANGING, MICRO_MATE-N-LOK(TM)` -> `DESCRIPTION` (anchor_found) |
| `10-AT3.pdf` | part_number | `None` -> `None` (anchor_found) | `Polyurethane Timing Belt` -> `None` (no_anchor) |
| `10-E1126TA035M7-L859F73Z.pdf` | none | `10-E1126TA035M7-L859F73Z` -> `None` (no_anchor) | `flow E1 12 mm housing` -> `None` (no_anchor) |
| `10-MXL.pdf` | none | `None` -> `None` (no_anchor) | `None` -> `None` (no_anchor) |
| `100_AT3_900_SFX.pdf` | description, part_number | `None` -> `Charts` (anchor_found) | `High Precision Drive Components The World Leader in Polyurethane Timing Belts High Performance POLYURETHANE TIMING BELTS AND PULLEYS` -> `None` (anchor_found) |
| `100-ATN10K6.pdf` | none | `None` -> `None` (no_anchor) | `None` -> `None` (no_anchor) |
| `101DOR010-S10.pdf` | none | `101DORO010-S10` -> `None` (no_anchor) | `O-Ring NBR, DN 10 ISO-KF` -> `None` (no_anchor) |

## Interpretation

Cases marked `no_anchor` remain unresolved by the bounded approach and must rely on the separately reported full-page path or human review. Cases where bounded OCR disagrees with the existing vision result are not counted as aligned. Inspect exact TSV rows and anchor coordinates under `.runs/anchor-ocr-experiment/runs/<run_id>/` before considering integration.

Recommendation: do not integrate into the main OCR branch yet. In this sample, anchors were found for 8/20 part-number fields and 11/20 description fields, but only 2 part-number fields became aligned and 3 description cases regressed relative to a baseline that already aligned with the existing vision result. The bounded-region rules need refinement and a larger review set before production integration. This report intentionally does not alter the production OCR branch.
