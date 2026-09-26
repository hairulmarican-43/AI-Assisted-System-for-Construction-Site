# Work-at-Height Compliance Screening

Screens site photographs and video walkthroughs against three Singapore
work-at-height standards **in a single pass**, using Claude's vision capability.

| Standard | Covers |
|---|---|
| SS 659:2020 | Scaffolds |
| SS 528:2006 (Parts 1–6) | Personal fall-arrest systems |
| SS 570-1 & 570-2:2022 | Anchor devices and horizontal lifeline systems |

There is no category selector. Every image is assessed against all three
standards, and only the categories actually present in the photo are reported.

## Batch upload

Drag in as many photographs as you like. Every image is decoded before any
billable request is made, so unreadable files are reported up front rather than
mid-run. Images already screened this session are detected by content hash and
skipped, so a browser refresh does not re-bill you. If one image fails, the rest
of the batch continues, and results are shown worst-first by RPN.

## Output

- **Live screening** — annotated image with the model's own bounding boxes, findings ordered by risk
- **Collated statistics** — cross-standard totals, severity distribution, most frequent non-compliances, open items needing on-site verification
- **Word report** (`.docx`) — scope, risk basis, statistics, priority actions, per-image detail with embedded annotated photos, and a sign-off block
- **CSV export** — one row per finding, ready for a spreadsheet or RA form

Findings are scored on the **WSH Risk Assessment matrix**: severity 1–5,
likelihood 1–5, RPN = S × L, banded Low (1–3) / Medium (4–12) / High (15–25).

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

On Streamlit Cloud, add the key under **Settings → Secrets** instead:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, prompt construction, parsing, statistics |
| `report.py` | Word report generation (python-docx) |
| `checks.yaml` | Check definitions, severity baselines, clause references |

## Before you use it for real

`checks.yaml` ships with `TODO` placeholders in every `clause_ref` field.
Fill these in from your own licensed copies of the standards. Clause *numbers*
and your own paraphrase are fine to commit; **clause text is not** — the
standards are copyrighted by Enterprise Singapore and this repo must not
redistribute them.

Adding, removing, or reweighting a check means editing `checks.yaml` only. The
prompt, the parser, the statistics and the report all build themselves from
that file.

## Design notes

**Bounding boxes are real.** The model returns normalised `[x0,y0,x1,y1]`
coordinates and the app draws exactly those. Boxes that are missing, malformed,
or inverted are rejected rather than approximated, and the element is listed as
"could not be localised". Low-confidence boxes are drawn thinner and marked `~`.

**Structured output with truncation recovery.** The model returns JSON validated
against `checks.yaml`, so invented check IDs are dropped and severity values
outside 1–5 are clamped. If a response is cut off mid-object, the parser
recovers every element that closed cleanly and flags the screening as
incomplete rather than losing everything.

**"Not visible" is a first-class verdict.** The prompt instructs the model never
to mark something compliant merely because no defect is visible. Unassessable
items are counted separately and listed in the report as open items requiring
verification on foot.

## Limitations

Single-frame visual screening cannot assess load capacity, tie adequacy,
foundation bearing, equipment inspection history, or anything behind the camera.
Absence of a finding is not evidence of compliance. This tool does not discharge
any duty under the WSH Act and is not a substitute for examination by a
competent person or, for scaffolds, an approved Scaffold Supervisor.
