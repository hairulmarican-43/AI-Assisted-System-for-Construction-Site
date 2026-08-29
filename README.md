# Work-at-Height Compliance Screening | AI-Assisted-System for Construction Site

Preliminary visual screening of site photographs and video walkthroughs against
Singapore work-at-height standards, using Claude's vision capability.

| Category | Standard |
|---|---|
| Scaffolds | SS 659:2020 Code of practice for scaffolds |
| Fall-arrest PPE | SS 528:2006 Personal fall-arrest systems (Parts 1–6) |
| Anchors & lifelines | SS 570-1 & 570-2:2022 Anchor devices and horizontal lifeline systems |

Findings are scored on the **WSH Risk Assessment matrix**: severity 1–5,
likelihood 1–5, RPN = S × L, banded Low / Medium / High.

## Run

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Before you use it for real

`checks.yaml` ships with `TODO` placeholders in every `clause_ref` field.
Fill these in from your own licensed copies of the standards. Clause *numbers*
and your own paraphrase are fine to commit; **clause text is not** — the
standards are copyrighted by Enterprise Singapore and this repo must not
redistribute them.

## Design notes

**Bounding boxes are real.** The model returns normalised `[x0,y0,x1,y1]`
coordinates and the app draws exactly those. Boxes that are missing, malformed,
or inverted are rejected rather than approximated, and the element is listed as
"could not be localised". Low-confidence boxes are drawn thinner and marked `~`.

**Structured output, not prose parsing.** The model returns JSON. Findings are
validated against `checks.yaml`, so invented check IDs are dropped and severity
values outside 1–5 are clamped.

**"Not visible" is a first-class verdict.** The prompt instructs the model never
to mark something compliant merely because no defect is visible. Unassessable
items are reported separately so the inspector knows what still needs checking
on foot.

## Limitations

Single-frame visual screening cannot assess load capacity, tie adequacy,
foundation bearing, equipment inspection history, or anything behind the camera.
Absence of a finding is not evidence of compliance. This tool does not discharge
any duty under the WSH Act and is not a substitute for examination by a
competent person or, for scaffolds, a Scaffold Supervisor.

