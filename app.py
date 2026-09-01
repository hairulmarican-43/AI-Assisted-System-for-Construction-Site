"""
Work-at-Height Compliance Screening
===================================
Screens site photographs and video walkthroughs against SS 659 (scaffolds),
SS 528 (personal fall-arrest systems) and SS 570 (anchor devices and horizontal
lifeline systems) in a single pass.

Findings are scored on the WSH Council Risk Assessment scale (severity 1-5 x
likelihood 1-5), collated across all three standards, and exported as a Word
report.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

from report import build_report

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Phones and WhatsApp frequently produce HEIC while naming the file .jpeg.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "checks.yaml"
MAX_IMAGE_EDGE = 1568

# All three standards in one response means more output. Too low a ceiling
# truncates the JSON mid-object.
MAX_OUTPUT_TOKENS = 16000

MODELS = {
    "Claude Sonnet 5 (recommended)": "claude-sonnet-5",
    "Claude Opus 5 (highest accuracy, higher cost)": "claude-opus-5",
    "Claude Haiku 4.5 (fastest, lowest cost)": "claude-haiku-4-5-20251001",
}

APPROX_COST_PER_FRAME_USD = {
    "claude-sonnet-5": 0.03,
    "claude-opus-5": 0.14,
    "claude-haiku-4-5-20251001": 0.008,
}

STATUS_COLOURS = {
    "compliant":     ((0, 170, 70, 45),    (0, 140, 55, 240)),
    "non_compliant": ((210, 30, 30, 55),   (170, 20, 20, 245)),
    "not_visible":   ((130, 130, 130, 30), (110, 110, 110, 200)),
}

RISK_COLOURS = {"Low": "#0f9d58", "Medium": "#f4a300", "High": "#c62828"}

STANDARD_SHORT = {
    "scaffold": "SS 659",
    "fall_arrest_ppe": "SS 528",
    "anchor_lifeline": "SS 570",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def risk_band(rpn: int, bands: list) -> tuple[str, str]:
    for band in bands:
        if rpn <= band["max"]:
            return band["label"], band["action"]
    return bands[-1]["label"], bands[-1]["action"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    category: str
    check_id: str
    check_title: str
    clause_ref: str
    status: str
    observation: str
    severity: int
    likelihood: int
    severity_rationale: str = ""
    bands: list = field(default_factory=list, repr=False)

    @property
    def rpn(self) -> int:
        if self.status != "non_compliant":
            return 0
        return self.severity * self.likelihood

    @property
    def band(self) -> str:
        return risk_band(self.rpn, self.bands)[0] if self.rpn else "-"

    @property
    def standard_short(self) -> str:
        return STANDARD_SHORT.get(self.category, self.category)


@dataclass
class Element:
    element_id: str
    category: str
    label: str
    bbox: tuple[float, float, float, float] | None
    bbox_confidence: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(f.status == "non_compliant" for f in self.findings):
            return "non_compliant"
        if self.findings and all(f.status == "not_visible"
                                 for f in self.findings):
            return "not_visible"
        return "compliant"

    @property
    def max_rpn(self) -> int:
        return max((f.rpn for f in self.findings), default=0)

    @property
    def max_severity(self) -> int:
        vals = [f.severity for f in self.findings
                if f.status == "non_compliant"]
        return max(vals, default=0)

    @property
    def standard_short(self) -> str:
        return STANDARD_SHORT.get(self.category, self.category)


@dataclass
class Assessment:
    elements: list[Element]
    summary: str
    scene_notes: str = ""
    raw: str = ""
    truncated: bool = False

    @property
    def max_rpn(self) -> int:
        return max((e.max_rpn for e in self.elements), default=0)


@dataclass
class ScreeningRecord:
    source: str
    timestamp: datetime
    assessment: Assessment
    annotated_png: bytes | None = None


# ---------------------------------------------------------------------------
# Prompt — all three standards in one pass
# ---------------------------------------------------------------------------

def build_prompt(config: dict) -> str:
    blocks = []
    for key, cat in config["categories"].items():
        checks = "\n".join(
            f'  - check_id "{c["id"]}" — {c["title"]}\n'
            f'    Look for: {c["look_for"].strip()}\n'
            f'    Baseline severity if breached: {c["base_severity"]}'
            for c in cat["checks"]
        )
        blocks.append(
            f'### category "{key}" — {cat["standard"]}\n'
            f'Element type: {cat["element_name"]} '
            f'(id prefix {cat["element_prefix"]})\n'
            f'Scope: {cat["scope"].strip()}\n'
            f'Checks:\n{checks}'
        )
    categories_block = "\n\n".join(blocks)

    sev = "\n".join(f"  {k} = {v}"
                    for k, v in config["severity_scale"].items())
    lik = "\n".join(f"  {k} = {v}"
                    for k, v in config["likelihood_scale"].items())

    return f"""You are assisting a workplace safety officer with a preliminary visual screening of a work-at-height site photograph in Singapore.

Assess the image against all three standards below in a single pass. A photograph may contain elements from one, two, or all three categories, or none at all.

{categories_block}

## Step 1 — Identify elements

Work through each category and identify the elements of that type that are actually visible. Assign each element the correct "category" value from the three above. Only report categories that are genuinely present — if there is no scaffold in the image, return no scaffold elements. Do not invent elements to fill out the response.

If nothing relevant is visible at all, return an empty elements list and explain why in scene_notes.

## Step 2 — Assess each element

For each element, return a finding for every check_id belonging to that element's category, and only those. Do not mix checks across categories.

Use status "not_visible" when the image does not show enough to judge. This is expected and preferable to guessing. Never mark something compliant merely because you cannot see a defect.

## Step 3 — Rate severity and likelihood

Severity — the worst credible outcome if this breach leads to an incident:
{sev}

Likelihood — how probable that incident is given what you can see:
{lik}

Start from the baseline severity for each check. You may adjust by one point based on what you observe, and if you do, say why briefly. For compliant or not_visible findings, still give your severity estimate but set likelihood to 1.

## Step 4 — Bounding boxes

Give each element a bounding box as [x0, y0, x1, y1] in normalised coordinates from 0.0 to 1.0, origin at top-left. The box must tightly enclose the element. Set bbox_confidence to "high", "medium" or "low". If you cannot localise the element, set bbox to null rather than guessing — a wrong box on a safety report is worse than no box.

## Output format

Respond with ONLY a JSON object. No markdown fences, no preamble, no commentary.

Keep it compact. Each observation must be a single sentence under 30 words. Include severity_rationale only when you adjusted the baseline, and keep it under 20 words. A busy frame may contain many elements across three standards, and an over-long response will be cut off before it is complete.

{{
  "scene_notes": "one or two sentences on what the image shows and anything limiting the assessment",
  "elements": [
    {{
      "element_id": "SC-1",
      "category": "scaffold",
      "label": "short descriptor",
      "bbox": [0.12, 0.05, 0.48, 0.91],
      "bbox_confidence": "high",
      "findings": [
        {{
          "check_id": "one of the ids for this element's category",
          "status": "compliant | non_compliant | not_visible",
          "observation": "one sentence, under 30 words",
          "severity": 1,
          "likelihood": 1,
          "severity_rationale": "only if adjusted"
        }}
      ]
    }}
  ],
  "summary": "two or three sentences covering the most serious findings across all standards and what needs attention first"
}}
"""


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def sniff_format(head: bytes) -> str:
    if head[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[4:8] == b"ftyp":
        if head[8:12] in (b"heic", b"heix", b"hevc", b"heim", b"heis",
                          b"mif1", b"msf1", b"avif", b"avis"):
            return "HEIC/AVIF"
        return "MP4/MOV video"
    if head[:4] == b"GIF8":
        return "GIF"
    if head[:2] == b"BM":
        return "BMP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return "unknown"


def load_uploaded_image(uploaded) -> Image.Image:
    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(0)

    if not data:
        raise ValueError(
            "The uploaded file is empty (0 bytes). The transfer was probably "
            "interrupted — try uploading it again."
        )

    detected = sniff_format(data[:16])

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except UnidentifiedImageError:
        if detected == "HEIC/AVIF":
            raise ValueError(
                f"This file is HEIC/HEIF, not a JPEG, despite being named "
                f"'{uploaded.name}'. Install HEIC support with "
                "`pip install pillow-heif`, or re-save the photo as JPEG."
            ) from None
        if detected.endswith("video"):
            raise ValueError(
                "This is a video file, not a photograph. Switch to "
                "'Video walkthrough' in the sidebar."
            ) from None
        raise ValueError(
            f"Could not decode this file. It is named '{uploaded.name}' but "
            f"its contents look like: {detected}. Try re-saving it as JPEG."
        ) from None
    except OSError as exc:
        raise ValueError(
            f"The image file appears to be truncated or damaged ({exc}). "
            "Try uploading it again."
        ) from None

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    return image.convert("RGB")


def file_key(uploaded) -> str:
    """
    Stable identity for an uploaded file, used to avoid re-billing for an
    image that has already been screened this session. Streamlit re-sends
    every uploaded file on each rerun, so name alone is not enough — two
    different photos could share a name, and the same photo keeps its
    contents.
    """
    uploaded.seek(0)
    data = uploaded.read()
    uploaded.seek(0)
    return f"{uploaded.name}:{len(data)}:{hashlib.md5(data).hexdigest()[:12]}"


def downscale(image: Image.Image,
              max_edge: int = MAX_IMAGE_EDGE) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    scale = max_edge / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def call_claude(api_key: str, model: str, image: Image.Image,
                prompt: str) -> tuple[str, str]:
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "The 'anthropic' package is not installed. "
            "Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg",
                            "data": encode_image(downscale(image))}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(b.text for b in message.content if b.type == "text")
    return text, (message.stop_reason or "")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _repair_truncated_json(text: str) -> dict | None:
    """Recover complete elements from a response cut off mid-object."""
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]

    depth = 0
    in_string = False
    escaped = False
    last_element_end = -1

    for i, ch in enumerate(s):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            # root object = 1, elements array = 2, element object = 3.
            if depth == 2 and ch == "}":
                last_element_end = i

    if last_element_end == -1:
        return None

    try:
        data = json.loads(s[:last_element_end + 1] + "]}")
    except json.JSONDecodeError:
        return None

    data["_truncated"] = True
    return data


def _extract_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    repaired = _repair_truncated_json(cleaned)
    if repaired is not None:
        return repaired

    raise ValueError(
        "The model response was not valid JSON and could not be repaired."
    )


def _clamp_bbox(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v < -0.05 or v > 1.05 for v in vals):
        return None
    x0, y0, x1, y1 = [min(max(v, 0.0), 1.0) for v in vals]
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def parse_assessment(text: str, config: dict) -> Assessment:
    data = _extract_json(text)
    bands = config["risk_bands"]

    # check_id -> (category, meta). Ids are unique across categories, so a
    # mislabelled element category can be corrected from the checks it used.
    lookup = {}
    for cat_key, cat in config["categories"].items():
        for chk in cat["checks"]:
            lookup[chk["id"]] = (cat_key, chk)

    elements: list[Element] = []
    for i, raw_elem in enumerate(data.get("elements", []), start=1):
        category = str(raw_elem.get("category", "")).strip()
        if category not in config["categories"]:
            category = ""

        findings: list[Finding] = []
        for raw_find in raw_elem.get("findings", []):
            cid = raw_find.get("check_id", "")
            if cid not in lookup:
                continue                        # invented check id
            true_cat, meta = lookup[cid]
            if not category:
                category = true_cat             # infer from checks used

            status = raw_find.get("status", "not_visible")
            if status not in ("compliant", "non_compliant", "not_visible"):
                status = "not_visible"

            def _rating(key, default):
                try:
                    return min(max(int(raw_find.get(key, default)), 1), 5)
                except (TypeError, ValueError):
                    return default

            findings.append(Finding(
                category=true_cat,
                check_id=cid,
                check_title=meta["title"],
                clause_ref=meta.get("clause_ref", ""),
                status=status,
                observation=str(raw_find.get("observation", "")).strip(),
                severity=_rating("severity", meta["base_severity"]),
                likelihood=_rating("likelihood", 1),
                severity_rationale=str(
                    raw_find.get("severity_rationale", "")).strip(),
                bands=bands,
            ))

        if not category:
            category = next(iter(config["categories"]))

        prefix = config["categories"][category]["element_prefix"]
        elements.append(Element(
            element_id=str(raw_elem.get("element_id", f"{prefix}-{i}")),
            category=category,
            label=str(raw_elem.get("label", f"Element {i}")),
            bbox=_clamp_bbox(raw_elem.get("bbox")),
            bbox_confidence=str(raw_elem.get("bbox_confidence",
                                             "low")).lower(),
            findings=findings,
        ))

    return Assessment(
        elements=elements,
        summary=str(data.get("summary", "")).strip(),
        scene_notes=str(data.get("scene_notes", "")).strip(),
        raw=text,
        truncated=bool(data.get("_truncated", False)),
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def _font(width: int):
    size = max(14, width // 55)
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate(image: Image.Image,
             assessment: Assessment) -> tuple[Image.Image, int]:
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _font(w)
    unlocated = 0

    for elem in assessment.elements:
        if elem.bbox is None:
            unlocated += 1
            continue

        x0, y0 = int(elem.bbox[0] * w), int(elem.bbox[1] * h)
        x1, y1 = int(elem.bbox[2] * w), int(elem.bbox[3] * h)
        fill, border = STATUS_COLOURS.get(elem.status,
                                          STATUS_COLOURS["not_visible"])
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=border,
                       width=4 if elem.bbox_confidence == "high" else 2)

        sev = elem.max_severity
        marker = "" if elem.bbox_confidence == "high" else " ~"
        label = (f"{elem.element_id}{marker} · {elem.standard_short} · "
                 f"{'S' + str(sev) if sev else 'OK'}")

        try:
            box = draw.textbbox((0, 0), label, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:
            tw, th = len(label) * 8, 16

        pad = 6
        ly0 = max(0, y0 - th - pad * 2)
        draw.rectangle(
            [x0, ly0, min(w, x0 + tw + pad * 2), ly0 + th + pad * 2],
            fill=border[:3] + (225,))
        draw.text((x0 + pad, ly0 + pad), label,
                  fill=(255, 255, 255), font=font)

    composed = Image.alpha_composite(image.convert("RGBA"), overlay)
    return composed.convert("RGB"), unlocated


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, interval_s: float, max_frames: int):
    if not CV2_AVAILABLE:
        return []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 25.0
    step = max(1, int(round(fps * interval_s)))
    frames, idx = [], 0
    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append((
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                    idx / fps,
                ))
            idx += 1
    finally:
        cap.release()
    return frames


# ---------------------------------------------------------------------------
# Statistics — collated across all three standards
# ---------------------------------------------------------------------------

def compute_statistics(records: list[ScreeningRecord], config: dict) -> dict:
    bands = config["risk_bands"]
    cats = config["categories"]

    per_cat = {
        k: {"key": k, "standard": cats[k]["standard"],
            "short": STANDARD_SHORT.get(k, k), "elements": 0, "breaches": 0,
            "High": 0, "Medium": 0, "Low": 0,
            "peak_severity": 0, "peak_rpn": 0}
        for k in cats
    }

    severity_dist = Counter()
    band_dist = Counter()
    check_counter = Counter()
    check_peak = defaultdict(int)
    check_meta = {}
    nv_counter = Counter()
    all_breaches = []
    element_count = 0
    nv_count = 0

    for rec in records:
        for elem in rec.assessment.elements:
            element_count += 1
            if elem.category in per_cat:
                per_cat[elem.category]["elements"] += 1

            for find in elem.findings:
                key = (find.category, find.check_id)
                check_meta[key] = (find.check_title, find.standard_short)

                if find.status == "not_visible":
                    nv_count += 1
                    nv_counter[key] += 1
                    continue
                if find.status != "non_compliant":
                    continue

                band_label, _ = risk_band(find.rpn, bands)
                severity_dist[find.severity] += 1
                band_dist[band_label] += 1
                check_counter[key] += 1
                check_peak[key] = max(check_peak[key], find.rpn)

                cat = per_cat.get(find.category)
                if cat:
                    cat["breaches"] += 1
                    cat[band_label] = cat.get(band_label, 0) + 1
                    cat["peak_severity"] = max(cat["peak_severity"],
                                               find.severity)
                    cat["peak_rpn"] = max(cat["peak_rpn"], find.rpn)

                all_breaches.append({
                    "band": band_label,
                    "rpn": find.rpn,
                    "severity": find.severity,
                    "likelihood": find.likelihood,
                    "source": rec.source,
                    "element": elem.element_id,
                    "standard": find.standard_short,
                    "check": find.check_title,
                    "observation": find.observation,
                    "clause": find.clause_ref,
                })

    all_breaches.sort(key=lambda b: -b["rpn"])

    top_checks = [
        {"check": check_meta[k][0], "standard": check_meta[k][1],
         "count": c, "peak_rpn": check_peak[k]}
        for k, c in check_counter.most_common(10)
    ]
    nv_by_check = [
        {"check": check_meta[k][0], "standard": check_meta[k][1], "count": c}
        for k, c in nv_counter.most_common(10)
    ]

    return {
        "image_count": len(records),
        "element_count": element_count,
        "breach_count": len(all_breaches),
        "not_visible_count": nv_count,
        "by_category": list(per_cat.values()),
        "severity_distribution": dict(severity_dist),
        "band_distribution": dict(band_dist),
        "top_checks": top_checks,
        "not_visible_by_check": nv_by_check,
        "all_breaches": all_breaches,
        "peak_rpn": all_breaches[0]["rpn"] if all_breaches else 0,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_assessment(image: Image.Image, assessment: Assessment,
                      config: dict) -> bytes | None:
    bands = config["risk_bands"]
    annotated_png = None

    if assessment.truncated:
        st.warning(
            "The analysis was cut off before it finished. The elements below "
            "were recovered, but this image may contain further elements that "
            "were never reported. Treat this screening as incomplete.",
            icon="⚠️",
        )

    if assessment.elements:
        annotated, unlocated = annotate(image, assessment)
        st.image(annotated, width="stretch",
                 caption="Boxes are the model's own localisation. "
                         "A tilde (~) marks lower-confidence placement.")
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        annotated_png = buf.getvalue()
        if unlocated:
            st.caption(
                f"{unlocated} element(s) were assessed but could not be "
                "localised, so no box is drawn for them."
            )
    else:
        st.image(image, width="stretch")
        st.info("No relevant elements were identified in this image.")

    if assessment.scene_notes:
        st.caption(assessment.scene_notes)

    rpn = assessment.max_rpn
    label, action = (risk_band(rpn, bands) if rpn
                     else ("None", "No breach identified."))

    if rpn == 0:
        st.success("No non-compliance identified in this image.")
    else:
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:8px;"
            f"background:{RISK_COLOURS.get(label, '#666')};color:white;"
            f"font-size:1.05rem;'><strong>Highest risk: {label}</strong>"
            f" &nbsp;·&nbsp; RPN {rpn} &nbsp;·&nbsp; {action}</div>",
            unsafe_allow_html=True,
        )

    if assessment.summary:
        st.write("")
        st.write(assessment.summary)

    # Which standards were actually engaged by this image
    by_cat = defaultdict(list)
    for elem in assessment.elements:
        by_cat[elem.category].append(elem)
    if by_cat:
        st.caption(" · ".join(
            f"{config['categories'][k]['label']}: {len(v)}"
            for k, v in by_cat.items()
        ))

    breaches = sorted(
        [(e, f) for e in assessment.elements for f in e.findings
         if f.status == "non_compliant"],
        key=lambda pair: -pair[1].rpn,
    )

    if breaches:
        st.subheader("Non-compliances")
        for elem, find in breaches:
            band_label, _ = risk_band(find.rpn, bands)
            with st.expander(
                f"{band_label} · RPN {find.rpn} · {find.standard_short} · "
                f"{elem.element_id} — {find.check_title}",
                expanded=(band_label == "High"),
            ):
                st.write(find.observation)
                c1, c2, c3 = st.columns(3)
                c1.metric("Severity", find.severity)
                c2.metric("Likelihood", find.likelihood)
                c3.metric("RPN", find.rpn)
                if find.severity_rationale:
                    st.caption(f"Severity note: {find.severity_rationale}")
                st.caption(f"Reference: {find.clause_ref}")

    unknowns = [(e, f) for e in assessment.elements for f in e.findings
                if f.status == "not_visible"]
    if unknowns:
        with st.expander(
                f"Could not be assessed from this image ({len(unknowns)})"):
            for elem, find in unknowns:
                st.write(f"**{elem.element_id} · {find.standard_short} — "
                         f"{find.check_title}**")
                if find.observation:
                    st.caption(find.observation)

    with st.expander("Raw model response"):
        st.code(assessment.raw, language="json")

    return annotated_png


def register_to_csv(records: list[ScreeningRecord], config: dict) -> str:
    bands = config["risk_bands"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "screened_at", "source", "standard", "element_id", "element_label",
        "check", "clause_ref", "status", "observation", "severity",
        "likelihood", "rpn", "risk_band", "action",
    ])
    for rec in records:
        for elem in rec.assessment.elements:
            for find in elem.findings:
                band_label, action = (risk_band(find.rpn, bands)
                                      if find.rpn else ("-", "-"))
                writer.writerow([
                    rec.timestamp.isoformat(timespec="seconds"), rec.source,
                    find.standard_short, elem.element_id, elem.label,
                    find.check_title, find.clause_ref, find.status,
                    find.observation, find.severity, find.likelihood,
                    find.rpn, band_label, action,
                ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Work-at-Height Compliance Screening",
                   page_icon="🪜", layout="wide")

config = load_config()

if "records" not in st.session_state:
    st.session_state.records = []
if "processed_keys" not in st.session_state:
    st.session_state.processed_keys = set()

st.title("Work-at-Height Compliance Screening")
st.markdown(
    "Screens site photographs against **SS 659** (scaffolds), **SS 528** "
    "(personal fall-arrest systems) and **SS 570** (anchor devices and "
    "horizontal lifeline systems) in a single pass. Severity is reported on "
    "the WSH Risk Assessment scale."
)

st.warning(
    "Screening aid only. This is not a formal inspection, does not discharge "
    "any duty under the WSH Act or its subsidiary legislation, and must not "
    "replace examination by a competent person or, for scaffolds, an approved "
    "Scaffold Supervisor. Assessment is limited to what is visible in the "
    "image — absence of a finding is not evidence of compliance.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Settings")

    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    if not default_key:
        default_key = os.environ.get("ANTHROPIC_API_KEY", "")

    api_key = st.text_input("Anthropic API key", type="password",
                            value=default_key,
                            help="Not persisted after the session ends.")

    model = MODELS[st.selectbox("Model", list(MODELS.keys()))]

    input_mode = st.radio("Input", ["Photograph", "Video walkthrough"])
    frame_interval, max_frames = 5.0, 5
    if input_mode == "Video walkthrough":
        frame_interval = st.slider("Frame interval (seconds)",
                                   1.0, 30.0, 5.0, step=0.5)
        max_frames = st.slider("Maximum frames", 1, 20, 5)

    st.divider()
    st.subheader("Report details")
    project = st.text_input("Project")
    location = st.text_input("Location")
    inspector = st.text_input("Screened by")

    st.divider()
    st.caption(f"Session register: {len(st.session_state.records)} image(s)")
    if st.session_state.records and st.button("Clear register"):
        st.session_state.records = []
        st.session_state.processed_keys = set()
        st.rerun()

    st.caption(
        "Clause references in `checks.yaml` are placeholders. Fill them in "
        "from your own licensed copies of the standards."
    )

prompt = build_prompt(config)

tab_screen, tab_stats = st.tabs(["Screening", "Statistics & report"])

# --- Screening -------------------------------------------------------------

with tab_screen:
    if input_mode == "Photograph":
        uploaded_files = st.file_uploader(
            "Upload site photographs",
            type=["jpg", "jpeg", "png", "webp", "heic", "heif",
                  "bmp", "tiff"],
            accept_multiple_files=True,
            help="Select or drag in as many photographs as you like. "
                 "Each is screened against all three standards.",
        )

        # Decode everything up front so unreadable files are reported before
        # any billable request is made.
        loaded: list[tuple[str, Image.Image, str]] = []   # (name, image, key)
        failed: list[tuple[str, str]] = []

        for uf in uploaded_files or []:
            try:
                img = load_uploaded_image(uf)
            except ValueError as exc:
                failed.append((uf.name, str(exc)))
            else:
                loaded.append((uf.name, img, file_key(uf)))

        if failed:
            with st.expander(f"{len(failed)} file(s) could not be read",
                             expanded=True):
                for name, msg in failed:
                    st.error(f"**{name}** — {msg}")

        if loaded:
            already = st.session_state.processed_keys
            pending = [item for item in loaded if item[2] not in already]
            skipped = len(loaded) - len(pending)

            st.caption(
                f"{len(loaded)} image(s) ready"
                + (f" · {skipped} already screened this session and will be "
                   "skipped" if skipped else "")
            )

            # Thumbnail strip
            cols = st.columns(min(len(loaded), 5))
            for i, (name, img, key) in enumerate(loaded):
                with cols[i % len(cols)]:
                    st.image(img, width="stretch")
                    done = " ✓" if key in already else ""
                    st.caption(f"{name[:26]}{done}")

            est = APPROX_COST_PER_FRAME_USD.get(model, 0.03) * len(pending)
            if pending:
                st.info(
                    f"{len(pending)} image(s) will be sent as separate "
                    f"requests — roughly ${est:.2f}."
                )
            else:
                st.success(
                    "All uploaded images have already been screened this "
                    "session. Clear the register in the sidebar to re-run."
                )

            rerun_all = st.checkbox(
                "Re-screen images already in the register", value=False)

            if st.button("Run screening", type="primary",
                         disabled=not (pending or rerun_all)):
                if not api_key:
                    st.error("Enter an Anthropic API key in the sidebar.")
                else:
                    queue = loaded if rerun_all else pending
                    progress = st.progress(0.0)
                    status = st.empty()
                    batch: list[tuple[str, Image.Image, Assessment | None,
                                      str | None]] = []

                    for i, (name, img, key) in enumerate(queue):
                        status.caption(
                            f"Assessing {i + 1} of {len(queue)}: {name}")
                        try:
                            raw, stop_reason = call_claude(
                                api_key, model, img, prompt)
                            a = parse_assessment(raw, config)
                            if stop_reason == "max_tokens":
                                a.truncated = True
                            batch.append((name, img, a, None))
                            st.session_state.processed_keys.add(key)
                        except Exception as exc:
                            # One bad image must not abandon the rest.
                            batch.append((name, img, None, str(exc)))
                        progress.progress((i + 1) / len(queue))

                    status.empty()
                    st.divider()

                    ok = [b for b in batch if b[2] is not None]
                    errored = [b for b in batch if b[2] is None]

                    # Batch summary before the detail
                    if ok:
                        worst = max(a.max_rpn for _, _, a, _ in ok)
                        worst_band = (
                            risk_band(worst, config["risk_bands"])[0]
                            if worst else "None")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Screened", len(ok))
                        m2.metric("Failed", len(errored))
                        m3.metric("Peak RPN", worst)
                        m4.metric("Peak band", worst_band)

                    if errored:
                        with st.expander(
                                f"{len(errored)} image(s) failed",
                                expanded=True):
                            for name, _, _, err in errored:
                                st.error(f"**{name}** — {err}")

                    # Worst images first so the urgent ones are on top
                    ok.sort(key=lambda b: -b[2].max_rpn)

                    for name, img, a, _ in ok:
                        band = (risk_band(a.max_rpn,
                                          config["risk_bands"])[0]
                                if a.max_rpn else "None")
                        with st.expander(
                            f"{band} · RPN {a.max_rpn} · {name}",
                            expanded=(band == "High"),
                        ):
                            png = render_assessment(img, a, config)

                        st.session_state.records.append(ScreeningRecord(
                            source=name,
                            timestamp=datetime.now(),
                            assessment=a,
                            annotated_png=png,
                        ))

                    if ok:
                        st.success(
                            f"{len(ok)} screening(s) added to the register. "
                            "Open the **Statistics & report** tab for "
                            "collated figures across all three standards."
                        )

    else:
        if not CV2_AVAILABLE:
            st.error("Video mode requires OpenCV: "
                     "pip install opencv-python-headless")
        else:
            uploaded_video = st.file_uploader(
                "Upload video walkthrough",
                type=["mp4", "mov", "avi", "mkv"])

            if uploaded_video:
                est = APPROX_COST_PER_FRAME_USD.get(model, 0.03) * max_frames
                st.info(
                    f"Up to {max_frames} frames will be sent as separate "
                    f"requests — roughly ${est:.2f}."
                )

                if st.button("Extract frames and run screening",
                             type="primary"):
                    if not api_key:
                        st.error("Enter an Anthropic API key in the sidebar.")
                    else:
                        tmp_path = None
                        try:
                            suffix = os.path.splitext(uploaded_video.name)[1]
                            with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=suffix) as tmp:
                                tmp.write(uploaded_video.read())
                                tmp_path = tmp.name

                            frames = extract_frames(tmp_path, frame_interval,
                                                    max_frames)
                            if not frames:
                                st.error("No frames could be read.")
                            else:
                                progress = st.progress(0.0)
                                results = []
                                for i, (frame_img, ts) in enumerate(frames):
                                    try:
                                        raw, stop_reason = call_claude(
                                            api_key, model, frame_img, prompt)
                                        a = parse_assessment(raw, config)
                                        if stop_reason == "max_tokens":
                                            a.truncated = True
                                        results.append(
                                            (ts, frame_img, a, None))
                                    except Exception as exc:
                                        results.append(
                                            (ts, frame_img, None, str(exc)))
                                    progress.progress((i + 1) / len(frames))

                                st.divider()
                                for ts, frame_img, a, err in results:
                                    if err:
                                        with st.expander(
                                                f"t = {ts:.1f}s — failed"):
                                            st.error(err)
                                        continue
                                    b = (risk_band(
                                        a.max_rpn, config["risk_bands"])[0]
                                        if a.max_rpn else "None")
                                    with st.expander(
                                            f"t = {ts:.1f}s — {b} "
                                            f"(RPN {a.max_rpn})"):
                                        png = render_assessment(
                                            frame_img, a, config)
                                    st.session_state.records.append(
                                        ScreeningRecord(
                                            source=(f"{uploaded_video.name} "
                                                    f"@ {ts:.1f}s"),
                                            timestamp=datetime.now(),
                                            assessment=a,
                                            annotated_png=png,
                                        ))
                                st.success(
                                    f"{len(results)} frame(s) added to the "
                                    "session register."
                                )
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.unlink(tmp_path)

# --- Statistics ------------------------------------------------------------

with tab_stats:
    records = st.session_state.records

    if not records:
        st.info(
            "No screenings yet. Run one or more images in the **Screening** "
            "tab and the collated figures will build up here."
        )
    else:
        stats = compute_statistics(records, config)
        peak_band = (risk_band(stats["peak_rpn"], config["risk_bands"])[0]
                     if stats["peak_rpn"] else "None")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Images", stats["image_count"])
        c2.metric("Elements", stats["element_count"])
        c3.metric("Non-compliances", stats["breach_count"])
        c4.metric("Peak RPN", stats["peak_rpn"])
        c5.metric("Peak band", peak_band)

        st.divider()
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("By standard")
            st.dataframe(
                [{
                    "Standard": row["short"],
                    "Elements": row["elements"],
                    "Breaches": row["breaches"],
                    "High": row["High"],
                    "Medium": row["Medium"],
                    "Low": row["Low"],
                    "Peak RPN": row["peak_rpn"],
                } for row in stats["by_category"]],
                width="stretch", hide_index=True,
            )

        with col_b:
            st.subheader("Severity distribution")
            dist = stats["severity_distribution"]
            if dist:
                st.bar_chart(
                    {"Breaches": [dist.get(i, 0) for i in range(1, 6)]})
                st.caption("Severity 1 (negligible) to 5 (catastrophic)")
            else:
                st.caption("No breaches recorded.")

        st.subheader("Most frequent non-compliances")
        if stats["top_checks"]:
            st.dataframe(
                [{
                    "Requirement": t["check"], "Standard": t["standard"],
                    "Occurrences": t["count"], "Peak RPN": t["peak_rpn"],
                } for t in stats["top_checks"]],
                width="stretch", hide_index=True,
            )
        else:
            st.caption("No breaches recorded.")

        st.subheader("Priority actions")
        if stats["all_breaches"]:
            st.dataframe(
                [{
                    "Band": b["band"], "RPN": b["rpn"],
                    "Standard": b["standard"], "Source": b["source"],
                    "Element": b["element"], "Requirement": b["check"],
                    "Observation": b["observation"],
                } for b in stats["all_breaches"][:50]],
                width="stretch", hide_index=True,
            )

        st.subheader("Open items requiring on-site verification")
        st.caption(
            f"{stats['not_visible_count']} check(s) could not be assessed "
            "from the images supplied."
        )
        if stats["not_visible_by_check"]:
            st.dataframe(
                [{
                    "Requirement": n["check"], "Standard": n["standard"],
                    "Occurrences": n["count"],
                } for n in stats["not_visible_by_check"]],
                width="stretch", hide_index=True,
            )

        st.divider()
        st.subheader("Export")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d1, d2 = st.columns(2)

        with d1:
            try:
                docx_bytes = build_report(
                    records, stats, config,
                    project=project, location=location, inspector=inspector,
                )
                st.download_button(
                    "Download Word report (.docx)", data=docx_bytes,
                    file_name=f"WAH_screening_report_{stamp}.docx",
                    mime=("application/vnd.openxmlformats-officedocument"
                          ".wordprocessingml.document"),
                    type="primary",
                )
            except Exception as exc:
                st.error(f"Could not build the Word report: {exc}")

        with d2:
            st.download_button(
                "Download findings (.csv)",
                data=register_to_csv(records, config),
                file_name=f"WAH_findings_{stamp}.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Work-at-Height Compliance Screening · preliminary aid only · "
    "no legal or regulatory standing under the WSH Act."
)
