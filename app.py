"""
Work-at-Height Compliance Screening Tool
========================================
Preliminary visual screening of site photographs and video walkthroughs
against SS 659 (scaffolds), SS 528 (personal fall-arrest systems) and
SS 570 (anchor devices and horizontal lifeline systems).

Severity is reported on the WSH Council Risk Assessment scale (1-5) and
combined with likelihood to give a risk band, so that output can be
transcribed directly into an RA form.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Optional: lets the app open HEIC/HEIF files, which phones and WhatsApp
# frequently produce while still naming them .jpg or .jpeg.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIF_AVAILABLE = True
except ImportError:
    HEIF_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "checks.yaml"

# Anthropic vision works best at or below this long edge. Larger images cost
# more tokens without improving accuracy.
MAX_IMAGE_EDGE = 1568

# A busy frame can carry several elements, each with a finding per check.
# Too low a ceiling truncates the JSON mid-object.
MAX_OUTPUT_TOKENS = 12000

MODELS = {
    "Claude Sonnet 5 (recommended)": "claude-sonnet-5",
    "Claude Opus 5 (highest accuracy, higher cost)": "claude-opus-5",
    "Claude Haiku 4.5 (fastest, lowest cost)": "claude-haiku-4-5-20251001",
}

# Rough per-image cost used only for the pre-run warning in video mode.
APPROX_COST_PER_FRAME_USD = {
    "claude-sonnet-5": 0.02,
    "claude-opus-5": 0.10,
    "claude-haiku-4-5-20251001": 0.005,
}

STATUS_COLOURS = {
    # (fill RGBA, border RGBA)
    "compliant":     ((0, 170, 70, 45),    (0, 140, 55, 240)),
    "non_compliant": ((210, 30, 30, 55),   (170, 20, 20, 245)),
    "not_visible":   ((130, 130, 130, 30), (110, 110, 110, 200)),
}

RISK_COLOURS = {
    "Low": "#0f9d58",
    "Medium": "#f4a300",
    "High": "#c62828",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def risk_band(rpn: int, bands: list) -> tuple[str, str]:
    """Map a risk priority number to a band label and required action."""
    for band in bands:
        if rpn <= band["max"]:
            return band["label"], band["action"]
    return bands[-1]["label"], bands[-1]["action"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    check_id: str
    check_title: str
    clause_ref: str
    status: str            # compliant | non_compliant | not_visible
    observation: str
    severity: int          # 1-5, WSH RA severity axis
    likelihood: int        # 1-5
    severity_rationale: str = ""

    @property
    def rpn(self) -> int:
        if self.status != "non_compliant":
            return 0
        return self.severity * self.likelihood


@dataclass
class Element:
    element_id: str
    label: str
    bbox: tuple[float, float, float, float] | None   # normalised 0-1
    bbox_confidence: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def status(self) -> str:
        if any(f.status == "non_compliant" for f in self.findings):
            return "non_compliant"
        if all(f.status == "not_visible" for f in self.findings):
            return "not_visible"
        return "compliant"

    @property
    def max_rpn(self) -> int:
        return max((f.rpn for f in self.findings), default=0)

    @property
    def max_severity(self) -> int:
        vals = [f.severity for f in self.findings if f.status == "non_compliant"]
        return max(vals, default=0)


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

    @property
    def all_findings(self) -> list[Finding]:
        return [f for e in self.elements for f in e.findings]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(cat_key: str, config: dict) -> str:
    cat = config["categories"][cat_key]

    check_lines = []
    for chk in cat["checks"]:
        check_lines.append(
            f'- check_id "{chk["id"]}" — {chk["title"]}\n'
            f'  What to look for: {chk["look_for"].strip()}\n'
            f'  Baseline severity if breached: {chk["base_severity"]}'
        )
    checks_block = "\n".join(check_lines)

    sev_block = "\n".join(
        f"  {k} = {v}" for k, v in config["severity_scale"].items()
    )
    lik_block = "\n".join(
        f"  {k} = {v}" for k, v in config["likelihood_scale"].items()
    )

    return f"""You are assisting a workplace safety officer with a preliminary visual screening of a work-at-height site photograph in Singapore, against {cat['standard']}.

## Step 1 — Identify elements

{cat['scope'].strip()}

If you cannot see any relevant element in the image, return an empty elements list and explain why in scene_notes. Do not invent elements.

## Step 2 — Assess each element against these checks

{checks_block}

For every element, return a finding for EVERY check_id listed above. Use status "not_visible" when the image does not show enough to judge — this is expected and preferable to guessing. Never mark something compliant merely because you cannot see a defect.

## Step 3 — Rate severity and likelihood

Severity is the worst credible outcome if this breach leads to an incident:
{sev_block}

Likelihood is how probable that incident is given what you can see (exposure, height, whether anyone is currently working there):
{lik_block}

Start from the baseline severity given for each check. You may adjust it up or down by one point based on what you actually observe — for example, a missing guardrail on a 12 m facade scaffold with workers present versus the same defect on a 1.5 m platform. If you adjust, say why in severity_rationale. For compliant or not_visible findings, still return your severity estimate but set likelihood to 1.

## Step 4 — Bounding boxes

For each element, give a bounding box as [x0, y0, x1, y1] in NORMALISED coordinates from 0.0 to 1.0, where (0,0) is the top-left of the image and (1,1) is the bottom-right. The box must tightly enclose the element you are describing.

Set bbox_confidence to "high", "medium", or "low". If you genuinely cannot localise the element, set bbox to null rather than guessing — a wrong box on a safety report is worse than no box.

## Output format

Respond with ONLY a JSON object. No markdown fences, no preamble, no commentary.

Keep it compact. Each observation must be a single sentence under 30 words, and severity_rationale under 20 words — omit it entirely when you did not adjust the baseline. A busy frame may contain several elements, and an over-long response will be cut off before it is complete.

{{
  "scene_notes": "one or two sentences on what the image shows, viewing angle, and anything limiting the assessment",
  "elements": [
    {{
      "element_id": "{cat['element_prefix']}-1",
      "label": "{cat['element_name']} 1 — short descriptor",
      "bbox": [0.12, 0.05, 0.48, 0.91],
      "bbox_confidence": "high",
      "findings": [
        {{
          "check_id": "one of the ids listed above",
          "status": "compliant | non_compliant | not_visible",
          "observation": "what you actually see — ONE sentence, under 30 words",
          "severity": 1,
          "likelihood": 1,
          "severity_rationale": "only if you adjusted from the baseline"
        }}
      ]
    }}
  ],
  "summary": "two or three sentences covering the most serious findings and what needs attention first"
}}
"""


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def sniff_format(head: bytes) -> str:
    """Identify the real container from magic bytes, ignoring the filename."""
    if head[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1",
                     b"msf1", b"avif", b"avis"):
            return "HEIC/AVIF"
        return "MP4/MOV video"
    if head[:4] in (b"GIF8",):
        return "GIF"
    if head[:2] == b"BM":
        return "BMP"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "TIFF"
    return "unknown"


def load_uploaded_image(uploaded) -> Image.Image:
    """
    Open an uploaded file as an RGB image.

    Streamlit's UploadedFile is a buffer whose position may already have moved,
    and phone photos routinely carry a misleading extension. Read the bytes
    once, check what the file actually is, and raise something an operator can
    act on rather than PIL's bare UnidentifiedImageError.
    """
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
                f"'{uploaded.name}'. Phones and WhatsApp often do this.\n\n"
                "Either install HEIC support with `pip install pillow-heif` "
                "and add `pillow-heif` to requirements.txt, or re-save the "
                "photo as JPEG before uploading."
            ) from None
        if detected.endswith("video"):
            raise ValueError(
                "This is a video file, not a photograph. Switch to "
                "'Video walkthrough' in the sidebar."
            ) from None
        raise ValueError(
            f"Could not decode this file. It is named '{uploaded.name}' but "
            f"its contents look like: {detected}. The file may be corrupt or "
            "in an unsupported format — try re-saving it as a JPEG or PNG."
        ) from None
    except OSError as exc:
        raise ValueError(
            f"The image file appears to be truncated or damaged ({exc}). "
            "Try uploading it again."
        ) from None

    # Phone photos carry rotation in EXIF rather than in the pixel data.
    # Without this, a portrait photo is analysed sideways.
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    return image.convert("RGB")


def downscale(image: Image.Image, max_edge: int = MAX_IMAGE_EDGE) -> Image.Image:
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
# API call
# ---------------------------------------------------------------------------

def call_claude(api_key: str, model: str, image: Image.Image,
                prompt: str) -> tuple[str, str]:
    """Return (response_text, stop_reason)."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)
    image_b64 = encode_image(downscale(image))

    message = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(
        block.text for block in message.content if block.type == "text"
    )
    return text, (message.stop_reason or "")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _repair_truncated_json(text: str) -> dict | None:
    """
    Recover the complete elements from a response that was cut off mid-object.

    Walks the string tracking brace depth (ignoring braces inside strings) and
    finds the last point at which an element object closed cleanly. Everything
    after that is discarded and the structure is closed off. Better to report
    two fully-assessed workers than to lose all three.
    """
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
            # Returning to depth 2 means one element just closed.
            if depth == 2 and ch == "}":
                last_element_end = i

    if last_element_end == -1:
        return None

    candidate = s[:last_element_end + 1] + "]}"
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    data["_truncated"] = True
    return data


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the response, tolerating stray fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost brace pair.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
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


def _clamp_bbox(raw) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v < -0.05 or v > 1.05 for v in vals):
        return None
    x0, y0, x1, y1 = [min(max(v, 0.0), 1.0) for v in vals]
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def parse_assessment(text: str, cat_key: str, config: dict) -> Assessment:
    data = _extract_json(text)
    cat = config["categories"][cat_key]
    check_lookup = {c["id"]: c for c in cat["checks"]}

    elements: list[Element] = []
    for i, raw_elem in enumerate(data.get("elements", []), start=1):
        findings: list[Finding] = []
        for raw_find in raw_elem.get("findings", []):
            cid = raw_find.get("check_id", "")
            meta = check_lookup.get(cid)
            if meta is None:
                continue  # model invented a check id — drop it

            status = raw_find.get("status", "not_visible")
            if status not in ("compliant", "non_compliant", "not_visible"):
                status = "not_visible"

            def _rating(key, default):
                try:
                    return min(max(int(raw_find.get(key, default)), 1), 5)
                except (TypeError, ValueError):
                    return default

            findings.append(Finding(
                check_id=cid,
                check_title=meta["title"],
                clause_ref=meta.get("clause_ref", ""),
                status=status,
                observation=str(raw_find.get("observation", "")).strip(),
                severity=_rating("severity", meta["base_severity"]),
                likelihood=_rating("likelihood", 1),
                severity_rationale=str(
                    raw_find.get("severity_rationale", "")).strip(),
            ))

        elements.append(Element(
            element_id=str(raw_elem.get(
                "element_id", f"{cat['element_prefix']}-{i}")),
            label=str(raw_elem.get(
                "label", f"{cat['element_name']} {i}")),
            bbox=_clamp_bbox(raw_elem.get("bbox")),
            bbox_confidence=str(raw_elem.get("bbox_confidence", "low")).lower(),
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
# Annotation — draws only boxes the model actually returned
# ---------------------------------------------------------------------------

def _load_fonts(width: int):
    size = max(14, width // 55)
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def annotate(image: Image.Image, assessment: Assessment) -> tuple[Image.Image, int]:
    """Return the annotated image and the count of elements that had no box."""
    w, h = image.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_fonts(w)

    unlocated = 0

    for elem in assessment.elements:
        if elem.bbox is None:
            unlocated += 1
            continue

        x0 = int(elem.bbox[0] * w)
        y0 = int(elem.bbox[1] * h)
        x1 = int(elem.bbox[2] * w)
        y1 = int(elem.bbox[3] * h)

        fill, border = STATUS_COLOURS.get(
            elem.status, STATUS_COLOURS["not_visible"])

        # Dashed-looking border for low-confidence localisation.
        width_px = 4 if elem.bbox_confidence == "high" else 2
        draw.rectangle([x0, y0, x1, y1], fill=fill,
                       outline=border, width=width_px)

        sev = elem.max_severity
        conf_marker = "" if elem.bbox_confidence == "high" else " ~"
        label = (f"{elem.element_id}{conf_marker}  |  S{sev}"
                 if sev else f"{elem.element_id}{conf_marker}  |  OK")

        try:
            box = draw.textbbox((0, 0), label, font=font)
            tw, th = box[2] - box[0], box[3] - box[1]
        except Exception:
            tw, th = len(label) * 8, 16

        pad = 6
        ly0 = max(0, y0 - th - pad * 2)
        draw.rectangle(
            [x0, ly0, min(w, x0 + tw + pad * 2), ly0 + th + pad * 2],
            fill=border[:3] + (225,),
        )
        draw.text((x0 + pad, ly0 + pad), label,
                  fill=(255, 255, 255), font=font)

    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    return combined.convert("RGB"), unlocated


# ---------------------------------------------------------------------------
# Video frame extraction — sequential read, no seeking
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, interval_s: float, max_frames: int):
    if not CV2_AVAILABLE:
        return []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = 25.0

    step = max(1, int(round(fps * interval_s)))
    frames = []
    idx = 0

    try:
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((Image.fromarray(rgb), idx / fps))
            idx += 1
    finally:
        cap.release()

    return frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_assessment(image: Image.Image, assessment: Assessment,
                      config: dict, key_prefix: str = ""):
    bands = config["risk_bands"]

    if assessment.truncated:
        st.warning(
            "The model response was cut off before it finished. The elements "
            "below were recovered, but the image may contain further elements "
            "that were never reported. Treat this screening as incomplete and "
            "re-run it.",
            icon="⚠️",
        )

    if assessment.elements:
        annotated, unlocated = annotate(image, assessment)
        st.image(annotated, width="stretch",
                 caption="Boxes are the model's own localisation. "
                         "A tilde (~) marks lower-confidence placement.")
        if unlocated:
            st.caption(
                f"{unlocated} element(s) were assessed but could not be "
                "localised in the image, so no box is drawn for them."
            )
    else:
        st.image(image, width="stretch")
        st.info("No relevant elements were identified in this image.")

    if assessment.scene_notes:
        st.caption(assessment.scene_notes)

    # Headline risk
    rpn = assessment.max_rpn
    label, action = risk_band(rpn, bands) if rpn else ("None", "No breach identified.")

    if rpn == 0:
        st.success("No non-compliance identified in this image.")
    else:
        colour = RISK_COLOURS.get(label, "#666")
        st.markdown(
            f"<div style='padding:14px 18px;border-radius:8px;"
            f"background:{colour};color:white;font-size:1.05rem;'>"
            f"<strong>Highest risk: {label}</strong> &nbsp;·&nbsp; "
            f"RPN {rpn} &nbsp;·&nbsp; {action}</div>",
            unsafe_allow_html=True,
        )

    if assessment.summary:
        st.write("")
        st.write(assessment.summary)

    # Findings table, worst first
    breaches = sorted(
        [(e, f) for e in assessment.elements for f in e.findings
         if f.status == "non_compliant"],
        key=lambda pair: pair[1].rpn,
        reverse=True,
    )

    if breaches:
        st.subheader("Non-compliances")
        for elem, find in breaches:
            band_label, _ = risk_band(find.rpn, bands)
            with st.expander(
                f"{band_label} · RPN {find.rpn} · {elem.element_id} — "
                f"{find.check_title}",
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

    # Items that could not be judged
    unknowns = [(e, f) for e in assessment.elements for f in e.findings
                if f.status == "not_visible"]
    if unknowns:
        with st.expander(
            f"Could not be assessed from this image ({len(unknowns)})"
        ):
            for elem, find in unknowns:
                st.write(f"**{elem.element_id} — {find.check_title}**")
                if find.observation:
                    st.caption(find.observation)

    with st.expander("Raw model response"):
        st.code(assessment.raw, language="json")


def assessment_to_csv(assessment: Assessment, config: dict,
                      source_label: str) -> str:
    bands = config["risk_bands"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "source", "element_id", "element_label", "check", "clause_ref",
        "status", "observation", "severity", "likelihood", "rpn",
        "risk_band", "action",
    ])
    for elem in assessment.elements:
        for find in elem.findings:
            band_label, action = (
                risk_band(find.rpn, bands) if find.rpn else ("-", "-")
            )
            writer.writerow([
                source_label, elem.element_id, elem.label, find.check_title,
                find.clause_ref, find.status, find.observation,
                find.severity, find.likelihood, find.rpn,
                band_label, action,
            ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Work-at-Height Compliance Screening",
    page_icon="🪜",
    layout="wide",
)

config = load_config()

st.title("Work-at-Height Compliance Screening")
st.markdown(
    "Preliminary visual screening of site photographs against "
    "**SS 659** (scaffolds), **SS 528** (personal fall-arrest systems) and "
    "**SS 570** (anchor devices and horizontal lifeline systems). "
    "Severity is reported on the WSH Risk Assessment scale."
)

st.warning(
    "Screening aid only. This is not a formal inspection, does not discharge "
    "any duty under the WSH Act or its subsidiary legislation, and must not be "
    "relied on in place of examination by a competent person or, for scaffolds, "
    "a Scaffold Supervisor. Assessment is limited to what is visible in a "
    "single frame — absence of a finding is not evidence of compliance.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Read from the ANTHROPIC_API_KEY environment variable if set. "
             "Not persisted after the session ends.",
    )

    model_label = st.selectbox("Model", list(MODELS.keys()))
    model = MODELS[model_label]

    cat_key = st.selectbox(
        "Inspection category",
        options=list(config["categories"].keys()),
        format_func=lambda k: config["categories"][k]["label"],
    )
    st.caption(config["categories"][cat_key]["standard"])

    input_mode = st.radio("Input", ["Photograph", "Video walkthrough"])

    if input_mode == "Video walkthrough":
        frame_interval = st.slider(
            "Frame interval (seconds)", 1.0, 30.0, 5.0, step=0.5)
        max_frames = st.slider("Maximum frames", 1, 20, 5)

    st.divider()
    st.caption(
        "Clause references in `checks.yaml` are placeholders. Fill them in "
        "from your own licensed copies of the standards."
    )

prompt = build_prompt(cat_key, config)

# --- Photograph mode -------------------------------------------------------

if input_mode == "Photograph":
    uploaded = st.file_uploader(
        "Upload site photograph",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif", "bmp", "tiff"])

    image = None
    if uploaded:
        try:
            image = load_uploaded_image(uploaded)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.image(image, caption="Uploaded photograph",
                     width="stretch")

    if image is not None:
        if st.button("Run screening", type="primary"):
            if not api_key:
                st.error("Enter an Anthropic API key in the sidebar.")
            else:
                raw = ""
                assessment = None
                with st.spinner("Analysing…"):
                    try:
                        raw, stop_reason = call_claude(
                            api_key, model, image, prompt)
                        assessment = parse_assessment(raw, cat_key, config)
                        if stop_reason == "max_tokens":
                            assessment.truncated = True
                    except (ValueError, json.JSONDecodeError) as exc:
                        st.error(f"Could not parse the model response: {exc}")
                        if raw:
                            st.code(raw, language="text")
                    except Exception as exc:
                        st.error(f"Request failed: {exc}")

                if assessment is not None:
                    st.divider()
                    render_assessment(image, assessment, config)
                    st.download_button(
                        "Download findings (CSV)",
                        data=assessment_to_csv(
                            assessment, config, uploaded.name),
                        file_name=(
                            f"wah_screening_"
                            f"{datetime.now():%Y%m%d_%H%M%S}.csv"),
                        mime="text/csv",
                    )

# --- Video mode ------------------------------------------------------------

else:
    if not CV2_AVAILABLE:
        st.error(
            "Video mode requires OpenCV. Run: "
            "pip install opencv-python-headless"
        )
    else:
        uploaded_video = st.file_uploader(
            "Upload video walkthrough", type=["mp4", "mov", "avi", "mkv"])

        if uploaded_video:
            est = APPROX_COST_PER_FRAME_USD.get(model, 0.03) * max_frames
            st.info(
                f"Up to {max_frames} frames will be sent as separate API "
                f"requests — roughly ${est:.2f} at current pricing. "
                "Reduce the frame count to lower cost."
            )

            if st.button("Extract frames and run screening", type="primary"):
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

                        frames = extract_frames(
                            tmp_path, frame_interval, max_frames)

                        if not frames:
                            st.error("No frames could be read from the video.")
                        else:
                            progress = st.progress(0.0)
                            results = []

                            for i, (frame_img, ts) in enumerate(frames):
                                try:
                                    raw, stop_reason = call_claude(
                                        api_key, model, frame_img, prompt)
                                    assessment = parse_assessment(
                                        raw, cat_key, config)
                                    if stop_reason == "max_tokens":
                                        assessment.truncated = True
                                    results.append((ts, frame_img, assessment,
                                                    None))
                                except Exception as exc:
                                    results.append((ts, frame_img, None,
                                                    str(exc)))
                                progress.progress((i + 1) / len(frames))

                            ok = [r for r in results if r[2] is not None]
                            worst = max(
                                (r[2].max_rpn for r in ok), default=0)
                            band, action = (
                                risk_band(worst, config["risk_bands"])
                                if worst else ("None", "No breach identified.")
                            )

                            st.divider()
                            st.subheader("Walkthrough summary")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Frames analysed", len(results))
                            c2.metric("Frames with breaches",
                                      sum(1 for r in ok if r[2].max_rpn > 0))
                            c3.metric("Highest RPN", worst)
                            c4.metric("Risk band", band)
                            st.caption(action)

                            all_csv = "".join(
                                assessment_to_csv(
                                    a, config, f"t={ts:.1f}s")
                                if i == 0 else
                                assessment_to_csv(
                                    a, config, f"t={ts:.1f}s"
                                ).split("\n", 1)[1]
                                for i, (ts, _, a, _) in enumerate(ok)
                            )
                            st.download_button(
                                "Download all findings (CSV)",
                                data=all_csv,
                                file_name=(
                                    f"wah_walkthrough_"
                                    f"{datetime.now():%Y%m%d_%H%M%S}.csv"),
                                mime="text/csv",
                            )

                            st.subheader("Frame detail")
                            for ts, frame_img, assessment, err in results:
                                if err:
                                    with st.expander(
                                            f"t = {ts:.1f}s — request failed"):
                                        st.error(err)
                                    continue
                                b, _ = (
                                    risk_band(assessment.max_rpn,
                                              config["risk_bands"])
                                    if assessment.max_rpn else ("None", "")
                                )
                                with st.expander(
                                    f"t = {ts:.1f}s — {b} "
                                    f"(RPN {assessment.max_rpn})"
                                ):
                                    render_assessment(
                                        frame_img, assessment, config,
                                        key_prefix=f"f{ts}")
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)

st.divider()
st.caption(
    "Work-at-Height Compliance Screening · preliminary aid only · "
    "no legal or regulatory standing under the WSH Act."
)
