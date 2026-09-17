"""Dashboard look for the WSH screening app.

Mixes the dark navy sidebar and dark-header KPI tiles of reference 1 with the
soft blue-grey canvas, big coloured figures, wave fills and progress bars of
reference 2. Import once at the top of app.py and call inject_theme().
"""
from __future__ import annotations

from contextlib import contextmanager

import altair as alt
import pandas as pd
import streamlit as st

NAVY = "#1F2233"
INK = "#2A2E45"
MUTED = "#8A90A6"
CORAL = "#F07C5F"
TONES = {  # (solid, gradient end) — also used as risk-band colours
    "high": ("#FF5A5F", "#FF8A65"),
    "medium": ("#F7A928", "#FFD166"),
    "low": ("#1FC99A", "#4DE3D0"),
    "info": ("#2F80ED", "#3FC6FF"),
    "accent": (CORAL, "#F9A58F"),
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input {{ font-family: 'Montserrat', sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* sidebar */
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #fff; }}
section[data-testid="stSidebar"] label {{ color: #B7BDD0 !important; }}
/* sidebar text inherits a light colour; keep typed values dark on the light fields */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb="select"] div,
section[data-testid="stSidebar"] [data-baseweb="input"] div {{
  color: {INK} !important; -webkit-text-fill-color: {INK} !important; }}
section[data-testid="stSidebar"] input::placeholder,
section[data-testid="stSidebar"] textarea::placeholder {{
  color: {MUTED} !important; -webkit-text-fill-color: {MUTED} !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] svg,
section[data-testid="stSidebar"] [data-testid="stTextInput"] button svg {{ fill: {INK}; color: {INK}; }}
.wsh-brand {{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 1.4rem; }}
.wsh-brand .dot {{ width:34px; height:34px; border-radius:50%; background:{CORAL};
  display:grid; place-items:center; color:#fff; font-weight:700; }}
.wsh-brand .name {{ color:#fff; font-weight:600; font-size:1rem; line-height:1.2; }}
.wsh-brand .sub {{ color:#8A90A6; font-size:.72rem; }}

/* header */
.wsh-head h1 {{ font-size:1.6rem; font-weight:700; color:{INK}; margin:0; padding:0; }}
.wsh-crumb {{ color:{MUTED}; font-size:.8rem; margin:.15rem 0 1.1rem; }}
.wsh-crumb b {{ color:#5B6180; }}

/* generic white panel: st.container(key="card_*") */
[class*="st-key-card_"] {{ background:#fff; border-radius:.6rem; padding:1.1rem 1.2rem;
  box-shadow: 0 6px 18px rgba(47,62,110,.08); }}
.wsh-panel-title {{ font-weight:600; color:{INK}; font-size:.95rem; margin-bottom:.6rem; }}

/* KPI tile */
.wsh-kpi {{ background:#fff; border-radius:.6rem; overflow:hidden; height:100%;
  box-shadow: 0 6px 18px rgba(47,62,110,.08); }}
.wsh-kpi .top {{ background:{NAVY}; color:#fff; font-size:.72rem; font-weight:600;
  letter-spacing:.04em; padding:.45rem .9rem; }}
.wsh-kpi .body {{ padding:.8rem .9rem 0; }}
.wsh-kpi .val {{ font-size:1.9rem; font-weight:700; line-height:1.1; }}
.wsh-kpi .cap {{ color:{MUTED}; font-size:.75rem; margin-top:.2rem; }}
.wsh-kpi svg {{ display:block; width:100%; height:46px; margin-top:.5rem; }}

/* progress rows */
.wsh-bar {{ margin:.2rem 0 .85rem; }}
.wsh-bar .lbl {{ display:flex; justify-content:space-between; font-size:.75rem; color:#5B6180; }}
.wsh-bar .track {{ height:7px; background:#ECEFF4; border-radius:4px; margin-top:.3rem; }}
.wsh-bar .fill {{ height:7px; border-radius:4px; }}

.wsh-note {{ background:#FFF6E5; border-left:4px solid #F7A928; color:#6B5320;
  font-size:.8rem; padding:.6rem .9rem; border-radius:.4rem; margin-bottom:1rem; }}
.wsh-intro {{ color:#5B6180; font-size:.9rem; margin:-.6rem 0 .8rem; max-width:80ch; }}
.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
.stTabs [data-baseweb="tab"] {{ background:#fff; border-radius:.5rem .5rem 0 0; padding:.4rem 1rem; }}
/* buttons & uploader */
.stButton > button[kind="primary"] {{ box-shadow: 0 4px 12px rgba(240,124,95,.35); }}
[data-testid="stFileUploaderDropzone"] {{ background:#F6F8FB; border:1.5px dashed #C9D1E0; }}
@media (prefers-reduced-motion: no-preference) {{
  .wsh-bar .fill {{ transition: width .6s ease; }}
}}
</style>
"""


def inject_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_brand(name: str = "WSH Screener", sub: str = "SS 659 · SS 528 · SS 570") -> None:
    st.sidebar.markdown(
        f'<div class="wsh-brand"><div class="dot">W</div>'
        f'<div><div class="name">{name}</div><div class="sub">{sub}</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, *crumbs: str) -> None:
    trail = " › ".join([f"<b>{crumbs[0]}</b>", *crumbs[1:]]) if crumbs else ""
    st.markdown(
        f'<div class="wsh-head"><h1>{title}</h1></div><div class="wsh-crumb">{trail}</div>',
        unsafe_allow_html=True,
    )


def _wave(tone: str) -> str:
    a, b = TONES[tone]
    gid = f"g{tone}"
    return (
        f'<svg viewBox="0 0 200 46" preserveAspectRatio="none">'
        f'<defs><linearGradient id="{gid}" x1="0" x2="1"><stop offset="0" stop-color="{b}"/>'
        f'<stop offset="1" stop-color="{a}"/></linearGradient></defs>'
        f'<path d="M0,18 C30,40 55,6 90,24 C120,38 150,4 200,10 L200,46 L0,46 Z" fill="url(#{gid})"/></svg>'
    )


def kpi_card(title: str, value, caption: str = "", tone: str = "info") -> None:
    colour = TONES[tone][0]
    st.markdown(
        f'<div class="wsh-kpi"><div class="top">{title}</div><div class="body">'
        f'<div class="val" style="color:{colour}">{value}</div>'
        f'<div class="cap">{caption}</div></div>{_wave(tone)}</div>',
        unsafe_allow_html=True,
    )


def progress_rows(rows: dict[str, float], tones: list[str] | None = None) -> None:
    """rows: label -> fraction 0..1 (e.g. compliance rate per standard)."""
    tones = tones or ["high", "info", "medium", "low"]
    html = ""
    for i, (label, frac) in enumerate(rows.items()):
        a, _ = TONES[tones[i % len(tones)]]
        pct = max(0.0, min(1.0, float(frac))) * 100
        html += (
            f'<div class="wsh-bar"><div class="lbl"><span>{label}</span><span>{pct:.0f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct}%;background:{a}"></div></div></div>'
        )
    st.markdown(html, unsafe_allow_html=True)


@contextmanager
def panel(key: str, title: str | None = None):
    with st.container(key=f"card_{key}"):
        if title:
            st.markdown(f'<div class="wsh-panel-title">{title}</div>', unsafe_allow_html=True)
        yield


def _style(chart: alt.Chart, height: int) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#EEF1F6",
                        gridDash=[2, 3], domain=False, tickColor="transparent",
                        labelFont="Montserrat", titleFont="Montserrat")
    )


def risk_band_bars(counts: dict[str, int], height: int = 240) -> None:
    """Coloured bars per risk band, e.g. {'High': 4, 'Medium': 7, 'Low': 12}."""
    counts = {b: counts.get(b, 0) for b in ("High", "Medium", "Low")}
    df = pd.DataFrame({"band": list(counts), "findings": list(counts.values())})
    colours = [TONES.get(b.lower(), TONES["info"])[0] for b in df["band"]]
    chart = alt.Chart(df).mark_bar(size=26, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("band:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("findings:Q", title=None),
        color=alt.Color("band:N", scale=alt.Scale(domain=list(df["band"]), range=colours), legend=None),
        tooltip=["band", "findings"],
    )
    st.altair_chart(_style(chart, height), width="stretch")


def rpn_line(labels: list[str], rpns: list[float], names: list[str] | None = None,
             height: int = 240) -> None:
    """Worst RPN per image, in screening order. labels must be unique."""
    df = pd.DataFrame({"image": labels, "RPN": rpns, "source": names or labels})
    base = alt.Chart(df).encode(x=alt.X("image:N", sort=None, title=None,
                                        axis=alt.Axis(labelAngle=0, labelLimit=80)),
                                y=alt.Y("RPN:Q", title=None, scale=alt.Scale(domain=[0, 25])),
                                tooltip=["source", "RPN"])
    chart = base.mark_line(color="#F7B733", strokeWidth=2.5) + base.mark_circle(
        size=70, color=TONES["info"][0], opacity=1)
    st.altair_chart(_style(chart, height), width="stretch")


def simple_bars(labels: list[str], values: list[int], tone: str = "info",
                height: int = 240) -> None:
    df = pd.DataFrame({"label": labels, "value": values})
    chart = alt.Chart(df).mark_bar(size=26, cornerRadiusTopLeft=3, cornerRadiusTopRight=3,
                                   color=TONES[tone][0]).encode(
        x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("value:Q", title=None), tooltip=["label", "value"])
    st.altair_chart(_style(chart, height), width="stretch")


def band_tone(band: str) -> str:
    return {"High": "high", "Medium": "medium", "Low": "low"}.get(band, "low")
