"""
Smart Drain Blockage Detection - hackathon MVP
Run:  streamlit run app.py
Setup: pip install streamlit google-genai pillow requests pandas pydeck
       set GEMINI_API_KEY in your environment
Put drain photos in a ./drains folder and list them in DRAINS below.
"""
import hashlib
import json
import os
import random
import time

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
BASE_LAT, BASE_LON = 28.6139, 77.2090  # TODO: set to your demo area
CACHE_FILE = "results_cache.json"       # saved Gemini results, survives restarts

# One entry per drain photo. Coordinates are made up around BASE_LAT/LON.
DRAINS = [
    {"id": "D1", "name": "Market Road drain", "file": "drains/d1.jpg", "lat": BASE_LAT + 0.004, "lon": BASE_LON + 0.002, "flood_history": 0.9},
    {"id": "D2", "name": "Bus Stand drain", "file": "drains/d2.jpg", "lat": BASE_LAT - 0.003, "lon": BASE_LON + 0.005, "flood_history": 0.6},
    {"id": "D3", "name": "School Lane drain", "file": "drains/d3.jpg", "lat": BASE_LAT + 0.001, "lon": BASE_LON - 0.004, "flood_history": 0.3},
    {"id": "D4", "name": "Park Street drain", "file": "drains/d4.jpg", "lat": BASE_LAT - 0.005, "lon": BASE_LON - 0.002, "flood_history": 0.5},
]

PROMPT = """You are inspecting a stormwater drain photo for a municipal maintenance team.
Return ONLY JSON with these keys:
{"blockage_percent": integer 0-100 (how much of the drain opening/flow path is obstructed),
 "severity": "clear" | "partial" | "severe",
 "debris_types": list of strings (e.g. plastic, silt, leaves, construction waste),
 "confidence": number 0-1,
 "reasoning": one short sentence}"""

st.set_page_config(page_title="Smart Drain Monitor", layout="wide")
st.title("AI Smart Drain Blockage Detection")


@st.cache_resource
def get_client():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def analyze_drain(img_bytes: bytes) -> dict:
    """Use the saved result if we have one; otherwise call Gemini and save the answer."""
    key = hashlib.md5(img_bytes).hexdigest()
    cache = load_cache()
    if key in cache:
        return cache[key]
    last_err = None
    for attempt in range(6):
        try:
            resp = get_client().models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = json.loads(resp.text)
            cache[key] = result
            with open(CACHE_FILE, "w") as f:
                json.dump(cache, f, indent=2)
            return result
        except genai_errors.ServerError as e:              # 503: busy server, retry
            last_err = e
            time.sleep(2 ** attempt)
        except genai_errors.ClientError as e:
            if e.code == 429 and "PerDay" not in str(e):   # per-minute limit: wait and retry
                last_err = e
                time.sleep(15)
            else:                                          # daily quota or other error: stop
                raise
    raise last_err


@st.cache_data(ttl=1800, show_spinner=False)
def forecast_max_rain(lat: float, lon: float) -> float:
    """Max forecast precipitation (mm/hr) over the next 24h from Open-Meteo."""
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "hourly": "precipitation",
                    "forecast_days": 2, "timezone": "auto"},
            timeout=10,
        ).json()
        vals = r["hourly"]["precipitation"]
        times = pd.to_datetime(r["hourly"]["time"])
        start = int(times.searchsorted(pd.Timestamp.now().floor("h")))
        return float(max(vals[start:start + 24] or [0]))
    except Exception:
        return 0.0


def risk_score(blockage_pct: float, rain_mm: float, history: float) -> float:
    rain_factor = min(rain_mm / 20.0, 1.0)  # 20 mm/hr ~ very heavy rain
    return round((blockage_pct / 100) * (0.3 + 0.7 * rain_factor) * (0.7 + 0.3 * history), 3)


def level(score: float):
    if score >= 0.5:
        return "HIGH", [220, 40, 40]
    if score >= 0.25:
        return "MEDIUM", [240, 160, 30]
    return "LOW", [40, 170, 80]


# ---------- Sidebar controls ----------
st.sidebar.header("Rainfall")
live_rain = forecast_max_rain(BASE_LAT, BASE_LON)
st.sidebar.metric("Forecast peak (next 24h)", f"{live_rain:.1f} mm/hr")
simulate = st.sidebar.checkbox("Simulate heavy rain")
rain = st.sidebar.slider("Simulated rain (mm/hr)", 0, 50, 25) if simulate else live_rain

# ---------- Analyze drains ----------
if "results" not in st.session_state:
    st.session_state["results"] = {}

# Load any saved results automatically (no API calls)
cache = load_cache()
for d in DRAINS:
    if d["id"] not in st.session_state["results"] and os.path.exists(d["file"]):
        with open(d["file"], "rb") as f:
            key = hashlib.md5(f.read()).hexdigest()
        if key in cache:
            st.session_state["results"][d["id"]] = cache[key]

if st.button("Run inspection on all drains", type="primary"):
    with st.spinner("Analyzing drain images..."):
        for d in DRAINS:
            if d["id"] not in st.session_state["results"] and os.path.exists(d["file"]):
                with open(d["file"], "rb") as f:
                    try:
                        st.session_state["results"][d["id"]] = analyze_drain(f.read())
                    except Exception as e:
                        st.error(f"{d['name']}: {str(e)[:200]}")

results = st.session_state["results"]
if not results:
    st.info("Add photos to ./drains and click 'Run inspection on all drains'.")
    st.stop()

rows = []
for d in DRAINS:
    a = results.get(d["id"])
    if not a:
        continue
    score = risk_score(a["blockage_percent"], rain, d["flood_history"])
    lvl, color = level(score)
    rows.append({
        **d, "blockage_%": a["blockage_percent"], "severity": a["severity"],
        "debris": ", ".join(a["debris_types"]), "confidence": a["confidence"],
        "why": a["reasoning"], "risk": score, "level": lvl, "color": color,
    })
df = pd.DataFrame(rows).sort_values("risk", ascending=False)

# ---------- Alerts ----------
high = df[df["level"] == "HIGH"]
if len(high):
    st.error(f"ALERT: {len(high)} drain(s) at high flood risk with {rain:.0f} mm/hr rain expected. Dispatch crews now.")
    for _, r in high.iterrows():
        st.write(f"- **{r['name']}**: {r['blockage_%']}% blocked ({r['debris']}), risk {r['risk']}")
else:
    st.success("No drains at high risk under current rainfall.")

# ---------- Map + ranked list ----------
c1, c2 = st.columns([3, 2])
with c1:
    layer = pdk.Layer("ScatterplotLayer",
                      df[["lat", "lon", "color", "name", "blockage_%", "risk", "level"]],
                      get_position="[lon, lat]", get_fill_color="color",
                      get_radius=120, radius_min_pixels=12, pickable=True)
    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=BASE_LAT, longitude=BASE_LON, zoom=14),
        tooltip={"text": "{name}\nBlockage: {blockage_%}%\nRisk: {risk} ({level})"},
    ))
with c2:
    st.subheader("Maintenance priority")
    st.dataframe(df[["name", "blockage_%", "debris", "risk", "level"]],
                 hide_index=True, use_container_width=True)

# ---------- Drain detail + simulated sensor ----------
st.subheader("Drain detail")
choice = st.selectbox("Select drain", df["name"])
sel = df[df["name"] == choice].iloc[0]
d1, d2 = st.columns(2)
with d1:
    st.image(sel["file"], caption=f"{sel['severity']} - {sel['why']}")
with d2:
    st.caption("Water level sensor (SIMULATED)")
    random.seed(sel["id"])
    base = 18
    st.line_chart([base + random.uniform(-1.5, 1.5) + i * sel["blockage_%"] / 100 * 0.9 for i in range(24)])