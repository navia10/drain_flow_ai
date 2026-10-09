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
import smtplib
import time
from email.message import EmailMessage

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
BASE_LAT, BASE_LON = 28.6139, 77.2090  # TODO: set to your demo area
SENSOR_DRAIN = "D2"      # which drain the IoT sensor is installed in
SENSOR_BLOCK_CM = 25     # water level (cm) that counts as blocked
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


# ---------- Alerts: Telegram + email ----------
def get_secret(name: str) -> str:
    """Read a setting from environment variables or Streamlit secrets."""
    v = os.environ.get(name, "")
    if not v:
        try:
            v = str(st.secrets[name])
        except Exception:
            v = ""
    return v


def build_alert(rows, rain_mm: float, sensor_cm=None) -> str:
    lines = [f"FLOOD RISK ALERT - {time.strftime('%d %b %Y %H:%M')}",
             f"Rainfall: {rain_mm:.0f} mm/hr", ""]
    for _, r in rows.iterrows():
        lines.append(f"{r['name']} ({r['id']}): {r['blockage_%']}% blocked, "
                     f"debris: {r['debris'] or 'n/a'}, risk {r['risk']} ({r['level']})")
        lines.append(f"Location: https://maps.google.com/?q={r['lat']},{r['lon']}")
    if sensor_cm is not None:
        lines.append(f"\nSensor water level ({SENSOR_DRAIN}): {sensor_cm:.1f} cm")
    lines.append("\nAction: send a crew to clear the drain(s) above before the rain peaks.")
    lines.append("Source: Smart Drain Monitor (hackathon demo, sample locations).")
    return "\n".join(lines)


def send_telegram(text: str):
    token, chat = get_secret("TELEGRAM_TOKEN"), get_secret("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return None
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat, "text": text}, timeout=10)
        return r.ok, ("sent" if r.ok else r.text[:120])
    except Exception as e:
        return False, str(e)[:120]


def send_email(subject: str, body: str):
    user, pwd, to = get_secret("SMTP_USER"), get_secret("SMTP_APP_PASSWORD"), get_secret("ALERT_EMAIL_TO")
    if not (user and pwd and to):
        return None
    try:
        m = EmailMessage()
        m["Subject"], m["From"], m["To"] = subject, user, to      # "to" can be comma-separated
        m.set_content(body)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
            srv.login(user, pwd)
            srv.send_message(m)
        return True, "sent"
    except Exception as e:
        return False, str(e)[:120]


def send_alerts(text: str) -> dict:
    out = {}
    for name, res in (("Telegram", send_telegram(text)),
                      ("Email", send_email("Flood risk alert: blocked drain(s)", text))):
        if res is not None:
            out[name] = res
    return out


@st.cache_resource
def mqtt_listener(topic: str, host: str = "broker.hivemq.com", port: int = 1883, user: str = "", pwd: str = ""):
    """Background MQTT subscriber (public HiveMQ broker, no sign-up). Returns a dict that updates live."""
    state = {"level": None, "time": None, "error": None}
    try:
        import paho.mqtt.client as mqtt

        def on_connect(client, userdata, flags, reason_code, properties=None):
            client.subscribe(topic)

        def on_message(client, userdata, msg):
            try:
                state["level"] = float(msg.payload.decode().strip())
                state["time"] = time.time()
            except ValueError:
                pass

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        if user:
            client.username_pw_set(user, pwd)
        if port in (8883, 8884):
            client.tls_set()
        client.connect(host, int(port), 60)
        client.loop_start()
    except Exception as e:
        state["error"] = str(e)
    return state


@st.cache_data(ttl=10, show_spinner=False)
def read_thingspeak(channel_id: str, api_key: str = ""):
    """Latest water level (cm) from field1 of a ThingSpeak channel, or None."""
    try:
        params = {"api_key": api_key} if api_key else {}
        j = requests.get(f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json",
                         params=params, timeout=8).json()
        return float(j["field1"])
    except Exception:
        return None


def risk_score(blockage_pct: float, rain_mm: float, history: float) -> float:
    rain_factor = min(rain_mm / 20.0, 1.0)  # 20 mm/hr ~ very heavy rain
    return round((blockage_pct / 100) * (0.3 + 0.7 * rain_factor) * (0.7 + 0.3 * history), 3)


def level(score: float):
    if score >= 0.45:
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

# ---------- IoT sensor (simulated, or real ESP32 via MQTT / ThingSpeak) ----------
st.sidebar.header("IoT sensor")
sensor_src = st.sidebar.radio("Source", ["Simulated", "MQTT (real ESP32, no sign-up)", "ThingSpeak (real ESP32)"])
st.session_state.setdefault("sensor_hist", [])
ch = ""      # ThingSpeak channel (set below if used)
topic = ""   # MQTT topic (set below if used)
mq_host, mq_user = "", ""
hist = st.session_state["sensor_hist"]
if sensor_src == "Simulated":
    sim_block = st.sidebar.checkbox("Simulate blockage")
    last = hist[-1] if hist else 8.0
    target = 50 if sim_block else 8
    level_cm = max(0.0, last + (target - last) * 0.35 + random.uniform(-1, 1))
elif sensor_src.startswith("MQTT"):
    mq_host = st.sidebar.text_input("MQTT broker", "broker.hivemq.com")
    mq_port = st.sidebar.number_input("Port", value=1883, step=1)
    mq_user = st.sidebar.text_input("Username (if required)")
    mq_pwd = st.sidebar.text_input("Password (if required)", type="password")
    topic = st.sidebar.text_input("MQTT topic", "drainai-demo-7421/level")
    st.sidebar.caption("Use your friend's broker, port, topic and login. The value must be the water "
                       "level in cm as plain text, e.g. 30 (not JSON).")
    mq = mqtt_listener(topic, mq_host, int(mq_port), mq_user, mq_pwd) if topic and mq_host else None
    level_cm = mq["level"] if mq else None
    if mq and mq["error"]:
        st.sidebar.warning("MQTT error: " + mq["error"][:80])
    elif topic and level_cm is None:
        st.sidebar.info("Connected. Waiting for the first reading...")
    elif mq and mq["time"]:
        st.sidebar.caption(f"Last message {int(time.time() - mq['time'])} s ago")
else:
    ch = st.sidebar.text_input("ThingSpeak channel ID")
    key = st.sidebar.text_input("Read API key (only if channel is private)", type="password")
    level_cm = read_thingspeak(ch, key) if ch else None
    if ch and level_cm is None:
        st.sidebar.warning("No reading yet. Check the channel ID and that field1 has data.")
if level_cm is not None:
    hist.append(round(level_cm, 1))
    del hist[:-30]
    st.sidebar.metric(f"Water level ({SENSOR_DRAIN})", f"{level_cm:.1f} cm",
                      "BLOCKED" if level_cm >= SENSOR_BLOCK_CM else "clear", delta_color="off")
auto_refresh = st.sidebar.checkbox("Auto-refresh sensor (10 s)")

st.sidebar.header("Alerts")
auto_alert = st.sidebar.checkbox("Auto-send alert when a drain turns HIGH")
cooldown_min = st.sidebar.number_input("Don't repeat for the same drain (minutes)", 1, 240, 30)

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
    pct = a["blockage_percent"]
    if d["id"] == SENSOR_DRAIN and level_cm is not None and level_cm >= SENSOR_BLOCK_CM:
        pct = max(pct, min(100, int(level_cm / 60 * 100)))   # sensor confirms a blockage
    score = risk_score(pct, rain, d["flood_history"])
    lvl, color = level(score)
    rows.append({
        **d, "blockage_%": pct, "severity": a["severity"],
        "debris": ", ".join(a["debris_types"]), "confidence": a["confidence"],
        "why": a["reasoning"], "risk": score, "level": lvl, "color": color,
    })
df = pd.DataFrame(rows).sort_values("risk", ascending=False)

tab_vision, tab_iot = st.tabs(["Vision & flood risk", "IoT live dashboard"])

with tab_vision:
    # ---------- Alerts ----------
    high = df[df["level"] == "HIGH"]
    if len(high):
        st.error(f"ALERT: {len(high)} drain(s) at high flood risk with {rain:.0f} mm/hr rain expected. Dispatch crews now.")
        for _, r in high.iterrows():
            st.write(f"- **{r['name']}**: {r['blockage_%']}% blocked ({r['debris']}), risk {r['risk']}")
        sent = st.session_state.setdefault("alerted", {})          # drain id -> time last alerted
        now = time.time()
        fresh = high[high["id"].map(lambda i: now - sent.get(i, 0) > cooldown_min * 60)]
        with st.expander("Alert message preview"):
            st.code(build_alert(high, rain, level_cm))
        send_now = st.button("Send alert now (maintenance team / authorities)")
        if send_now or (auto_alert and len(fresh)):
            targets = high if send_now else fresh
            results_sent = send_alerts(build_alert(targets, rain, level_cm))
            if not results_sent:
                st.warning("No alert channel configured. Set TELEGRAM_TOKEN + TELEGRAM_CHAT_ID "
                           "and/or SMTP_USER + SMTP_APP_PASSWORD + ALERT_EMAIL_TO.")
            else:
                for i in targets["id"]:
                    sent[i] = now
                for channel, (ok, info) in results_sent.items():
                    (st.success if ok else st.warning)(f"{channel}: {info}")
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
        if sel["id"] == SENSOR_DRAIN and hist:
            st.caption("Water level sensor (" + ("SIMULATED" if sensor_src == "Simulated" else "LIVE sensor") + ")")
            st.line_chart(hist)
        else:
            st.caption("Water level sensor (SIMULATED)")
            random.seed(sel["id"])
            base = 18
            st.line_chart([base + random.uniform(-1.5, 1.5) + i * sel["blockage_%"] / 100 * 0.9 for i in range(24)])



with tab_iot:
    st.caption("Browser dashboard: sample-drain map plus a live sensor view (MQTT / ThingSpeak). "
               "A ThingSpeak channel ID entered in the sidebar is passed in automatically, "
               "so this tab and the risk score read the same sensor.")
    if os.path.exists("drain_guard.html"):
        with open("drain_guard.html", encoding="utf-8") as f:
            html = f.read()
        if html.lstrip().startswith("!DOCTYPE"):          # fix a pasted first line missing "<"
            html = "<" + html.lstrip()
        html = html.replace("const CENTER = [12.9716, 77.5946];",
                            f"const CENTER = [{BASE_LAT}, {BASE_LON}];")   # same city as the app
        js = ""
        if ch:
            js += 'document.getElementById("chId").value=' + json.dumps(ch) + ';document.getElementById("connectBtn").click();'
        if topic and mq_host == "broker.hivemq.com" and not mq_user:   # dashboard tab only supports the public HiveMQ broker
            js += 'document.getElementById("mqttTopic").value=' + json.dumps(topic) + ';document.getElementById("mqttBtn").click();'
        if js:
            html = html.replace("</body>", "<script>" + js + "</script></body>")
        components.html(html, height=1500, scrolling=True)
    else:
        st.warning("drain_guard.html not found. Put it next to app2.py.")

if auto_refresh:
    time.sleep(10)
    st.rerun()