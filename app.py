
from __future__ import annotations

import json
import tempfile
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tomllib
import google.generativeai as genai
from telemetry_core import process_bin_file


def fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{digits}f}"


def render_metric_card(container, label: str, value: str, unit: str = "") -> None:
    unit_html = f"<span class='unit'>{unit}</span>" if unit else ""
    container.markdown(
        f'''<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{unit_html}</div>
        </div>''',
        unsafe_allow_html=True,
    )


def make_summary_text(metrics, gps_meta, imu_meta) -> str:
    parts = []
    distance = metrics.get("distance_m", np.nan)
    flight_time = metrics.get("flight_time_s", np.nan)
    hspeed = metrics.get("max_horizontal_speed_mps", np.nan)
    vspeed = metrics.get("max_vertical_speed_mps", np.nan)
    again = metrics.get("altitude_gain_m", np.nan)

    if not np.isnan(distance):
        parts.append(f"Flight distance by GPS: {distance:.1f} m.")
    if not np.isnan(flight_time):
        parts.append(f"GPS section duration: {flight_time:.2f} s.")
    if not np.isnan(hspeed):
        parts.append(f"Maximum horizontal speed: {hspeed:.2f} m/s.")
    if not np.isnan(vspeed):
        parts.append(f"Maximum vertical speed: {vspeed:.2f} m/s.")
    if not np.isnan(again):
        parts.append(f"Maximum height gain: {again:.2f} m.")

    gfs = gps_meta.get("gps_sampling_hz", np.nan)
    if not np.isnan(gfs):
        parts.append(f"GPS sampling rate: approx. {gfs:.2f} Hz.")
    ifs = imu_meta.get("imu_sampling_hz_empirical", np.nan)
    if not np.isnan(ifs):
        parts.append(f"IMU sampling rate: approx. {ifs:.2f} Hz.")
    return "\n".join([f"- {p}" for p in parts]) if parts else "Data processed successfully."
    st.text(make_summary_text(metrics, gps_meta, imu_meta))


def build_3d_figure(gps):
    fig = px.scatter_3d(
        gps,
        x="east_m",
        y="north_m",
        z="up_m",
        color="speed_mps" if "speed_mps" in gps.columns else "time_s",
        hover_data=["time_s", "alt_m", "speed_mps", "vz_mps"],
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title="3D Trajectory (ENU)",
        scene=dict(xaxis_title="East (m)", yaxis_title="North (m)", zaxis_title="Up (m)"),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_alt_speed_figure(gps):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gps["time_s"], y=gps["alt_m"], mode="lines", name="Altitude (m)"))
    if "speed_mps" in gps.columns:
        fig.add_trace(go.Scatter(x=gps["time_s"], y=gps["speed_mps"], mode="lines", name="Horizontal speed (m/s)", yaxis="y2"))
    fig.update_layout(
        title="Altitude and Speed",
        xaxis_title="Time (s)",
        yaxis_title="Altitude (m)",
        yaxis2=dict(title="Speed (m/s)", overlaying="y", side="right"),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_imu_figure(imu):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=imu["time_s"], y=imu["acc_lin_norm_mps2"], mode="lines", name="Linear accel norm (m/s²)"))
    fig.add_trace(go.Scatter(x=imu["time_s"], y=imu["vel_est_norm_mps"], mode="lines", name="Integrated speed (m/s)", yaxis="y2"))
    fig.update_layout(
        title="IMU Acceleration and Integrated Speed",
        xaxis_title="Time (s)",
        yaxis_title="Acceleration (m/s²)",
        yaxis2=dict(title="Integrated speed (m/s)", overlaying="y", side="right"),
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


st.set_page_config(page_title="Drone Flight Analyzer", layout="wide")

st.markdown('''
<style>
.block-container { max-width: 1280px; padding-top: 32px; padding-bottom: 32px; }
.hero h1 { font-size: 50px; line-height: 1.0; margin-bottom: 8px; text-align: center; }
.hero p { margin-top: 0; margin-bottom: -20px; }
.metric-card { background: rgba(125, 150, 175, 1); border: 1px rgba(205, 237, 255, 0.8); border-radius: 18px; padding: 20px; min-height: 130px; min-width: 250px; justify-self: center; text-align: center; padding-right: 50px; padding-left: 50px; margin-bottom: 50px}  
.metric-label { font-size: 15px; margin-bottom: 10px; }
.metric-value { font-size: 35px; font-weight: 700; line-height: 1.1; }
.unit { font-size: 16px; margin-left: 6px; }
.section-title { font-size: 32px; font-weight: 700; margin-top: 24px; margin-bottom: 10px; padding-right: -20px; justify-self: center;}
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="hero">
    <h1>Drone Flight Analyzer</h1>
    <p>Upload the ArduPilot binary file  </p>
</div>
''', unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["bin"])
if uploaded is None:
    st.info("You'll get metrics, 3D trajectory, tables with raw data and flight analysis by AI")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp:
    tmp.write(uploaded.getbuffer())
    temp_path = tmp.name

with st.spinner("Procesing Data..."):
    result = process_bin_file(temp_path)

gps = result["gps"]
imu = result["imu"]
metrics = result["metrics"]
gps_meta = result["gps_meta"]
imu_meta = result["imu_meta"]

st.markdown('<div class="section-title">Flight Metrics</div>', unsafe_allow_html=True)
row1 = st.columns(3)
render_metric_card(row1[0], "Distance", fmt(metrics.get("distance_m", np.nan)), "m")
render_metric_card(row1[1], "Max Horizontal Speed", fmt(metrics.get("max_horizontal_speed_mps", np.nan)), "m/s")
render_metric_card(row1[2], "Max Acceleration", fmt(metrics.get("max_acceleration_mps2", np.nan)), "m/s²")

row2 = st.columns(3)
render_metric_card(row2[0], "Flight Time", fmt(metrics.get("flight_time_s", np.nan)), "s")
render_metric_card(row2[1], "Max Vertical Speed", fmt(metrics.get("max_vertical_speed_mps", np.nan)), "m/s")
render_metric_card(row2[2], "Altitude Gain", fmt(metrics.get("altitude_gain_m", np.nan)), "m")


tab1, tab2, tab3 = st.tabs(["3D Trajectory", "Raw Data", "Summary"])

with tab1:
    if gps.empty:
        st.warning("GPS data is missing or not filtered.")
    else:
        st.plotly_chart(build_3d_figure(gps), use_container_width=True)


with tab2:
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("GPS table")
        if gps.empty:
            st.write("No GPS data.")
        else:
            st.dataframe(gps[["time_s", "lat_deg", "lon_deg", "alt_m", "speed_mps", "vz_mps", "east_m", "north_m", "up_m"]].round(4), use_container_width=True, height=360)
    with d2:
        st.subheader("IMU table")
        if imu.empty:
            st.write("No IMU data.")
        else:
            st.dataframe(imu[["time_s", "acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "acc_lin_norm_mps2", "vel_est_norm_mps"]].round(4), use_container_width=True, height=360)

with tab3:
    st.subheader("Automatic summary")
    st.write(make_summary_text(metrics, gps_meta, imu_meta))
    summary_payload = {"metrics": metrics, "gps_meta": gps_meta, "imu_meta": imu_meta}

def analyze_flight_with_llm(metrics_dict):
    st.subheader(" AI: Flight analysis (LLM)")
    
    try:
        with open("api_key.toml", "rb") as f:
            config = tomllib.load(f)
            api_key = config["GEMINI_API_KEY"]
            
        genai.configure(api_key=api_key)
    except FileNotFoundError:
        st.warning("File api_key.toml not found in project folder.")
        return
    except Exception as e:
        st.warning(f"Reading key error. Check file extension of api_key.toml. Details: {e}")
        return

    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    You are an expert in UAV flight analysis.
    Your task is to analyse the flight metrics from the Ardupilot flight recorder and write a brief positive report (3–4 sentences).
    Keep your answers brief and to the point – for each indicator, state the key points on a new line, followed by a short description or analysis (3–4 words), and 
    finish with an overall conclusion on the flight.

    Here are the details of this flight:
    - Distance travelled: {metrics_dict.get('distance_m', 0):.2f} metres
    - Flight time: {metrics_dict.get('flight_time_s', 0):.1f} seconds
    - Max. horizontal speed: {metrics_dict.get('max_horizontal_speed_mps', 0):.2f} m/s
    - Max. vertical speed: {metrics_dict.get('max_vertical_speed_mps', 0):.2f} m/s
    - Max. acceleration (g-force): {metrics_dict.get('max_acceleration_mps2', 0):.2f} m/s²
    - Total altitude gain: {metrics_dict.get('altitude_gain_m', 0):.2f} metres

    Analysis rules:
    1. If acceleration > 30 m/s², this indicates a possible collision or a very hard landing. Flag this!
    2. If vertical velocity > 10 m/s, this may indicate a steep descent (spin).
    3. Draw a conclusion regarding the overall smoothness and safety of the flight.
    4. Respond in English, using a professional tone suitable for the general public.
    """

    with st.spinner("AI analyzes..."):
        try:
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"Error conection with LLM: {e}")

st.markdown("<br>", unsafe_allow_html=True) 
if metrics:
    analyze_flight_with_llm(metrics) 

