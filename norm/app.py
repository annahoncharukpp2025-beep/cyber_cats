
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
        parts.append(f"Політова дистанція за GPS: {distance:.1f} м.")
    if not np.isnan(flight_time):
        parts.append(f"Тривалість GPS-ділянки: {flight_time:.2f} с.")
    if not np.isnan(hspeed):
        parts.append(f"Максимальна горизонтальна швидкість: {hspeed:.2f} м/с.")
    if not np.isnan(vspeed):
        parts.append(f"Максимальна вертикальна швидкість: {vspeed:.2f} м/с.")
    if not np.isnan(again):
        parts.append(f"Максимальний набір висоти: {again:.2f} м.")

    gfs = gps_meta.get("gps_sampling_hz", np.nan)
    if not np.isnan(gfs):
        parts.append(f"GPS частота семплювання: близько {gfs:.2f} Гц.")
    ifs = imu_meta.get("imu_sampling_hz_empirical", np.nan)
    if not np.isnan(ifs):
        parts.append(f"IMU частота семплювання: близько {ifs:.2f} Гц.")
    return " ".join(parts) if parts else "Дані успішно оброблені."


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
.block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 2rem; }
.hero h1 { font-size: 3.2rem; line-height: 1.0; margin-bottom: 0.5rem; }
.hero p { color: #9CA3AF; margin-top: 0; }
.success-box { background: rgba(34, 197, 94, 0.16); border: 1px solid rgba(34, 197, 94, 0.35); padding: 1rem 1.2rem; border-radius: 14px; margin: 1rem 0 1.5rem 0; font-size: 1.05rem; }
.metric-card { background: #0F172A; border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 20px; min-height: 130px; }
.metric-label { color: #CBD5E1; font-size: 1rem; margin-bottom: 10px; }
.metric-value { color: #F8FAFC; font-size: 2.4rem; font-weight: 700; line-height: 1.1; }
.unit { font-size: 1rem; color: #94A3B8; margin-left: 0.35rem; }
.small-note { color: #94A3B8; font-size: 0.95rem; margin-top: -8px; margin-bottom: 20px; }
.section-title { font-size: 2rem; font-weight: 700; margin-top: 24px; margin-bottom: 10px; }
</style>
''', unsafe_allow_html=True)

st.markdown('''
<div class="hero">
    <h1>Drone Flight Analyzer</h1>
    <p>Завантаж ArduPilot .BIN і одразу отримай метрики, 3D-траєкторію та акуратні графіки.</p>
</div>
''', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload ArduPilot .BIN", type=["bin", "BIN"])
if uploaded is None:
    st.info("Завантаж `.BIN` файл, щоб побачити аналіз.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".BIN") as tmp:
    tmp.write(uploaded.getbuffer())
    temp_path = tmp.name

with st.spinner("Обробляю телеметрію..."):
    result = process_bin_file(temp_path)

gps = result["gps"]
imu = result["imu"]
metrics = result["metrics"]
gps_meta = result["gps_meta"]
imu_meta = result["imu_meta"]

st.markdown(
    f'''
    <div class="success-box">
        Оброблено рядків → GPS: {gps_meta.get("gps_samples_valid", 0)} | IMU: {imu_meta.get("imu_samples_valid", 0)}
    </div>
    ''',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Flight Metrics</div>', unsafe_allow_html=True)
row1 = st.columns(3)
render_metric_card(row1[0], "Distance", fmt(metrics.get("distance_m", np.nan)), "m")
render_metric_card(row1[1], "Max Horizontal Speed", fmt(metrics.get("max_horizontal_speed_mps", np.nan)), "m/s")
render_metric_card(row1[2], "Max Acceleration", fmt(metrics.get("max_acceleration_mps2", np.nan)), "m/s²")

row2 = st.columns(3)
render_metric_card(row2[0], "Flight Time", fmt(metrics.get("flight_time_s", np.nan)), "s")
render_metric_card(row2[1], "Max Vertical Speed", fmt(metrics.get("max_vertical_speed_mps", np.nan)), "m/s")
render_metric_card(row2[2], "Altitude Gain", fmt(metrics.get("altitude_gain_m", np.nan)), "m")

st.markdown(
    f'''
    <div class="small-note">
        GPS ~ {fmt(gps_meta.get("gps_sampling_hz", np.nan), 2)} Hz &nbsp;&nbsp;|&nbsp;&nbsp;
        IMU ~ {fmt(imu_meta.get("imu_sampling_hz_empirical", np.nan), 2)} Hz
    </div>
    ''',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(["3D Trajectory", "Flight Charts", "Raw Data", "Summary"])

with tab1:
    if gps.empty:
        st.warning("GPS дані відсутні або не пройшли фільтрацію.")
    else:
        st.plotly_chart(build_3d_figure(gps), use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        if not gps.empty:
            st.plotly_chart(build_alt_speed_figure(gps), use_container_width=True)
    with c2:
        if not imu.empty:
            st.plotly_chart(build_imu_figure(imu), use_container_width=True)

with tab3:
    d1, d2 = st.columns(2)
    with d1:
        st.subheader("GPS table")
        if gps.empty:
            st.write("Немає GPS.")
        else:
            st.dataframe(gps[["time_s", "lat_deg", "lon_deg", "alt_m", "speed_mps", "vz_mps", "east_m", "north_m", "up_m"]].round(4), use_container_width=True, height=360)
    with d2:
        st.subheader("IMU table")
        if imu.empty:
            st.write("Немає IMU.")
        else:
            st.dataframe(imu[["time_s", "acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "acc_lin_norm_mps2", "vel_est_norm_mps"]].round(4), use_container_width=True, height=360)

with tab4:
    st.subheader("Automatic summary")
    st.write(make_summary_text(metrics, gps_meta, imu_meta))
    summary_payload = {"metrics": metrics, "gps_meta": gps_meta, "imu_meta": imu_meta}
    st.download_button(
        "Download JSON summary",
        data=json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_summary.json",
        mime="application/json",
    )

st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    if not gps.empty:
        st.download_button("Download GPS CSV", data=gps.to_csv(index=False).encode("utf-8"), file_name=f"{Path(uploaded.name).stem}_gps.csv", mime="text/csv")
with c2:
    if not imu.empty:
        st.download_button("Download IMU CSV", data=imu.to_csv(index=False).encode("utf-8"), file_name=f"{Path(uploaded.name).stem}_imu.csv", mime="text/csv")
with c3:
    st.download_button("Download Metrics JSON", data=json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"), file_name=f"{Path(uploaded.name).stem}_metrics.json", mime="application/json")
