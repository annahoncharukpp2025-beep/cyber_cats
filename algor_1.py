#!/usr/bin/env python3
"""
Drone Flight Analyzer - Streamlit UI for ArduPilot DataFlash .BIN logs

Features:
- Upload .BIN directly in browser
- Parse GPS + IMU from ArduPilot DataFlash
- Correct GPS filtering (Status/NSats)
- WGS84 -> local ENU meters
- Haversine total distance
- IMU trapezoidal integration after ATT-based rotation + gravity compensation
- Clean user-facing dashboard with metric cards
- Interactive 3D trajectory and time-series charts

Run:
    streamlit run drone_flight_analyzer_app.py
"""

from __future__ import annotations

import io
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from pymavlink import DFReader
except Exception as exc:
    st.error(
        "Не вдалося імпортувати pymavlink.\n\n"
        "Встанови залежності у venv:\n"
        "`python -m pip install streamlit pymavlink pandas numpy plotly pyarrow`"
    )
    st.stop()

EARTH_RADIUS_M = 6371000.0
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GRAVITY_MPS2 = 9.80665


# =========================
# Parsing helpers
# =========================

def msg_to_dict(msg: Any) -> Dict[str, Any]:
    if hasattr(msg, "to_dict"):
        data = msg.to_dict()
        if isinstance(data, dict):
            return data

    out: Dict[str, Any] = {}
    fieldnames = getattr(msg, "_fieldnames", None)
    if fieldnames:
        for name in fieldnames:
            try:
                out[name] = getattr(msg, name)
            except Exception:
                pass

    if "mavpackettype" not in out:
        try:
            out["mavpackettype"] = msg.get_type()
        except Exception:
            out["mavpackettype"] = type(msg).__name__
    return out


def iter_bin_messages(bin_path: str | Path) -> Iterable[Dict[str, Any]]:
    reader = DFReader.DFReader_binary(str(bin_path))
    while True:
        msg = reader.recv_msg()
        if msg is None:
            break
        try:
            yield msg_to_dict(msg)
        except Exception:
            continue


def collect_message_tables(bin_path: str | Path,
                           wanted_types: Iterable[str] = ("GPS", "IMU", "ATT", "AHR2", "MODE", "MSG")
                           ) -> Dict[str, pd.DataFrame]:
    tables: Dict[str, List[Dict[str, Any]]] = {t: [] for t in wanted_types}
    for record in iter_bin_messages(bin_path):
        mtype = record.get("mavpackettype") or record.get("type")
        if mtype in tables:
            tables[mtype].append(record)
    return {mtype: (pd.DataFrame(rows) if rows else pd.DataFrame()) for mtype, rows in tables.items()}


def pick_first_existing(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required columns not found. Need one of: {candidates}. Available: {list(df.columns)}")
    return None


def to_numeric_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def normalize_time(df: pd.DataFrame, time_col: str = "TimeUS") -> pd.DataFrame:
    df = df.copy()
    if time_col not in df.columns:
        raise KeyError(f"{time_col} not found in columns: {list(df.columns)}")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty:
        df["time_s"] = pd.Series(dtype="float64")
        return df
    df["time_s"] = (df[time_col] - df[time_col].iloc[0]) / 1e6
    return df


def choose_primary_instance(df: pd.DataFrame, instance_col: str = "I") -> Tuple[pd.DataFrame, Optional[int]]:
    if df.empty or instance_col not in df.columns:
        return df.copy(), None
    counts = df[instance_col].value_counts(dropna=False)
    best_instance = counts.index[0]
    filtered = df[df[instance_col] == best_instance].copy().reset_index(drop=True)
    return filtered, int(best_instance) if pd.notna(best_instance) else None


# =========================
# GPS geometry
# =========================

def haversine_vec(lat1_deg: np.ndarray, lon1_deg: np.ndarray,
                  lat2_deg: np.ndarray, lon2_deg: np.ndarray) -> np.ndarray:
    lat1 = np.radians(lat1_deg.astype(float))
    lon1 = np.radians(lon1_deg.astype(float))
    lat2 = np.radians(lat2_deg.astype(float))
    lon2 = np.radians(lon2_deg.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return EARTH_RADIUS_M * c


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, alt_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return x, y, z


def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray, lat0_deg: float, lon0_deg: float, alt0_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, z0 = geodetic_to_ecef(np.array([lat0_deg]), np.array([lon0_deg]), np.array([alt0_m]))
    dx = x - x0[0]
    dy = y - y0[0]
    dz = z - z0[0]
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)
    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return east, north, up


# =========================
# Preprocessing
# =========================

def preprocess_gps(gps_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if gps_raw.empty:
        return pd.DataFrame(), {"warning": "No GPS messages found"}

    gps_raw = normalize_time(gps_raw)
    gps_raw, gps_instance = choose_primary_instance(gps_raw)

    lat_col = pick_first_existing(gps_raw, ["Lat"])
    lon_col = pick_first_existing(gps_raw, ["Lng", "Lon", "Long"])
    alt_col = pick_first_existing(gps_raw, ["Alt"])
    spd_col = pick_first_existing(gps_raw, ["Spd"], required=False)
    vz_col = pick_first_existing(gps_raw, ["VZ"], required=False)
    status_col = pick_first_existing(gps_raw, ["Status"], required=False)
    nsats_col = pick_first_existing(gps_raw, ["NSats", "NSat"], required=False)
    hdop_col = pick_first_existing(gps_raw, ["HDop"], required=False)

    gps = pd.DataFrame({
        "TimeUS": to_numeric_series(gps_raw, "TimeUS"),
        "time_s": to_numeric_series(gps_raw, "time_s"),
        "instance": to_numeric_series(gps_raw, "I", default=np.nan),
        "lat_deg": to_numeric_series(gps_raw, lat_col),
        "lon_deg": to_numeric_series(gps_raw, lon_col),
        "alt_m": to_numeric_series(gps_raw, alt_col),
        "speed_mps": to_numeric_series(gps_raw, spd_col) if spd_col else np.nan,
        "vz_mps": to_numeric_series(gps_raw, vz_col) if vz_col else np.nan,
        "status": to_numeric_series(gps_raw, status_col, default=np.nan) if status_col else np.nan,
        "nsats": to_numeric_series(gps_raw, nsats_col, default=np.nan) if nsats_col else np.nan,
        "hdop": to_numeric_series(gps_raw, hdop_col, default=np.nan) if hdop_col else np.nan,
    }).dropna(subset=["lat_deg", "lon_deg", "alt_m"]).reset_index(drop=True)

    gps = gps[
        gps["lat_deg"].between(-90, 90)
        & gps["lon_deg"].between(-180, 180)
        & ~((gps["lat_deg"].abs() < 1e-9) & (gps["lon_deg"].abs() < 1e-9))
    ].copy()

    # IMPORTANT: filter broken GPS fixes
    if "status" in gps.columns and gps["status"].notna().any():
        gps = gps[gps["status"] >= 3].copy()

    if "nsats" in gps.columns and gps["nsats"].notna().any():
        gps = gps[gps["nsats"] >= 6].copy()

    gps = gps.drop_duplicates(subset=["TimeUS"]).sort_values("TimeUS").reset_index(drop=True)
    if gps.empty:
        return gps, {"warning": "No valid GPS samples after filtering", "gps_instance": gps_instance}

    gps["seg_m"] = haversine_vec(
        gps["lat_deg"].shift(1).to_numpy(),
        gps["lon_deg"].shift(1).to_numpy(),
        gps["lat_deg"].to_numpy(),
        gps["lon_deg"].to_numpy(),
    )
    gps["seg_m"] = gps["seg_m"].fillna(0.0)
    gps["cum_dist_m"] = gps["seg_m"].cumsum()

    dt = gps["time_s"].diff()
    gps_fs_hz = float(1.0 / np.nanmedian(dt.to_numpy())) if len(gps) > 1 else np.nan

    lat0 = float(gps["lat_deg"].iloc[0])
    lon0 = float(gps["lon_deg"].iloc[0])
    alt0 = float(gps["alt_m"].iloc[0])

    x, y, z = geodetic_to_ecef(gps["lat_deg"].to_numpy(), gps["lon_deg"].to_numpy(), gps["alt_m"].to_numpy())
    east, north, up = ecef_to_enu(x, y, z, lat0, lon0, alt0)
    gps["east_m"] = east
    gps["north_m"] = north
    gps["up_m"] = up

    meta = {
        "gps_instance": gps_instance,
        "gps_samples_total": int(len(gps_raw)),
        "gps_samples_valid": int(len(gps)),
        "gps_sampling_hz": gps_fs_hz,
    }
    return gps, meta


def preprocess_att(att_raw: pd.DataFrame, ahr2_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    source = None
    att = pd.DataFrame()
    if not att_raw.empty:
        source = "ATT"
        att_raw = normalize_time(att_raw)
        roll_col = pick_first_existing(att_raw, ["Roll"])
        pitch_col = pick_first_existing(att_raw, ["Pitch"])
        yaw_col = pick_first_existing(att_raw, ["Yaw"])
        att = pd.DataFrame({
            "TimeUS": to_numeric_series(att_raw, "TimeUS"),
            "time_s": to_numeric_series(att_raw, "time_s"),
            "roll_deg": to_numeric_series(att_raw, roll_col),
            "pitch_deg": to_numeric_series(att_raw, pitch_col),
            "yaw_deg": to_numeric_series(att_raw, yaw_col),
        }).dropna()
    return att.reset_index(drop=True), {"att_source": source}


def body_to_nav_rotation_matrix(roll_rad: np.ndarray, pitch_rad: np.ndarray, yaw_rad: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    R = np.empty((roll_rad.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = cy * cp
    R[:, 0, 1] = cy * sp * sr - sy * cr
    R[:, 0, 2] = cy * sp * cr + sy * sr

    R[:, 1, 0] = sy * cp
    R[:, 1, 1] = sy * sp * sr + cy * cr
    R[:, 1, 2] = sy * sp * cr - cy * sr

    R[:, 2, 0] = -sp
    R[:, 2, 1] = cp * sr
    R[:, 2, 2] = cp * cr
    return R


def interp_att_to_imu(att: pd.DataFrame, imu_time_s: np.ndarray) -> pd.DataFrame:
    if att.empty or len(att) < 2:
        return pd.DataFrame({
            "roll_deg": np.zeros_like(imu_time_s),
            "pitch_deg": np.zeros_like(imu_time_s),
            "yaw_deg": np.zeros_like(imu_time_s),
        })

    att = att.drop_duplicates(subset=["time_s"]).sort_values("time_s")
    t = att["time_s"].to_numpy()
    roll = np.interp(imu_time_s, t, att["roll_deg"].to_numpy())
    pitch = np.interp(imu_time_s, t, att["pitch_deg"].to_numpy())
    yaw = np.interp(imu_time_s, t, att["yaw_deg"].to_numpy())
    return pd.DataFrame({"roll_deg": roll, "pitch_deg": pitch, "yaw_deg": yaw})


def trapz_integrate_vector(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(time_s) < 2:
        return out
    dt = np.diff(time_s)
    for i in range(1, len(time_s)):
        out[i] = out[i - 1] + 0.5 * (values[i] + values[i - 1]) * dt[i - 1]
    return out


def preprocess_imu(imu_raw: pd.DataFrame, att: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if imu_raw.empty:
        return pd.DataFrame(), {"warning": "No IMU messages found"}

    imu_raw = normalize_time(imu_raw)
    imu_raw, imu_instance = choose_primary_instance(imu_raw)

    accx_col = pick_first_existing(imu_raw, ["AccX"])
    accy_col = pick_first_existing(imu_raw, ["AccY"])
    accz_col = pick_first_existing(imu_raw, ["AccZ"])
    gyrx_col = pick_first_existing(imu_raw, ["GyrX"], required=False)
    gyry_col = pick_first_existing(imu_raw, ["GyrY"], required=False)
    gyrz_col = pick_first_existing(imu_raw, ["GyrZ"], required=False)
    ahz_col = pick_first_existing(imu_raw, ["AHz"], required=False)
    ghz_col = pick_first_existing(imu_raw, ["GHz"], required=False)

    imu = pd.DataFrame({
        "TimeUS": to_numeric_series(imu_raw, "TimeUS"),
        "time_s": to_numeric_series(imu_raw, "time_s"),
        "instance": to_numeric_series(imu_raw, "I", default=np.nan),
        "acc_x_mps2": to_numeric_series(imu_raw, accx_col),
        "acc_y_mps2": to_numeric_series(imu_raw, accy_col),
        "acc_z_mps2": to_numeric_series(imu_raw, accz_col),
        "gyr_x_rads": to_numeric_series(imu_raw, gyrx_col) if gyrx_col else np.nan,
        "gyr_y_rads": to_numeric_series(imu_raw, gyry_col) if gyry_col else np.nan,
        "gyr_z_rads": to_numeric_series(imu_raw, gyrz_col) if gyrz_col else np.nan,
        "acc_rate_reported_hz": to_numeric_series(imu_raw, ahz_col) if ahz_col else np.nan,
        "gyr_rate_reported_hz": to_numeric_series(imu_raw, ghz_col) if ghz_col else np.nan,
    }).dropna(subset=["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]).sort_values("TimeUS").reset_index(drop=True)

    if imu.empty:
        return imu, {"warning": "No valid IMU samples after filtering", "imu_instance": imu_instance}

    dt = imu["time_s"].diff()
    imu_fs_hz = float(1.0 / np.nanmedian(dt.to_numpy())) if len(imu) > 1 else np.nan

    att_sync = interp_att_to_imu(att, imu["time_s"].to_numpy())
    imu = pd.concat([imu, att_sync], axis=1)

    roll_rad = np.radians(imu["roll_deg"].to_numpy())
    pitch_rad = np.radians(imu["pitch_deg"].to_numpy())
    yaw_rad = np.radians(imu["yaw_deg"].to_numpy())

    acc_body = imu[["acc_x_mps2", "acc_y_mps2", "acc_z_mps2"]].to_numpy(dtype=np.float64)
    R = body_to_nav_rotation_matrix(roll_rad, pitch_rad, yaw_rad)
    acc_nav = np.einsum("nij,nj->ni", R, acc_body)

    acc_nav_lin = acc_nav.copy()
    acc_nav_lin[:, 2] -= GRAVITY_MPS2

    vel_nav = trapz_integrate_vector(imu["time_s"].to_numpy(), acc_nav_lin)

    imu["acc_body_norm_mps2"] = np.linalg.norm(acc_body, axis=1)
    imu["acc_lin_x_mps2"] = acc_nav_lin[:, 0]
    imu["acc_lin_y_mps2"] = acc_nav_lin[:, 1]
    imu["acc_lin_z_mps2"] = acc_nav_lin[:, 2]
    imu["acc_lin_norm_mps2"] = np.linalg.norm(acc_nav_lin, axis=1)

    imu["vel_est_x_mps"] = vel_nav[:, 0]
    imu["vel_est_y_mps"] = vel_nav[:, 1]
    imu["vel_est_z_mps"] = vel_nav[:, 2]
    imu["vel_est_norm_mps"] = np.linalg.norm(vel_nav, axis=1)

    meta = {
        "imu_instance": imu_instance,
        "imu_samples_total": int(len(imu_raw)),
        "imu_samples_valid": int(len(imu)),
        "imu_sampling_hz_empirical": imu_fs_hz,
        "imu_sampling_hz_reported_acc": float(np.nanmedian(imu["acc_rate_reported_hz"])) if "acc_rate_reported_hz" in imu else np.nan,
        "imu_sampling_hz_reported_gyr": float(np.nanmedian(imu["gyr_rate_reported_hz"])) if "gyr_rate_reported_hz" in imu else np.nan,
    }
    return imu, meta


def compute_metrics(gps: pd.DataFrame, imu: pd.DataFrame) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not gps.empty:
        metrics.update({
            "distance_m": float(gps["seg_m"].sum()),
            "flight_time_s": float(gps["time_s"].iloc[-1] - gps["time_s"].iloc[0]),
            "max_horizontal_speed_mps": float(gps["speed_mps"].max()) if "speed_mps" in gps else np.nan,
            "max_vertical_speed_mps": float(np.abs(gps["vz_mps"]).max()) if "vz_mps" in gps else np.nan,
            "altitude_gain_m": float(gps["alt_m"].max() - gps["alt_m"].iloc[0]),
            "horizontal_displacement_m": float(np.hypot(gps["east_m"].iloc[-1], gps["north_m"].iloc[-1])),
            "max_altitude_m": float(gps["alt_m"].max()),
            "start_altitude_m": float(gps["alt_m"].iloc[0]),
            "end_altitude_m": float(gps["alt_m"].iloc[-1]),
        })
    if not imu.empty:
        metrics.update({
            "max_acceleration_mps2": float(imu["acc_lin_norm_mps2"].max()),
            "max_integrated_speed_from_imu_mps": float(imu["vel_est_norm_mps"].max()),
        })
    return metrics


def process_bin_file(temp_bin_path: str | Path) -> Dict[str, Any]:
    tables = collect_message_tables(temp_bin_path, wanted_types=("GPS", "IMU", "ATT", "AHR2", "MODE", "MSG"))
    gps, gps_meta = preprocess_gps(tables.get("GPS", pd.DataFrame()))
    att, att_meta = preprocess_att(tables.get("ATT", pd.DataFrame()), tables.get("AHR2", pd.DataFrame()))
    imu, imu_meta = preprocess_imu(tables.get("IMU", pd.DataFrame()), att)
    metrics = compute_metrics(gps, imu)
    return {
        "gps": gps,
        "imu": imu,
        "att": att,
        "gps_meta": gps_meta,
        "imu_meta": imu_meta,
        "att_meta": att_meta,
        "metrics": metrics,
    }


# =========================
# UI helpers
# =========================

def fmt(v: float, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{digits}f}"


def metric_card(label: str, value: str, unit: str = "") -> str:
    unit_html = f"<span class='unit'>{unit}</span>" if unit else ""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}{unit_html}</div>
    </div>
    """


def make_summary_text(metrics: Dict[str, Any], gps_meta: Dict[str, Any], imu_meta: Dict[str, Any]) -> str:
    distance = metrics.get("distance_m", np.nan)
    flight_time = metrics.get("flight_time_s", np.nan)
    hspeed = metrics.get("max_horizontal_speed_mps", np.nan)
    vspeed = metrics.get("max_vertical_speed_mps", np.nan)
    again = metrics.get("altitude_gain_m", np.nan)

    parts = []
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


def build_3d_figure(gps: pd.DataFrame) -> go.Figure:
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
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Up (m)",
        ),
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_alt_speed_figure(gps: pd.DataFrame) -> go.Figure:
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


def build_imu_figure(imu: pd.DataFrame) -> go.Figure:
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


# =========================
# Streamlit page
# =========================

st.set_page_config(page_title="Drone Flight Analyzer", layout="wide")

st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: Inter, system-ui, sans-serif;
}
.block-container {
    max-width: 1280px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.hero {
    padding: 0.25rem 0 1rem 0;
}
.hero h1 {
    font-size: 3.2rem;
    line-height: 1.0;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #9CA3AF;
    margin-top: 0;
}
.success-box {
    background: rgba(34, 197, 94, 0.16);
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin: 1rem 0 1.5rem 0;
    font-size: 1.05rem;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px;
    margin-top: 8px;
    margin-bottom: 20px;
}
.metric-card {
    background: #0F172A;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    min-height: 130px;
}
.metric-label {
    color: #CBD5E1;
    font-size: 1rem;
    margin-bottom: 10px;
}
.metric-value {
    color: #F8FAFC;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
}
.unit {
    font-size: 1rem;
    color: #94A3B8;
    margin-left: 0.35rem;
}
.small-note {
    color: #94A3B8;
    font-size: 0.95rem;
    margin-top: -8px;
    margin-bottom: 20px;
}
.section-title {
    font-size: 2rem;
    font-weight: 700;
    margin-top: 24px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>Drone Flight Analyzer</h1>
    <p>Завантаж ArduPilot .BIN і одразу отримай метрики, 3D-траєкторію та акуратні графіки.</p>
</div>
""", unsafe_allow_html=True)

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
    f"""
    <div class="success-box">
        Оброблено рядків → GPS: {gps_meta.get("gps_samples_valid", 0)} | IMU: {imu_meta.get("imu_samples_valid", 0)}
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">Flight Metrics</div>', unsafe_allow_html=True)

cards_html = f"""
<div class="metric-grid">
    {metric_card("Distance", fmt(metrics.get("distance_m", np.nan)), "m")}
    {metric_card("Max Horizontal Speed", fmt(metrics.get("max_horizontal_speed_mps", np.nan)), "m/s")}
    {metric_card("Max Acceleration", fmt(metrics.get("max_acceleration_mps2", np.nan)), "m/s²")}
    {metric_card("Flight Time", fmt(metrics.get("flight_time_s", np.nan)), "s")}
    {metric_card("Max Vertical Speed", fmt(metrics.get("max_vertical_speed_mps", np.nan)), "m/s")}
    {metric_card("Altitude Gain", fmt(metrics.get("altitude_gain_m", np.nan)), "m")}
</div>
"""
st.markdown(cards_html, unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="small-note">
        GPS ~ {fmt(gps_meta.get("gps_sampling_hz", np.nan), 2)} Hz &nbsp;&nbsp;|&nbsp;&nbsp;
        IMU ~ {fmt(imu_meta.get("imu_sampling_hz_empirical", np.nan), 2)} Hz
    </div>
    """,
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
            st.dataframe(
                gps[["time_s", "lat_deg", "lon_deg", "alt_m", "speed_mps", "vz_mps", "east_m", "north_m", "up_m"]].round(4),
                use_container_width=True,
                height=360,
            )
    with d2:
        st.subheader("IMU table")
        if imu.empty:
            st.write("Немає IMU.")
        else:
            st.dataframe(
                imu[["time_s", "acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "acc_lin_norm_mps2", "vel_est_norm_mps"]].round(4),
                use_container_width=True,
                height=360,
            )

with tab4:
    st.subheader("Automatic summary")
    st.write(make_summary_text(metrics, gps_meta, imu_meta))

    summary_payload = {
        "metrics": metrics,
        "gps_meta": gps_meta,
        "imu_meta": imu_meta,
    }
    st.download_button(
        "Download JSON summary",
        data=json.dumps(summary_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_summary.json",
        mime="application/json",
    )

# Optional CSV downloads
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    if not gps.empty:
        st.download_button(
            "Download GPS CSV",
            data=gps.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_gps.csv",
            mime="text/csv",
        )
with c2:
    if not imu.empty:
        st.download_button(
            "Download IMU CSV",
            data=imu.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_imu.csv",
            mime="text/csv",
        )
with c3:
    st.download_button(
        "Download Metrics JSON",
        data=json.dumps(metrics, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_metrics.json",
        mime="application/json",
    )
