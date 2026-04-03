
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymavlink import DFReader

EARTH_RADIUS_M = 6371000.0
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GRAVITY_MPS2 = 9.80665


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


def collect_message_tables(
    bin_path: str | Path,
    wanted_types: Iterable[str] = ("GPS", "IMU", "ATT", "AHR2", "MODE", "MSG"),
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


def haversine_vec(
    lat1_deg: np.ndarray,
    lon1_deg: np.ndarray,
    lat2_deg: np.ndarray,
    lon2_deg: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(lat1_deg.astype(float))
    lon1 = np.radians(lon1_deg.astype(float))
    lat2 = np.radians(lat2_deg.astype(float))
    lon2 = np.radians(lon2_deg.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return EARTH_RADIUS_M * c


def geodetic_to_ecef(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def ecef_to_enu(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    alt0_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        })
    if not imu.empty:
        metrics.update({
            "max_acceleration_mps2": float(imu["acc_lin_norm_mps2"].max()),
            "max_integrated_speed_from_imu_mps": float(imu["vel_est_norm_mps"].max()),
        })
    return metrics


def process_bin_file(bin_path: str | Path) -> Dict[str, Any]:
    tables = collect_message_tables(bin_path, wanted_types=("GPS", "IMU", "ATT", "AHR2", "MODE", "MSG"))
    gps, gps_meta = preprocess_gps(tables.get("GPS", pd.DataFrame()))
    att, att_meta = preprocess_att(tables.get("ATT", pd.DataFrame()), tables.get("AHR2", pd.DataFrame()))
    imu, imu_meta = preprocess_imu(tables.get("IMU", pd.DataFrame()), att)
    metrics = compute_metrics(gps, imu)
    return {
        "gps": gps,
        "imu": imu,
        "att": att,
        "metrics": metrics,
        "gps_meta": gps_meta,
        "imu_meta": imu_meta,
        "att_meta": att_meta,
    }
