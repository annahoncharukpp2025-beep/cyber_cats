import streamlit as st
import pandas as pd
import numpy as np
from pymavlink import mavutil
import plotly.graph_objects as go
import google.generativeai as genai
import tomllib

# --- ФУНКЦІЇ ПАРСИНГУ ТА ОБЧИСЛЕНЬ ---
def parse_log(file_path):
    log = mavutil.mavlink_connection(file_path)
    gps_data = []
    imu_data = []
    while True:
        msg = log.recv_match(blocking=True)
        if msg is None:
            break
        msg_type = msg.get_type()
        data = msg.to_dict()

        if msg_type.startswith("GPS") and all(k in data for k in ["Lat", "Lng", "Alt"]):
            gps_data.append({
                "time": data.get("TimeUS", 0) / 1e6,
                "lat": data["Lat"] / 1e7,
                "lon": data["Lng"] / 1e7,
                "alt": data["Alt"] / 1000,
                "speed": data.get("Spd", 0)
            })
        elif "IMU" in msg_type and all(k in data for k in ["AccX", "AccY", "AccZ"]):
            imu_data.append({
                "time": data.get("TimeUS", 0) / 1e6,
                "ax": data["AccX"],
                "ay": data["AccY"],
                "az": data["AccZ"]
            })
    return pd.DataFrame(gps_data), pd.DataFrame(imu_data)

R = 6371000 # Середній радіус Землі

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def total_distance(gps_df):
    dist = 0
    for i in range(1, len(gps_df)):
        dist += haversine(gps_df.lat.iloc[i - 1], gps_df.lon.iloc[i - 1], gps_df.lat.iloc[i], gps_df.lon.iloc[i])
    return dist

def integrate_acceleration(acc, time):
    v = [0]
    for i in range(1, len(acc)):
        dt = time[i] - time[i - 1]
        vi = v[-1] + (acc[i] + acc[i - 1]) / 2 * dt
        v.append(vi)
    return np.array(v)

def compute_metrics(gps_df, imu_df):
    if gps_df.empty or imu_df.empty:
        return {"error": "GPS або IMU дані відсутні"}

    distance = total_distance(gps_df)
    flight_time = gps_df.time.iloc[-1] - gps_df.time.iloc[0]
    
    dz = np.diff(gps_df.alt)
    dt = np.diff(gps_df.time)
    vertical_speed = dz / dt
    max_vertical_speed = np.max(np.abs(vertical_speed))

    horizontal_speed = []
    for i in range(1, len(gps_df)):
        d = haversine(gps_df.lat.iloc[i - 1], gps_df.lon.iloc[i - 1], gps_df.lat.iloc[i], gps_df.lon.iloc[i])
        dt_i = gps_df.time.iloc[i] - gps_df.time.iloc[i - 1]
        horizontal_speed.append(d / dt_i if dt_i > 0 else 0)
    
    max_horizontal_speed = np.max(np.array(horizontal_speed))
    acc_mag = np.sqrt(imu_df.ax ** 2 + imu_df.ay ** 2 + imu_df.az ** 2)
    max_acc = np.max(acc_mag)
    max_alt_gain = gps_df.alt.max() - gps_df.alt.min()

    # Повертаємо ключі англійською, як було у тебе спочатку
    return {
        "distance (m)": distance,
        "flight_time (s)": flight_time,
        "max_horizontal_speed (m/s)": max_horizontal_speed,
        "max_vertical_speed (m/s)": max_vertical_speed,
        "max_acceleration (m/s^2)": max_acc,
        "max_altitude_gain (m)": max_alt_gain
    }

def wgs84_to_enu(gps_df):
    lat0, lon0, alt0 = gps_df.lat.iloc[0], gps_df.lon.iloc[0], gps_df.alt.iloc[0]
    dlat = np.radians(gps_df.lat - lat0)
    dlon = np.radians(gps_df.lon - lon0)
    x = dlon * R * np.cos(np.radians(lat0)) 
    y = dlat * R 
    z = gps_df.alt - alt0 
    return x, y, z

def plot_3d_trajectory(gps_df):
    if len(gps_df) < 2:
        return None
    x, y, z = wgs84_to_enu(gps_df)
    speed = [0]
    for i in range(1, len(gps_df)):
        d = haversine(gps_df.lat.iloc[i - 1], gps_df.lon.iloc[i - 1], gps_df.lat.iloc[i], gps_df.lon.iloc[i])
        dt = gps_df.time.iloc[i] - gps_df.time.iloc[i - 1]
        speed.append(d / dt if dt > 0 else 0)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=speed, colorscale='Earth', width=5)
    )])
    return fig

# --- ФУНКЦІЯ ШІ (Має бути ДО основного коду UI) ---
def analyze_flight_with_llm(metrics_dict):
    st.subheader(" AI-Експерт: Аналіз місії (LLM)")
    
    try:
        # Правильний спосіб прочитати твій файл .toml у Python
        with open("api_key.toml", "rb") as f:
            config = tomllib.load(f)
            api_key = config["GEMINI_API_KEY"]
            
        genai.configure(api_key=api_key)
    except FileNotFoundError:
        st.warning("Файл api_key.toml не знайдено у папці з проєктом.")
        return
    except Exception as e:
        st.warning(f"Помилка читання ключа. Перевір формат api_key.toml. Деталі: {e}")
        return

    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""
    Ти — суворий експерт з авіаційної безпеки та розслідування інцидентів з БПЛА (дронами).
    Твоє завдання: проаналізувати метрики польоту з бортового самописця Ardupilot і написати короткий звіт (3-4 речення) у форматі - 
    кожний показник виведи з нового рядка із його коротким описом і в кінці виведи загальний висновок.
    
    Ось дані цього польоту:
    - Пройдена дистанція: {metrics_dict['distance (m)']:.2f} метрів
    - Час польоту: {metrics_dict['flight_time (s)']:.1f} секунд
    - Макс. горизонтальна швидкість: {metrics_dict['max_horizontal_speed (m/s)']:.2f} м/с
    - Макс. вертикальна швидкість: {metrics_dict['max_vertical_speed (m/s)']:.2f} м/с
    - Макс. прискорення (перевантаження): {metrics_dict['max_acceleration (m/s^2)']:.2f} м/с²
    - Загальний набір висоти: {metrics_dict['max_altitude_gain (m)']:.2f} метрів

    Правила аналізу:
    1. Якщо прискорення > 30 м/с², це можливе зіткнення або дуже жорстка посадка. Вкажи на це!
    2. Якщо вертикальна швидкість > 10 м/с, це може бути різке падіння (штопор).
    3. Зроби висновок про загальну плавність та безпеку польоту.
    4. Відповідай українською мовою, професійним тоном.
    """

    with st.spinner("ШІ аналізує телеметрію..."):
        try:
            response = model.generate_content(prompt)
            st.info(response.text)
        except Exception as e:
            st.error(f"Помилка при зв'язку з LLM: {e}")

# --- ОСНОВНИЙ UI STREAMLIT ---
st.title(" Drone Flight Analyzer")

file = st.file_uploader("Upload ArduPilot .BIN", type=["bin"])

if file:
    with open("temp.bin", "wb") as f:
        f.write(file.read())

    gps, imu = parse_log("temp.bin")
    st.success(f"Оброблено рядків -> GPS: {len(gps)} | IMU: {len(imu)}")

    if len(gps) > 0 and len(imu) > 0:
        st.subheader(" Flight Metrics")
        m = compute_metrics(gps, imu)

        if m and "error" not in m:
            # Створюємо 2 ряди по 3 колонки для метрик
            col1, col2, col3 = st.columns(3)
            col1.metric("Distance (m)", f"{m['distance (m)']:.2f}")
            col2.metric("Max Horizontal Speed", f"{m['max_horizontal_speed (m/s)']:.2f}")
            col3.metric("Max Acceleration", f"{m['max_acceleration (m/s^2)']:.2f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Flight Time (s)", f"{m['flight_time (s)']:.2f}")
            col5.metric("Max Vertical Speed", f"{m['max_vertical_speed (m/s)']:.2f}")
            col6.metric("Altitude Gain", f"{m['max_altitude_gain (m)']:.2f}")

            # Викликаємо ШІ-аналіз, передаючи словник m
            st.divider()
            analyze_flight_with_llm(m)
        else:
            st.warning("Недостатньо даних для обчислення метрик.")

        st.subheader(" 3D Trajectory")
        fig = plot_3d_trajectory(gps)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Недостатньо точок для 3D графіка.")

        # Спойлери з сирими даними
        with st.expander("Показати сирі дані GPS"):
            st.dataframe(gps.head(10))
        with st.expander("Показати сирі дані IMU"):
            st.dataframe(imu.head(10))