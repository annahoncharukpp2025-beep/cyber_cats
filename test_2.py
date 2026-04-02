from pymavlink import mavutil
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


#Парсування бінарних логів
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

        #GPS дані
        if msg_type.startswith("GPS"):
            if all(k in data for k in ["Lat", "Lng", "Alt"]):
                gps_data.append({
                    "time": data.get("TimeUS", 0) / 1e6,
                    "lat": data["Lat"] / 1e7,
                    "lon": data["Lng"] / 1e7,
                    "alt": data["Alt"] / 1000,
                    "speed": data.get("Spd", 0)
                })

        #IMU дані
        elif "IMU" in msg_type:
            if all(k in data for k in ["AccX", "AccY", "AccZ"]):
                imu_data.append({
                    "time": data.get("TimeUS", 0) / 1e6,
                    "ax": data["AccX"],
                    "ay": data["AccY"],
                    "az": data["AccZ"]
                })

    gps_df = pd.DataFrame(gps_data)
    imu_df = pd.DataFrame(imu_data)

    return gps_df, imu_df


#Функція Haversine
R = 6371000 #Середній радіус Землі

def haversine(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def total_distance(gps_df):
    dist = 0
    for i in range(1, len(gps_df)):
        dist += haversine(
            gps_df.lat.iloc[i - 1],
            gps_df.lon.iloc[i - 1],
            gps_df.lat.iloc[i],
            gps_df.lon.iloc[i]
        )
    return dist


#Трапецієвидне інтегрування
def integrate_acceleration(acc, time):
    v = [0]
    for i in range(1, len(acc)):
        dt = time[i] - time[i - 1]
        vi = v[-1] + (acc[i] + acc[i - 1]) / 2 * dt
        v.append(vi)
    return np.array(v)



#Основна аналітика
def compute_metrics(gps_df, imu_df):
    if gps_df.empty or imu_df.empty:
        return {"error": "GPS або IMU дані відсутні"}

    #Дистанція
    distance = total_distance(gps_df)

    #Час польоту
    flight_time = gps_df.time.iloc[-1] - gps_df.time.iloc[0]

    #Вертикальна швидкість
    dz = np.diff(gps_df.alt)
    dt = np.diff(gps_df.time)
    vertical_speed = dz / dt
    max_vertical_speed = np.max(np.abs(vertical_speed))

    #Горизонтальна швидкість
    horizontal_speed = []
    for i in range(1, len(gps_df)):
        d = haversine(
            gps_df.lat.iloc[i - 1],
            gps_df.lon.iloc[i - 1],
            gps_df.lat.iloc[i],
            gps_df.lon.iloc[i]
        )
        dt_i = gps_df.time.iloc[i] - gps_df.time.iloc[i - 1]
        horizontal_speed.append(d / dt_i)
    horizontal_speed = np.array(horizontal_speed)
    max_horizontal_speed = np.max(horizontal_speed)

    #Прискорення
    acc_mag = np.sqrt(imu_df.ax ** 2 + imu_df.ay ** 2 + imu_df.az ** 2)
    max_acc = np.max(acc_mag)

    #Швидкість з IMU (інтегрування)
    velocity_from_imu = integrate_acceleration(acc_mag.values, imu_df.time.values)

    #Максимальний набір висоти
    max_alt_gain = gps_df.alt.max() - gps_df.alt.min()

    return {
        "distance (m)": distance,
        "flight_time (s)": flight_time,
        "max_horizontal_speed (m/s)": max_horizontal_speed,
        "max_vertical_speed (m/s)": max_vertical_speed,
        "max_acceleration (m/s^2)": max_acc,
        "max_altitude_gain (m)": max_alt_gain
    }


#Конвертація з WGS-84 у ENU
def wgs84_to_enu(gps_df):
    lat0 = gps_df.lat.iloc[0]
    lon0 = gps_df.lon.iloc[0]
    alt0 = gps_df.alt.iloc[0]

    dlat = np.radians(gps_df.lat - lat0)
    dlon = np.radians(gps_df.lon - lon0)


    x = dlon * R * np.cos(np.radians(lat0))  #Схід
    y = dlat * R                             #Північ
    z = gps_df.alt - alt0                    #Верх

    return x, y, z



#3D візуалізація
def plot_3d_trajectory(gps_df):

    pio.renderers.default = "browser"

    if len(gps_df) < 2:
        print("Недостатньо точок")
        return

    x, y, z = wgs84_to_enu(gps_df)

    speed = [0]
    for i in range(1, len(gps_df)):
        d = haversine(
            gps_df.lat.iloc[i - 1],
            gps_df.lon.iloc[i - 1],
            gps_df.lat.iloc[i],
            gps_df.lon.iloc[i]
        )
        dt = gps_df.time.iloc[i] - gps_df.time.iloc[i - 1]
        speed.append(d / dt if dt > 0 else 0)

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(color=speed, colorscale='Earth', width=5)
    )])

    fig.show()

#Обробка файлів
files = [
    "00000001.BIN",
    "00000019.BIN"
]

for file in files:
    print(f"\n\tFILE: {file}")

    gps_df, imu_df = parse_log(file)

    print("GPS DATA:")
    print(gps_df.head())

    print("IMU DATA:")
    print(imu_df.head())

    metrics = compute_metrics(gps_df, imu_df)
    print("\tFlight Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

    plot_3d_trajectory(gps_df)