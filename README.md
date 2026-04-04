# Drone Flight Analyzer

An interactive web application for analyzing ArduPilot DataFlash `.BIN` flight logs. The system automatically parses the binary log, extracts GPS and IMU data, calculates key flight metrics, and displays the results in a web interface.

## Project Launch

### 1. Navigate to the project folder

cd path/to/your/project

### 2. Create a virtual environment

#### Linux / macOS

python3 -m venv .venv

#### Windows

python -m venv .venv

### 3. Activate the virtual environment

#### Linux / macOS

source .venv/bin/activate

#### Windows

.venv\Scripts\activate

### 4. Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

### 5. Create the API key file

Create the `api_key.toml` file from `.env.example.toml`, and paste your key there

### 6. Run the application

streamlit run app.py

### 7. Open the browser

After launch, Streamlit will show a local address, usually:

http://localhost:8501

### 8. Upload the `.BIN` file

In the opened interface:
- click **Upload ArduPilot .BIN**
- select the log file
- wait until processing is completed
- view the metrics, charts, and 3D trajectory


---

## What the project does

The system performs the following actions:

- reads ArduPilot DataFlash `.BIN` directly
- extracts GPS, IMU, ATT messages
- filters invalid GPS points
- determines GPS and IMU sampling frequencies
- converts coordinates from the global WGS-84 system into the local ENU system
- calculates total distance using haversine
- estimates speed from IMU using trapezoidal integration
- displays results in the form of metrics, charts, tables, and 3D visualization

## Why this stack was chosen

This task required a stack that would allow a full working prototype to be implemented quickly: binary log parsing, numerical telemetry processing, chart generation, and web interface creation. Python is well suited for data analysis tasks, `pymavlink` provides access to ArduPilot logs, and `Streamlit + Plotly` make it possible to quickly create a convenient interface for demonstrating results.

## Main metrics

After processing the log file, the system automatically determines:

- total flight distance
- flight duration
- maximum horizontal speed
- maximum vertical speed
- maximum acceleration
- maximum altitude gain

---

## Architecture

### `app.py`

The Streamlit interface file. Responsible for:
- file upload
- displaying metrics
- building charts
- displaying tables
- summary
- export buttons

### `telemetry_core.py`

The file with the main telemetry processing logic. Responsible for:
- reading `.BIN`
- extracting GPS / IMU / ATT
- time series normalization
- GPS filtering
- coordinate conversion WGS-84 → ENU
- metric calculation

---

## Technologies used

- **Python** — main development language
- **Streamlit** — web interface creation
- **pymavlink** — working with ArduPilot DataFlash `.BIN`
- **pandas** — tabular data processing
- **numpy** — numerical calculations
- **plotly** — interactive charts and 3D visualization

---

## Working principle

### 1. Parsing

The `.BIN` file is read directly through `pymavlink`, after which the following messages are extracted from it:
- `GPS`
- `IMU`
- `ATT`

### 2. GPS

GPS data:
- is cleaned from invalid points
- is used to calculate distance, speed, and altitude
- is converted into local **ENU** coordinates

### 3. IMU

IMU data:
- is taken from acceleration fields along the axes
- is synchronized with `ATT` orientation data
- is transformed from body frame into navigation frame
- compensates for gravity
- is integrated using the trapezoidal method to estimate speed

### 4. Visualization

After processing, the results are displayed in the web interface as:
- metric cards
- charts
- 3D trajectory
- GPS and IMU tables