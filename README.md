# Raspberry Pi with Adafruit AS7341 Spectral Publisher to InfluxDB

Publishes **calibrated lux** and **relative spectral composition** across 9 bands (415–680 nm visible + ~910 nm NIR) to InfluxDB for visualization in Grafana.

## Features

- **Spectral accuracy improvements**: Responsivity correction, VIS8 normalized separately from NIR, minimum signal threshold
- **Performance optimizations**: Parallel HTTP writes, retry queue with budget-based flushing, cached calculations
- **Dark offset correction**: Temperature-compensated dark frame subtraction
- **Lux calibration**: Linear regression model with optional ridge regularization and K-fold cross-validation
- **Auto-ranging** (optional): Automatic gain adjustment to optimize dynamic range
- **Multi-endpoint support**: Parallel writes to multiple InfluxDB instances with retry queues

## Hardware Requirements

- Raspberry Pi (any model with I²C support)
- Adafruit AS7341 10-channel spectral sensor (I²C address 0x39)
- Cosine (diffuser) recommended for accurate lux measurements

## Software Requirements (Not Included)

- InfluxDB v1.x
- Grafana (optional, for visualization)

---

## Installation

### 1. Prerequisites (One-Time Setup)

Install required system packages and enable I²C:

```bash
sudo apt update
sudo apt install -y python3-full python3-venv git i2c-tools python3-libgpiod
```

Enable I²C interface:
```bash
sudo raspi-config
# Navigate to: Interface Options → I2C → Enable
```

Add your user to the `i2c` group:
```bash
sudo adduser $USER i2c
```

**Reboot** to apply changes:
```bash
sudo reboot
```

Verify the AS7341 sensor is detected:
```bash
i2cdetect -y 1
```
You should see `39` in the output grid (I²C address 0x39).

### 2. Clone Repository

```bash
cd ~
git clone https://github.com/fredrik-markussen/as7341-RPi-lightmeter.git
cd as7341-RPi-lightmeter
```

### 3. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

Verify the driver installation:
```bash
python3 -c "import adafruit_as7341; print('AS7341 driver OK')"
```

---

## Configuration

### 4. Edit Main Script Settings

Open [src/as7341_influx_nir.py](src/as7341_influx_nir.py) and configure:

```python
# Device identification (CHANGE THIS for each Pi)
DEVICE = "RPi-1"         # Unique name for this device
MEAS   = "LIGHT"         # InfluxDB measurement name

# Sensor settings (must match calibration settings)
ATIME_D = 15             # Integration time parameter (0-255)
ASTEP_D = 999            # Integration steps (0-65534)
GAIN_D  = Gain.GAIN_256X # Gain setting

# Measurement settings
AVG     = 5              # Frames to average per reading
PERIOD  = 60.0           # Seconds between measurements

# InfluxDB endpoints (can specify multiple for redundancy)
ENDPOINTS = [
    ("10.239.99.73", 8086, "AAB"),  # (host, port, database)
]

# Optional: Enable auto-ranging
AUTORANGE_ENABLE = False  # Set to True to enable automatic gain adjustment
```

**Important**: The sensor settings (`ATIME_D`, `ASTEP_D`, `GAIN_D`) must match those used during calibration (steps 5 & 6).

---

## Calibration (Required)

Calibration generates two files that correct for sensor-specific characteristics:

### 5. Dark Offset Calibration

Captures dark current offsets when the sensor is in complete darkness.

**Preparation**:
- Cover the sensor completely with an opaque cap or thick tape
- Turn off all lights in the room
- Or use your finger to cover the sensor

**Before running**, verify the settings in [src/as7341_dark_capture.py](src/as7341_dark_capture.py) match your main script:

```python
ATIME_D = 99              # Must match as7341_influx_nir.py
ASTEP_D = 999             # Must match as7341_influx_nir.py
GAIN_D  = Gain.GAIN_64X   # Must match as7341_influx_nir.py
```

Run the calibration:
```bash
source .venv/bin/activate
python3 src/as7341_dark_capture.py
```

This creates `as7341_dark.json` in the project root with dark offsets for all 9 channels.

### 6. Lux Calibration

Builds a linear regression model to convert spectral readings to lux.

**Preparation**:
- Reference lux meter (smartphone apps work but professional meters are better)
- 8–10 diverse lighting scenes (daylight, shade, LED, incandescent, different intensities)
- Place sensor and lux meter side-by-side, facing the same direction

Run the calibration:
```bash
source .venv/bin/activate
python3 src/as7341_calibrate_lux.py --samples 8 --avg 10
```

For each scene:
1. Position sensor and lux meter together
2. Press Enter to capture spectral data
3. Enter the lux reading from your reference meter
4. Move to next lighting scene and repeat

Optional arguments:
```bash
--gain GAIN_64X          # Gain setting (must match main script)
--atime 99               # ATIME value (must match main script)
--astep 999              # ASTEP value (must match main script)
--samples 8              # Number of calibration scenes
--avg 10                 # Captures per scene (median aggregated)
--ridge 0.01             # L2 regularization (reduces overfitting)
--kfold 5                # K-fold cross-validation
--nnls-lite              # Enforce non-negative coefficients
```

This creates `as7341_lux_cal.json` with calibration coefficients.

**Test the calibration** (optional):
```bash
# Run once and check if lux value looks reasonable
python3 src/as7341_influx_nir.py
# Press Ctrl+C after a few readings
```

---

## Running as a Service

### 7. Install systemd Service

Copy the service template (note: the template in repo may need path adjustments):

```bash
sudo cp systemd/as7341.service /etc/systemd/system/as7341@.service
```

**Edit the service file** to match your repository path:
```bash
sudo nano /etc/systemd/system/as7341@.service
```

Update the `WorkingDirectory` and `ExecStart` paths:
```ini
WorkingDirectory=%h/as7341-RPi-lightmeter
ExecStart=%h/as7341-RPi-lightmeter/.venv/bin/python %h/as7341-RPi-lightmeter/src/as7341_influx_nir.py
```

### 8. Enable and Start Service

Replace `<user>` with your Linux username (e.g., `pi`, `admin`):

```bash
sudo systemctl daemon-reload
sudo systemctl enable as7341@<user>.service
sudo systemctl start as7341@<user>.service
```

Check service status:
```bash
sudo systemctl status as7341@<user>.service
```

View real-time logs:
```bash
journalctl -u as7341@<user>.service -f
```

Restart after configuration changes:
```bash
sudo systemctl restart as7341@<user>.service
```

---

## Verification

### Check InfluxDB Measurements

List all measurements in the database:
```bash
curl -G http://<INFLUX_HOST>:8086/query \
  --data-urlencode "db=AAB" \
  --data-urlencode "q=SHOW MEASUREMENTS"
```

Query recent data for your device:
```bash
curl -G http://<INFLUX_HOST>:8086/query \
  --data-urlencode "db=AAB" \
  --data-urlencode "q=SELECT * FROM LIGHT WHERE Device='RPi-1' ORDER BY time DESC LIMIT 5"
```

### Data Structure in InfluxDB

**Spectral composition** (9 points per reading):
- **Measurement**: `LIGHT`
- **Tags**: `Device=RPi-1`, `wavelength_nm=415|445|480|515|555|590|630|680|910`
- **Field**: `rel_intensity` (normalized, VIS8 sum to 1.0; NIR as fraction of total)

**Lux measurement**:
- **Measurement**: `LIGHT_LUX`
- **Tags**: `Device=RPi-1`, `method=lin_basic`
- **Fields**: `lux` (calibrated), `clear` (raw CLEAR channel value)

---

## Troubleshooting

### Sensor Not Detected
```bash
# Check I²C is enabled
sudo raspi-config

# Verify user is in i2c group
groups $USER

# Check sensor address
i2cdetect -y 1
```

### Dark/Calibration Files Not Found
```bash
# Ensure files exist in project root
ls -la ~/as7341-RPi-lightmeter/*.json

# Check file paths in main script
grep "DARK_FILE\|CAL_FILE" src/as7341_influx_nir.py
```

### Saturation Warnings
```
[WARN] Near saturation: max=65000
```
- Lower the gain in [src/as7341_influx_nir.py](src/as7341_influx_nir.py) (e.g., `GAIN_256X` → `GAIN_64X`)
- Or reduce `ATIME_D` or `ASTEP_D`
- Recalibrate dark and lux after changing settings

### InfluxDB Connection Errors
```
[ERR] 10.239.99.73:8086/AAB: Connection refused
```
- Verify InfluxDB is running: `systemctl status influxdb`
- Check firewall rules
- Verify database exists: `influx -execute "SHOW DATABASES"`

---

## Additional Information

- **Integration time**: Calculated as `(ATIME + 1) × (ASTEP + 1) × 2.78 µs`
- **Full-scale ADC**: Max value is `min(65535, (ATIME + 1) × (ASTEP + 1))`
- **Responsivity correction**: Built-in correction factors account for different channel sensitivities
- **Retry queue**: Failed HTTP writes are queued (max 500) and retried on subsequent loops

For Grafana dashboards, query the `LIGHT` measurement with `wavelength_nm` filter for spectral plots, and `LIGHT_LUX` for lux trends.