# AS7341 Spectral Publisher (VIS8 + NIR) → InfluxDB

Pushes **relative spectral composition** across 9 bands (415–680 nm + ~910 nm NIR) and **calibrated lux** to InfluxDB; visualize in Grafana as an XY spectrum.

## Hardware
- Raspberry Pi with I²C enabled
- ams/Adafruit **AS7341** (addr 0x39)
- Cosine (bulk) diffuser recommended

## Quick Start
```bash
sudo apt update
sudo apt install -y python3-full python3-venv git i2c-tools
sudo raspi-config   # enable I2C (Interface Options → I2C → Enable)
```
Verify sensor is seen:
```bash
i2cdetect -y 1   # should show "0x39"
```
### Clone the repository
```
cd ~
git clone https://github.com/fredrik-markussen/as7341-RPi-lightmeter.git
cd as7341-RPi-lightmeter
```
### Create and activate venv
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Sanity check: 
```bash
python3 - << 'EOF'
import adafruit_as7341
print("Driver OK")
EOF
```

### Configure parameters (simple manual config section)
Edit as7341_influx_nir.py
```bash
MEAS           = "LIGHT"
DEVICE         = "RPi-NEW"     # <-- CHANGE FOR EACH PI
ATIME_D        = 99
ASTEP_D        = 999
GAIN_D         = Gain.GAIN_64X
ENDPOINTS = [
    ("10.239.99.73", 8086, "AAB"),
]
```

### Generate dark + calibration files (REQUIRED per device)
or use include files for simple test

Sensor in total darkness (finger over + turn off lights):
```bash
source .venv/bin/activate
python3 src/as7341_dark_capture.py
```

Collect 8–10 lighting scenes with a lux meter present:
```bash
python3 src/as7341_calibrate_lux.py
```

This will write:
as7341_dark.json
as7341_lux_cal.json


into project root.

### Install & enable systemd service

Copy systemd template:
```bash
sudo cp systemd/as7341@.service /etc/systemd/system/
sudo systemctl daemon-reload
```

Start service for this Linux user
```bash
sudo systemctl enable --now as7341@admin.service
```



### Verification

Check if measurement exists
```bash
curl -G http://<HOST>:8086/query \
  --data-urlencode "db=AAB" \
  --data-urlencode "q=SHOW MEASUREMENTS"
  ```

Check for your specific device
```bash
curl -G http://<HOST>:8086/query \
  --data-urlencode "db=AAB" \
  --data-urlencode "q=SELECT * FROM LIGHT WHERE Device='RPi-1' ORDER BY time DESC LIMIT 5"
   ```