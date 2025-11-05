# AS7341 Spectral Publisher (VIS8 + NIR) → InfluxDB

Pushes **relative spectral composition** across 9 bands (415–680 nm + ~910 nm NIR) and **calibrated lux** to InfluxDB; visualize in Grafana as an XY spectrum.

## Hardware
- Raspberry Pi with I²C enabled
- ams/Adafruit **AS7341** (addr 0x39)
- Cosine (bulk) diffuser recommended

### Prereqs (once per Pi)
```bash
sudo apt update
sudo apt install -y python3-full python3-venv git i2c-tools python3-libgpiod
sudo raspi-config   # Interface Options → I2C → Enable
sudo adduser $USER i2c
# Reboot to apply I2C group + raspi-config changes
sudo reboot
```

Verify sensor is seen:
```bash
i2cdetect -y 1   # should show "0x39"
```
### Clone repository
```bash
cd ~
git clone https://github.com/fredrik-markussen/as7341-RPi-lightmeter.git
cd as7341-RPi-lightmeter
```

### Create and activate venv
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip wheel setuptools
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pip install RPi.GPIO==0.7.1
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
```
### Ensure the user is in the i2c group
sudo adduser *** i2c


Start service for this Linux user
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now as7341@admin.service
sudo systemctl restart as7341@admin.service
sudo systemctl status as7341@admin.service
# Logs
journalctl -u as7341@admin.service -f
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