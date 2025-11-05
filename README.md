# AS7341 Spectral Publisher (VIS8 + NIR) → InfluxDB

Pushes **relative spectral composition** across 9 bands (415–680 nm + ~910 nm NIR) and **calibrated lux** to InfluxDB; visualize in Grafana as an XY spectrum.

## Hardware
- Raspberry Pi with I²C enabled
- ams/Adafruit **AS7341** (addr 0x39)
- Cosine (bulk) diffuser recommended

## Quick Start
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Dark capture (fully cover sensor)
python3 src/as7341_dark_capture.py && mv as7341_dark.json config/
# Lux calibration (with lux meter; diverse scenes)
python3 src/as7341_calibrate_lux.py --avg 15 --samples 12 --kfold 5 --ridge 0.1 && mv as7341_lux_cal.json config/
# Run publisher
python3 src/as7341_influx_nir.py# as7341-RPi-lightmeter
