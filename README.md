# Raspberry Pi AS7341 Spectral Light Meter → InfluxDB

Publishes calibrated **lux** and **relative spectral composition** across
9 bands (415–680 nm visible + ~910 nm NIR) from a Raspberry Pi to InfluxDB,
suitable for Grafana dashboarding and field measurements.

The codebase is centered on two scripts:

- `src/as7341_influx_nir.py` — the measurement service (run continuously by systemd).
- `src/as7341_calibrate.py` — the guided 3-phase calibration tool.

A short functional spec lives in [FSD.md](FSD.md).

## Features

- Two-preset auto-sensitivity (HI: gain ×256 / 50 ms for dim, LO: gain ×16 /
  10 ms for bright) with hysteresis to avoid flapping.
- Per-preset dark calibration and per-preset VIS8 lux calibration.
- Guided 3-phase calibration script that walks through dark, spectral
  responsivity, and lux against a Seconic C-7000 + CoolLED pE-4000.
- Multi-endpoint InfluxDB fan-out with parallel writes, per-endpoint retry
  queues, and a CSV archive fallback for prolonged outages.
- `.env`-driven configuration so the script does not need editing per device.

## Hardware

- Raspberry Pi (any model with I²C).
- Adafruit AS7341 10-channel spectral sensor (I²C @ 0x39).
- Cosine diffuser dome (recommended for accurate lux).
- Light-tight cap for dark-frame capture.

## Software prerequisites

- Raspberry Pi OS with I²C enabled.
- Python 3.11+ in a virtualenv.
- An InfluxDB v1.x endpoint accepting line protocol on `/write`.

---

## Installation

### 1. System packages and I²C

```bash
sudo apt update
sudo apt install -y python3-full python3-venv git i2c-tools python3-libgpiod
sudo raspi-config        # Interface Options → I2C → Enable
sudo adduser $USER i2c
sudo reboot
```

Verify the sensor:

```bash
i2cdetect -y 1           # expect address 0x39 in the grid
```

### 2. Clone and install

```bash
cd ~
git clone https://github.com/fredrik-markussen/as7341-RPi-lightmeter.git
cd as7341-RPi-lightmeter
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
python3 -c "import adafruit_as7341; print('AS7341 driver OK')"
```

---

## Configuration

All runtime configuration lives in a `.env` file at the project root. Copy the
sample and edit:

```bash
cp config/sample.env .env
$EDITOR .env
```

Keys you typically want to set:

| Key                  | Meaning |
|----------------------|---------|
| `DEVICE`             | Tag value used in InfluxDB (`Device=...`); make it unique per Pi. |
| `INFLUX_ENDPOINTS`   | JSON array of `[host, port, database]` triples. Multiple endpoints write in parallel. |
| `AVG`                | Frames averaged per measurement (5 is a good default). |
| `PERIOD`             | Seconds between measurements. |
| `AUTORANGE_*`        | HI/LO switching thresholds and hysteresis count. |
| `SENS_HI_*`, `SENS_LO_*` | Override gain / integration time for either preset. |

See `config/sample.env` for the full list. Anything not set in `.env` falls
back to the defaults at the top of `src/as7341_influx_nir.py`.

---

## Calibration (required before first run)

Calibration generates per-preset JSON files in the project root. The
measurement script will not start until at least the HI lux cal file exists.

Run the guided script with the sensor and reference instruments connected:

```bash
source .venv/bin/activate
python3 src/as7341_calibrate.py
```

This walks through three phases:

### Phase 1 — Dark capture

- Sensor covered with an opaque cap, room dark.
- 100 samples per preset; medians written with full meta (gain, ATIME, ASTEP, timestamp).
- Outputs: `as7341_dark_hi.json`, `as7341_dark_lo.json`.

### Phase 2 — Spectral responsivity (VIS8)

- AS7341 + Seconic C-7000 side-by-side under a CoolLED pE-4000.
- 5 intensity levels by default (10 / 25 / 50 / 75 / 90 %).
- C-7000 spectral data either pasted from a 2-column CSV
  (`wavelength_nm, irradiance_W_m2_nm`) or entered manually at the 8 channel
  wavelengths.
- Output: `as7341_responsivity.json` with two blocks:
  - `corrections` — per-channel multipliers normalised to 555 nm = 1.0,
    used to correct the relative spectrum (`rel_intensity`).
  - `responsivity_BC_per_W_m2_nm` — per-channel absolute responsivity in
    BasicCounts per W/m²/nm. When present, the runtime emits absolute
    spectral irradiance per VIS8 channel (`irr_*` CSV columns,
    `irradiance` Influx field).
  The main script picks both up automatically at startup; without the file,
  datasheet defaults are used for `corrections` and absolute irradiance is
  not emitted.
- NIR (~910 nm) is **not** measured here — the C-7000 covers 380–780 nm only.
  The runtime keeps a datasheet default for NIR composition correction
  (overridable via a `nir` key under `corrections`); absolute NIR irradiance
  is never emitted.

### Phase 3 — Lux model

- AS7341 + C-7000 side-by-side, same angle, no shadows.
- Default 12 diverse scenes per preset (≥ 10 needed for a stable fit).
- VIS8 ridge regression with intercept (α=0.01); MAD outlier rejection;
  K-fold CV reports goodness-of-fit. `--lux-nnls` constrains weights to be
  non-negative if the unconstrained fit goes wild.
- Outputs: `as7341_lux_cal_hi.json`, `as7341_lux_cal_lo.json`.

### Running individual phases

```bash
python3 src/as7341_calibrate.py --phase dark            # Phase 1 only
python3 src/as7341_calibrate.py --phase responsivity    # Phase 2 only
python3 src/as7341_calibrate.py --phase lux             # Phase 3 only
python3 src/as7341_calibrate.py --phase lux --preset hi # re-run HI lux only
```

Useful flags for Phase 3:

```
--lux-scenes 15        # collect more scenes (default 12)
--lux-ridge 0.0        # plain OLS instead of ridge (only with plenty of scenes)
--lux-nnls             # constrain weights to be non-negative
--lux-kfold 5          # K-fold CV folds
```

The standalone `src/as7341_dark_capture.py` is a smaller utility for ad-hoc
dark captures with custom settings; it defaults to writing
`as7341_dark_hi.json`. For the standard workflow, prefer the guided script.

---

## Running

### Foreground

```bash
source .venv/bin/activate
python3 src/as7341_influx_nir.py
```

You should see startup lines listing the endpoints, the active sensitivity
preset, ATIME/ASTEP, and the offline buffer location, followed by per-sample
log lines.

### As a systemd service

```bash
sudo cp systemd/as7341.service /etc/systemd/system/as7341@.service
sudo systemctl daemon-reload
sudo systemctl enable --now as7341@$USER.service
journalctl -u as7341@$USER.service -f
```

The unit assumes the repo lives at `~/as7341-RPi-lightmeter` with a `.venv`
directory. Edit `WorkingDirectory` / `ExecStart` if your layout differs.

---

## Data model

**Spectral composition** — measurement `LIGHT`, 9 points per cycle:

- Tags: `Device=<DEVICE>`, `wavelength_nm=415|445|480|515|555|590|630|680|910`.
- Fields:
  - `rel_intensity` — relative composition (VIS8 sums to 1.0; the NIR point
    is the fraction of corrected VIS+NIR energy).
  - `irradiance` — absolute spectral irradiance in W/m²/nm at the channel
    center. Emitted on the 8 VIS points only, and only after Phase 2 has
    produced an absolute responsivity calibration. Omitted from NIR (the
    C-7000 reference does not reach 910 nm).

**Lux** — measurement `LIGHT_LUX`, 1 point per cycle:

- Tags: `Device=<DEVICE>`, `method=lin_basic`.
- Fields: `lux` (calibrated), `clear` (raw CLEAR channel).

**CSV offline buffer** — same data, 21 columns:

```
timestamp_iso,device,lux,clear,
rel_415,rel_445,rel_480,rel_515,rel_555,rel_590,rel_630,rel_680,rel_nir,
irr_415,irr_445,irr_480,irr_515,irr_555,irr_590,irr_630,irr_680
```

The `irr_*` cells (W/m²/nm at channel center) are populated only when Phase 2
has produced an absolute responsivity; otherwise they are emitted empty so
the column count stays stable.

Quick sanity query:

```bash
curl -G http://<INFLUX_HOST>:8086/query \
  --data-urlencode "db=AAB" \
  --data-urlencode "q=SELECT * FROM LIGHT_LUX WHERE Device='RPi-1' ORDER BY time DESC LIMIT 5"
```

---

## Offline buffering

Failed writes go into per-endpoint in-memory retry queues (bounded by
`MAX_RETRY_QUEUE`, default 500) and replay when the endpoint recovers.

If **every** endpoint fails on a given sample, the row is also appended to a
10-minute tmp CSV in `~/Documents/Lightmeter_csv_out/daily_tmp/`. Tmp files
are merged into per-day aggregates in the parent directory; on startup, any
leftover tmps from a previous run are merged. The CSV is archive-only — it
is not automatically replayed to InfluxDB. For a prolonged outage that
exceeds the retry queue, import the daily CSVs manually after recovery.

Filenames and CSV row timestamps are both UTC, so daily files group by UTC
date regardless of the Pi's local timezone.

---

## Field operation: clock sync without RTC

A stock Raspberry Pi has no battery-backed RTC. At boot it restores the last
saved time from `fake-hwclock` and only corrects to wall-clock time once it
sees an NTP server. For field measurements that means CSV timestamps are
only as good as the last sync — a cold boot with no network leaves rows
stamped with the previous shutdown's time until NTP catches up.

**Recommended pre-deployment routine:**

1. Pre-configure the Pi to auto-connect to a phone hotspot (and optionally
   your home Wi-Fi). On Raspberry Pi OS Bookworm:
   ```bash
   sudo nmcli device wifi connect "<hotspot SSID>" password "<password>"
   ```
   Repeat for each network you want it to remember; NetworkManager keeps the
   profiles and reconnects automatically when in range.
2. In the field, power up the Pi within range of the configured hotspot.
3. Verify NTP sync — `timedatectl` should report `NTP service: active` and
   `System clock synchronized: yes`.
4. Disconnect the hotspot. The Pi keeps its synced clock for the duration
   of the experiment as long as it stays powered.

`as7341_influx_nir.py` checks `timedatectl` at startup and prints a warning
if the clock is unset (year < 2025) or NTP is unsynchronised, so you'll see
a loud message in `journalctl` rather than discovering bad timestamps later.

For unattended deployments where you can't guarantee NTP at every power-up,
add a battery-backed RTC HAT (e.g. DS3231) — Raspberry Pi OS supports it
out of the box via `dtoverlay=i2c-rtc,ds3231` in `/boot/firmware/config.txt`.

---

## Troubleshooting

**Sensor not detected**
```bash
groups $USER          # need 'i2c'
i2cdetect -y 1        # expect 0x39
```

**Script aborts at startup with `FileNotFoundError: as7341_lux_cal_hi.json`**
Run `python3 src/as7341_calibrate.py --phase lux --preset hi` to generate it.

**Saturation warnings on every sample**
The active preset is overexposed. With autorange on, the script will drop to
LO automatically after `AUTORANGE_HYST` consecutive saturated frames. If LO
also saturates, lower `SENS_LO_IT_MS` or `SENS_LO_GAIN`.

**`[WARN] Lux cal meta mismatch`**
The cal file was generated under different gain/ATIME/ASTEP than the active
preset. Either update the preset to match, or re-run the cal phase for that
preset.

**InfluxDB connection errors**
- Verify InfluxDB is up: `systemctl status influxdb`.
- Confirm the database exists: `influx -execute "SHOW DATABASES"`.
- The script keeps retrying in the background; check
  `journalctl -u as7341@$USER.service` for the retry queue size.

**`[WARN] System clock looks unset` or `[WARN] NTP not synchronised`**
The Pi has no RTC and hasn't reached an NTP server. See
[Field operation: clock sync without RTC](#field-operation-clock-sync-without-rtc)
above. Bring up a network with internet access (phone hotspot is the
simplest option) and wait for sync, or fit a DS3231 RTC HAT.

---

## Reference

- Integration time: `(ATIME + 1) × (ASTEP + 1) × 2.78 µs`.
- ADC full-scale: `min(65535, (ATIME + 1) × (ASTEP + 1))`.
- BasicCounts: `(raw - dark) / (gain × integration_time_ms)`. The lux model
  is fit and applied in this unit so that gain/IT changes between calibration
  and measurement do not invalidate the model.
- Absolute spectral irradiance: `irradiance_W_m2_nm[i] = BasicCounts[i] /
  responsivity_BC_per_W_m2_nm[i]`, where `responsivity_BC_per_W_m2_nm` is
  written by `as7341_calibrate.py` Phase 2 from C-7000 reference irradiance.
