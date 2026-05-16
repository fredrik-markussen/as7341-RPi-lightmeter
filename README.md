# Raspberry Pi AS7341 Spectral Light Meter → InfluxDB

Publishes calibrated spectral data from a Raspberry Pi to InfluxDB across
9 bands (415–680 nm visible + ~910 nm NIR), suitable for Grafana dashboarding
and field measurements. Primary outputs are **photon flux density**
(µmol/m²/s, 400–700 nm PAR range) and **relative spectral composition**;
lux is also computed as a secondary reference. **Lux readings are currently
unreliable** — the per-preset calibration models disagree at handoff points,
causing step jumps when autorange switches presets. Do not use lux for
quantitative work until the presets are recalibrated with overlapping scenes.

Two scripts do most of the work:

- `src/as7341_influx_nir.py` — the measurement service (run continuously by systemd).
- `src/as7341_calibrate.py` — the guided 3-phase calibration tool.

A short functional spec lives in [FSD.md](FSD.md).

## Features

- **Photon flux density** output (µmol/m²/s, 400–700 nm) from per-channel
  absolute irradiance — the primary quantity for circadian and seasonal biology.
- Three-preset auto-sensitivity (HI / LO / SUN) with hysteresis to avoid flapping.
- Per-preset dark calibration and per-preset VIS8 lux calibration.
- Guided 3-phase calibration script that walks through dark, spectral
  responsivity, and lux against a Seconic C-7000 + CoolLED pE-4000.
- Multi-endpoint InfluxDB fan-out with parallel writes, per-endpoint retry
  queues, and a CSV archive that writes every cycle by default.
- Optional local InfluxDB 1.x + Grafana stack for offline / hotspot live view.
- `.env`-driven configuration so the script does not need editing per device.

## Hardware

- Raspberry Pi (any model with I²C).
- Adafruit AS7341 10-channel spectral sensor (I²C @ 0x39).
- Cosine diffuser dome (recommended for accurate lux).
- Light-tight cap for dark-frame capture.

## Software prerequisites

- Raspberry Pi OS with I²C enabled.
- Python 3.11+ in a virtualenv.
- An InfluxDB v1.x endpoint (local or remote) accepting line protocol on `/write`.

---

## Maths

### Integration time and ADC full-scale

The AS7341 exposure is set via two registers:

```
t_int (ms) = (ATIME + 1) × (ASTEP + 1) × 2.78×10⁻³
ADC_FS     = min(65535, (ATIME + 1) × (ASTEP + 1))
```

ASTEP is maximised for the target integration time (better precision), ATIME
minimised. ADC full-scale caps at 65535 regardless of register product.

### BasicCounts — exposure-independent signal unit

Raw ADC readings depend on gain and integration time. BasicCounts remove that
dependency so calibration coefficients remain valid across presets:

```
BasicCounts[i] = (raw[i] - dark[i]) / (gain × t_int_ms)
```

Dark offsets are captured per preset (Phase 1) and are only valid when the
gain, ATIME, and ASTEP exactly match the capture conditions — mismatches are
flagged at startup and dark correction is silently disabled.

### Phase 1 — Dark calibration

With the sensor covered, 100 frames are captured per channel per preset. The
per-channel median is saved with full sensor metadata (gain, ATIME, ASTEP).
Median rather than mean rejects any single hot-pixel spikes or transients
during capture.

### Phase 2 — Spectral responsivity against CoolLED + C-7000

The CoolLED pE-4000 is operated in single-LED mode, one wavelength at a time
across a sweep of 12 LEDs in the 405–660 nm range. At each step:

1. The Seconic C-7000 measures the spectral irradiance (W/m²/nm) at each
   AS7341 channel center wavelength. These values come from the C-7000's native
   CSV export, which is parsed and linearly interpolated to the 8 channel
   centers (415, 445, 480, 515, 555, 590, 630, 680 nm).
2. The AS7341 captures BasicCounts simultaneously under the same illumination.
3. Per-channel responsivity (BasicCounts per W/m²/nm) is computed as the ratio
   of BC to irradiance at that step. A channel's ratio is only included for LED
   steps where that channel receives at least 20% of the peak irradiance — this
   rejects off-peak ratios that would amplify shot noise. The final responsivity
   per channel is the mean across all qualifying steps.

Two outputs are stored in `as7341_responsivity.json`:

- **`corrections`** — per-channel multipliers normalised to 555 nm = 1.0.
  Applied when computing relative spectral composition so that equal true
  irradiance at each wavelength produces equal BasicCounts after correction.
- **`responsivity_BC_per_W_m2_nm`** — absolute responsivity. Used to convert
  BasicCounts back to physical irradiance units in W/m²/nm.

### Relative spectral composition

Applied every measurement cycle using the correction factors from Phase 2:

```
BC_corrected[i] = BasicCounts[i] × correction[i]
rel_intensity[i] = BC_corrected[i] / Σ BC_corrected[VIS8]
```

The eight VIS8 channels (415–680 nm) normalise to sum = 1.0 independently.
The 910 nm NIR channel is written as a separate data point using the same
formula (NIR BC corrected / sum of VIS8 corrected) but is not part of the
VIS8 normalisation.

### Absolute irradiance

Requires Phase 2 absolute responsivity:

```
irradiance[i] (W/m²/nm) = BasicCounts[i] / responsivity_BC_per_W_m2_nm[i]
```

Emitted as the `irradiance` field on each VIS8 point in InfluxDB and as
`irr_*` columns in the CSV.

### Photon flux density (PFD)

Summed across all 8 VIS channels (415–680 nm, covering the PAR window):

```
PFD (µmol/m²/s) = Σ irradiance[i] × λ[i] (nm) × Δλ[i] (nm) / 119700
```

where Δλ is each channel's FWHM bandwidth (26, 30, 36, 39, 39, 40, 50, 52 nm
for 415–680 nm respectively) and 119700 = h × c × Nₐ scaled to µmol·nm/J.
Requires Phase 2 absolute responsivity.

### Phase 3 — Lux calibration

The AS7341 and C-7000 are placed side-by-side under the same illumination
across a diverse set of scenes (default 12 per preset). At each scene,
BasicCounts and the C-7000 lux reading are recorded simultaneously. A linear
model is fitted per preset:

```
lux = b0 + Σ w[i] × BasicCounts[i]   (i over VIS8)
```

Fitting uses ridge regression (α = 0.01) with MAD outlier rejection and
k-fold cross-validation to report goodness of fit. Calibration is per-preset
because changing gain or integration time rescales BasicCounts — the
coefficients are not transferable between presets. **Lux is currently
unreliable** pending a recalibration with overlapping scenes across all three
presets (see task #1 in the project log).

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
| `INFLUX_ENDPOINTS`   | JSON array of `[host, port, database]` triples. Defaults to localhost only; add remote hosts alongside it. |
| `AVG`                | Frames averaged per measurement (5 is a good default). |
| `PERIOD`             | Seconds between measurements. |
| `CSV_ALWAYS`         | `true` (default) — write a CSV row every cycle. `false` — write only when all Influx endpoints fail. |
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

Run on the Pi:
```bash
python3 src/as7341_calibrate.py --phase dark
```

Useful flags:
```
--dark-samples 200     # more samples for a quieter median (default 100)
--out-dir /path/to/    # write cal files to a different directory
```

The standalone `src/as7341_dark_capture.py` is a smaller utility for ad-hoc
dark captures with custom settings; for the standard workflow prefer the guided
script above.

### Phase 2 — Spectral responsivity (VIS8)

- AS7341 + Seconic C-7000 side-by-side under a CoolLED pE-4000 in
  **single-LED mode** — one wavelength at a time, dialled to a non-saturating
  intensity.
- Wavelength sweep: 12 in-VIS8-range pE-4000 LEDs by default
  (`405, 435, 460, 470, 490, 500, 525, 550, 580, 595, 635, 660` nm). The
  pE-4000 carries 16 LEDs grouped across 4 channels (365–770 nm); the default
  list picks the LEDs that land inside the AS7341 VIS8 passband.
- For each LED step, capture the AS7341 reading and either provide a C-7000
  CSV (`wavelength_nm, irradiance_W_m2_nm`) or type the channel-centre
  irradiances. Per-channel responsivity is averaged across only the LED steps
  where that channel sees significant power (`--resp-min-irr-frac`, default 0.2
  of the strongest channel at that step) — this rejects far-off-peak ratios
  that would amplify noise.
- **Pre-collected C-7000 data:** If you already have the C-7000 exports in the
  repo (`C-7000_out/`), pull the repo onto the Pi and use `--c7000-dir` to skip
  manual SPD entry — the script reads irradiance directly from the CSVs and only
  asks you to capture the AS7341 side:
  ```bash
  git pull
  python3 src/as7341_calibrate.py --phase responsivity --c7000-dir C-7000_out/
  ```
- Output: `as7341_responsivity.json` with two blocks:
  - `corrections` — per-channel multipliers normalised to 555 nm = 1.0,
    used to correct the relative spectrum (`rel_intensity`).
  - `responsivity_BC_per_W_m2_nm` — per-channel absolute responsivity in
    BasicCounts per W/m²/nm. When present, the runtime emits absolute
    spectral irradiance per VIS8 channel (`irr_*` CSV columns,
    `irradiance` Influx field).
  Plus `meta.wavelengths_nm`, `meta.n_samples_per_channel`, and a `raw_levels`
  block for post-hoc inspection.
- NIR (~910 nm) is **not** measured here — the C-7000 covers 380–780 nm only
  and the pE-4000 stops at 770 nm. The runtime keeps a datasheet default for
  NIR composition correction (overridable via a `nir` key under `corrections`).

Useful flags:
```
--resp-wavelengths 405,460,525,...   # custom LED sweep (default 12 LEDs)
--resp-min-irr-frac 0.1             # lower threshold to include more off-peak steps
--resp-avg 20                        # frames averaged per AS7341 capture (default 20)
```

### Phase 3 — Lux model

- AS7341 + C-7000 side-by-side, same angle, no shadows. Use diverse light
  scenes spanning the lux range you expect to measure.
- Default 12 scenes per preset (≥ 10 needed for a stable fit). HI preset
  saturates above ~5 000 lx — use those scenes for LO only.
- VIS8 ridge regression with intercept (α=0.01); MAD outlier rejection;
  K-fold CV reports goodness-of-fit.
- Outputs: `as7341_lux_cal_hi.json`, `as7341_lux_cal_lo.json`.

Run on the Pi (both presets in one go):
```bash
python3 src/as7341_calibrate.py --phase lux
```

Re-run a single preset:
```bash
python3 src/as7341_calibrate.py --phase lux --preset hi
python3 src/as7341_calibrate.py --phase lux --preset lo
```

Useful flags:
```
--lux-scenes 15        # collect more scenes (default 12)
--lux-ridge 0.0        # plain OLS instead of ridge (only with plenty of scenes)
--lux-nnls             # constrain weights to be non-negative
--lux-kfold 5          # K-fold CV folds
--lux-avg 10           # frames averaged per capture (default 10)
```

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

## Local stack: InfluxDB + Grafana on the Pi

Running InfluxDB 1.x and Grafana directly on the Pi means data is recorded
and viewable without internet access. The measurement script writes to
`127.0.0.1:8086` over the loopback, so it keeps going even when there's no
network.

### Install

Requires 64-bit Raspberry Pi OS (Bookworm 64-bit). 32-bit `armhf` is not
supported by upstream InfluxDB .deb releases.

```bash
sudo bash setup/install_local_stack.sh
```

The script (idempotent, re-runnable):
- Downloads and installs InfluxDB 1.12.4 from `dl.influxdata.com` (direct
  `.deb`, no apt repo — avoids the v1/v2 package-name churn in
  `repos.influxdata.com`).
- Creates the `lightmeter` database with a 90-day retention policy.
- Adds the official `apt.grafana.com` repo and installs Grafana.
- Provisions a `lightmeter` datasource in Grafana pointing at `localhost:8086`.
- Enables both services so they start automatically after reboot.

### First login

Open `http://<pi-hostname>.local:3000` in a browser (or `http://localhost:3000`
from the Pi itself). First-login credentials are `admin` / `admin`; Grafana
will prompt you to change the password.

### Building a dashboard

1. Go to **Dashboards → New → New dashboard**.
2. Add a panel, select the **lightmeter** datasource (InfluxDB).
3. Example query for lux over the last hour:
   ```
   SELECT mean("lux") FROM "LIGHT_LUX"
   WHERE $timeFilter
   GROUP BY time($__interval)
   ```
4. For spectral composition: select measurement `LIGHT`, filter by `wavelength_nm` tag.

### Adjusting retention

To keep more or less data than the 90-day default:

```bash
influx -execute 'ALTER RETENTION POLICY autogen ON lightmeter DURATION 30d REPLICATION 1 DEFAULT'
```

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

**Lux / PFD** — measurement `LIGHT_LUX`, 1 point per cycle:

- Tags: `Device=<DEVICE>`, `method=lin_basic`.
- Fields: `lux` (calibrated illuminance), `clear` (raw CLEAR channel ADC),
  `pfd` (total photon flux density in µmol/m²/s across 415–680 nm —
  emitted only when Phase 2 absolute responsivity calibration is present).

**CSV archive** — same data, 22 columns:

```
timestamp_iso,device,lux,clear,
rel_415,rel_445,rel_480,rel_515,rel_555,rel_590,rel_630,rel_680,rel_nir,
irr_415,irr_445,irr_480,irr_515,irr_555,irr_590,irr_630,irr_680,
pfd
```

The `irr_*` cells (W/m²/nm at channel center) and `pfd` (µmol/m²/s) are
populated only when Phase 2 has produced an absolute responsivity; otherwise
they are emitted empty so the column count stays stable.

Quick sanity query:

```bash
influx -database lightmeter \
  -execute "SELECT * FROM LIGHT_LUX WHERE Device='RPi-1' ORDER BY time DESC LIMIT 5"
```

---

## CSV output and offline buffering

Failed Influx writes go into per-endpoint in-memory retry queues (bounded by
`MAX_RETRY_QUEUE`, default 500) and replay when the endpoint recovers.

The CSV archive under `~/Documents/Lightmeter_csv_out/` is written in one of
two modes, controlled by `CSV_ALWAYS` in `.env`:

- **`CSV_ALWAYS=true` (default)** — a CSV row is written every measurement
  cycle regardless of Influx success. Every sample has a local copy; use this
  for field deployments where you want a guaranteed record alongside the live
  Influx feed.
- **`CSV_ALWAYS=false`** — CSV is written only when **every** Influx endpoint
  fails on a given sample. True offline-buffer behaviour; for outages
  exceeding the retry queue, import the daily CSVs manually after recovery.

Rows go into a 10-minute tmp CSV in `daily_tmp/`; tmp files are merged into
per-day aggregates in the parent directory; on startup, any leftover tmps from
a previous run are merged. Filenames and row timestamps are both UTC.

---

## Headless health check (status.json)

Every cycle the runtime atomically writes a small JSON snapshot to
`~/Documents/Lightmeter_csv_out/status.json`. It carries the latest sample
(lux, spectral composition, irradiance, sensor settings, saturation
fraction), per-endpoint health (success/failure counters, retry queue
depth, last-success and last-failure timestamps with the failure message),
the CSV state including `disk_free_mb` for the output filesystem, and
process uptime.

Quick liveness checks over SSH:

```bash
# Raw dump
ssh pi@RPi-1 'cat ~/Documents/Lightmeter_csv_out/status.json'

# One-line summary
ssh pi@RPi-1 'jq -r ".last_sample | \"\(.timestamp_iso) lux=\(.lux) sat=\(.saturation_frac) preset=\(.active_preset)\"" \
              ~/Documents/Lightmeter_csv_out/status.json'

# Disk free
ssh pi@RPi-1 'jq ".csv.disk_free_mb" ~/Documents/Lightmeter_csv_out/status.json'
```

If `status.json` is older than a couple of cycles, the process is wedged
or stopped — `systemctl status as7341@$USER.service` will tell you why.

---

## Storage and retention

With local InfluxDB and per-cycle CSV running continuously, storage management
matters.

### SD card choice

Use a high-endurance card rated for continuous writes. Good options:
- **Samsung PRO Endurance** — rated for years of 24/7 video surveillance writes.
- **SanDisk High Endurance** — similar rating, widely available.

A 64 GB card gives comfortable headroom. For deployments longer than about 6
months, or if the Pi is mounted somewhere inconvenient to service, booting from
a USB SSD is more reliable than any SD card.

### Filesystem tweak

Add `noatime` to the rootfs mount in `/etc/fstab` to eliminate access-time
writes (can halve write traffic on a read-heavy workload):

```
PARTUUID=xxxxxxxx-02  /  ext4  defaults,noatime  0  1
```

Edit the existing rootfs line; run `sudo mount -o remount,noatime /` to apply
without a reboot.

### InfluxDB footprint

At `PERIOD=60` with 10 fields per sample, TSI-compressed storage runs roughly
5–15 MB/day. The default 90-day retention policy therefore uses about 0.5–1.5 GB.
Adjust with:

```bash
influx -execute 'ALTER RETENTION POLICY autogen ON lightmeter DURATION 60d REPLICATION 1 DEFAULT'
```

InfluxDB enforces the policy automatically — old shards are dropped when they
age out.

### CSV footprint

| Period  | Rows/day | Uncompressed/day | Uncompressed/year |
|---------|----------|-----------------|-------------------|
| 60 s    | ~1 440   | ~250 KB         | ~90 MB            |
| 10 s    | ~8 640   | ~1.5 MB         | ~550 MB           |
| 5 s     | ~17 280  | ~3 MB           | ~1.1 GB           |

CSV daily files are not automatically pruned. Add a cron job to clean up old
ones (adjust the `-mtime` value to taste):

```bash
crontab -e
# Keep 180 days of daily CSV files:
0 3 * * * find ~/Documents/Lightmeter_csv_out -name '*_daily.csv' -mtime +180 -delete
```

### Out-of-space behaviour

If the filesystem fills up, InfluxDB refuses new writes (the script logs
`[ERR] <endpoint>: ...`), and the CSV append throws (logged as
`[ERR] CSV write failed: ...`). The main measurement loop keeps going, so
data resumes when space is freed. Monitor free space without SSH via the
`csv.disk_free_mb` field in `status.json`:

```bash
ssh pi@RPi-1 'jq ".csv.disk_free_mb" ~/Documents/Lightmeter_csv_out/status.json'
```

---

## Field operation: clock sync without RTC

A stock Raspberry Pi has no battery-backed RTC. At boot it restores the last
saved time from `fake-hwclock` and only corrects to wall-clock time once it
sees an NTP server. For field measurements that means CSV timestamps are
only as good as the last sync: a cold boot with no network leaves rows
stamped with the previous shutdown's time until NTP catches up.

**Recommended pre-deployment routine:**

1. Pre-configure the Pi to auto-connect to a phone hotspot (see
   [Field operation: hotspot live view](#field-operation-hotspot-live-view) below).
2. In the field, power up the Pi within range of the configured hotspot.
3. NTP syncs automatically once the hotspot connection is up.
4. Verify: `timedatectl` should report `System clock synchronized: yes`.
5. Disconnect the hotspot. The Pi keeps its synced clock for the duration
   of the experiment as long as it stays powered.

`as7341_influx_nir.py` checks `timedatectl` at startup and prints a warning
if the clock is unset (year < 2025) or NTP is unsynchronised.

For unattended deployments where you can't guarantee NTP at every power-up,
add a battery-backed RTC HAT (e.g. DS3231) — Raspberry Pi OS supports it
out of the box via `dtoverlay=i2c-rtc,ds3231` in `/boot/firmware/config.txt`.

---

## Field operation: hotspot live view

The Pi can serve Grafana to your phone over a hotspot connection with no
internet or external infrastructure. Data keeps writing to local InfluxDB
the entire time regardless of hotspot state.

### Pre-configure the hotspot profile

**Raspberry Pi OS Bookworm** (NetworkManager):

```bash
sudo bash setup/configure_hotspot.sh "<hotspot SSID>" "<password>"
```

This creates a NetworkManager profile (`lightmeter-hotspot`) that
auto-connects whenever the named hotspot is in range. The measurement
service keeps running uninterrupted.

The script autodetects the first NetworkManager-managed wifi interface.
On a Pi where the built-in `wlan0` is left unmanaged (e.g. because a
USB wifi adapter on `wlan1` is the active radio), it picks `wlan1`. If
you have multiple managed wifi interfaces and want a specific one, pass
`--interface wlan1` (or whichever).

Optional — fix the Pi's IP on the hotspot subnet so your phone always
reaches it at the same address:

```bash
# Android hotspot subnets are typically 192.168.43.x
sudo bash setup/configure_hotspot.sh "<SSID>" "<password>" --static-ip 192.168.43.50/24

# iPhone hotspot subnets are typically 172.20.10.x
sudo bash setup/configure_hotspot.sh "<SSID>" "<password>" --static-ip 172.20.10.5/28
```

**Raspberry Pi OS Bullseye / Buster** (wpa_supplicant):

Add a network block to `/etc/wpa_supplicant/wpa_supplicant.conf`:

```
network={
    ssid="<hotspot SSID>"
    psk="<password>"
    priority=50
}
```

Then `sudo wpa_cli -i wlan0 reconfigure`.

### Connecting

1. Enable the hotspot on your phone.
2. Power up (or leave powered on) the Pi — it auto-connects within ~30 s.
3. Open a browser on your phone:
   - `http://<pi-hostname>.local:3000` — mDNS (works out of the box on iOS;
     newer Android versions also support it via `.local` resolution).
   - `http://<static-ip>:3000` — use the static IP if `.local` doesn't
     resolve (common on older Android).

The Pi's hostname is shown by `hostname` on the Pi, or set it with
`sudo raspi-config` → System Options → Hostname.

Grafana login: `admin` / `admin` on first use (you'll be prompted to change it).

---

## Operating the lightmeter (end-user routine)

This section is for users running an already-configured lightmeter in the
field. The administrator has done the install, hotspot pairing, dashboard
building, and storage tuning. The end-user only needs their phone, the Pi,
and a power source.

### Admin handoff checklist

Before handing the lightmeter over, the admin confirms:

- [ ] `setup/install_local_stack.sh` ran cleanly; `systemctl is-active influxdb grafana-server` returns `active` for both.
- [ ] `setup/configure_hotspot.sh "<SSID>" "<password>"` was run with the end-user's hotspot SSID and password. Verified by powering the Pi within range and watching it associate (`nmcli connection show --active`).
- [ ] Pi hostname is set to something memorable (e.g. `lightmeter`, not the default `raspberrypi`) via `sudo raspi-config` → System Options → Hostname.
- [ ] Grafana admin password has been changed from the default and recorded on the reference card.
- [ ] A Grafana dashboard pointing at the `lightmeter` datasource has been built and saved (see [Local stack → Building a dashboard](#building-a-dashboard) for example queries).
- [ ] systemd service is enabled and started: `systemctl is-enabled as7341@<user>.service` returns `enabled`.
- [ ] Calibration files exist at the repo root: `as7341_dark_hi.json`, `as7341_dark_lo.json`, `as7341_lux_cal_hi.json`, `as7341_lux_cal_lo.json` (and optionally `as7341_responsivity.json`).
- [ ] High-endurance SD card or USB SSD installed; `noatime` on `/`; a CSV-cleanup cron job if applicable.
- [ ] The end-user has a reference card with the values below.

### Reference card (give this to the end-user)

```
Grafana URL: http://<hostname>.local:3000   (or  http://<static-ip>:3000)
Login:       <username> / <password>
Dashboard:   <dashboard name>
Hotspot:     <SSID>            (already saved on your phone)
```

### Starting a measurement session

1. Turn **ON** your phone's mobile hotspot.
2. Plug the Pi into power. The red LED lights immediately; the green
   activity LED flickers as it boots.
3. **Wait 10 minutes.** During this window the Pi:
   - Boots (~1–2 min on Pi 4, ~2–3 min on Pi Zero 2 W).
   - Joins your hotspot (~30 s).
   - Syncs the clock via NTP (~30 s after network is up).
   - Begins measuring — first sample within ~1 min of the service starting.
4. On your phone, open a browser and go to the Grafana URL on the reference card.
5. Log in. Open the dashboard.
6. Confirm new data is appearing — the most recent point should be timestamped
   within the last couple of minutes. At the default `PERIOD=60` a new point
   lands every minute.
7. Once you've seen at least one fresh point, turn the hotspot **OFF**.
   The Pi keeps measuring and recording locally; the hotspot is only needed
   for live viewing.

### Coming back hours or days later

The Pi stays powered the entire time; only the hotspot was off.

1. Turn **ON** the hotspot.
2. Wait ~30 s for the Pi to rejoin.
3. Open Grafana on your phone (you may still be logged in from last time).
4. Set the time range in the top-right corner — "Last 24 hours",
   "Last 7 days", or a custom range covering the period since your last visit.
5. All accumulated measurements appear.
6. When done viewing, turn the hotspot off again.

### Stopping a session

When the experiment is over and the Pi can be powered down:

- Cleanest option: with the hotspot on, SSH in (`ssh <user>@<hostname>.local`)
  and run `sudo shutdown -h now`. Wait for the green LED to stop flickering
  (~10 s), then unplug.
- If SSH isn't available, just unplug. The filesystem usually survives, but
  a clean shutdown is better for card longevity.

### Operational parameters

| Parameter                                | Value                              | Notes |
|------------------------------------------|------------------------------------|-------|
| Measurement cadence                      | 60 s                               | Set via `PERIOD` in `.env`. Lowering increases SD wear. |
| Time to first data after cold boot       | 3–10 min                           | Boot + hotspot join + NTP + first cycle. |
| Time to reconnect after hotspot ON       | ~30 s                              | Auto-rejoin via NetworkManager. |
| Pi idle power draw                       | 1–5 W                              | Zero 2 W: ~1 W; Pi 4: ~3–5 W. |
| Run time on 20 000 mAh USB power bank    | 24–100 h                           | Wall power recommended for multi-day deployments. |
| Run time on wall power                   | indefinite                         | |
| Storage per day                          | ~5–15 MB Influx + ~250 KB CSV      | At `PERIOD=60`. Scales linearly with cadence. |
| Default InfluxDB retention               | 90 days                            | Configurable; see [Storage and retention](#storage-and-retention). |
| Hotspot needed for                       | live viewing only                  | Data records continuously without network. |
| Clock drift while hotspot OFF            | ~10 s/day                          | Resets to NTP-accurate on next hotspot rejoin. |
| Max hotspot-OFF duration (data)          | until SD card fills                | At default cadence and 64 GB card: many months. |
| Max hotspot-OFF duration (clock)         | weeks                              | Drift accumulates from the Pi's onboard oscillator. |

### End-user troubleshooting

**Phone won't load the Grafana URL**
- Make sure your phone is on its own hotspot Wi-Fi. Some phones turn Wi-Fi
  off when sharing — flip it back on; the hotspot stays active.
- Some phones block the host device from reaching connected clients via
  the phone's own browser. If that's the case, join the hotspot from a
  second device (tablet, laptop) and open Grafana there.
- Wait an extra 2 minutes after first power-up; cold boots can be slow.
- Try the static-IP form on the reference card (`http://<ip>:3000`) if
  the `.local` hostname doesn't resolve.

**Grafana loads but the dashboard says "No data"**
- The time range (top-right) may be set to a window with no measurements
  yet. Switch to "Last 15 minutes" or "Last hour" and refresh.
- The Pi may still be booting; wait another minute and refresh.

**Timestamps look wrong (years in the past)**
- The Pi's clock didn't sync to NTP before measurements began — usually
  caused by turning the hotspot on **after** the Pi was already booting.
  Power-cycle the Pi with the hotspot **already on** and wait 10 min.
- Existing correctly-stamped data is not affected.

**Pi appears unresponsive (no Grafana, can't SSH)**
- LED check: solid red = power OK; green flickering during activity = OK.
  No green flicker for >30 s after boot suggests something is wrong.
- Unplug, wait 10 s, plug back in. Allow another 10 min before retrying.
- If that fails, the admin will need to attach a monitor or read the SD card.

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

**Disk full — CSV and InfluxDB writes failing**
Check free space: `jq ".csv.disk_free_mb" ~/Documents/Lightmeter_csv_out/status.json`.
Delete old daily CSV files or reduce `INFLUX_ENDPOINTS` retention. See
[Storage and retention](#storage-and-retention).

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
- Photon flux density: `pfd (µmol/m²/s) = Σ irr_i × λ_i(nm) × Δλ_i(nm) / 119700`
  summed across the 8 VIS channels (415–680 nm); channel bandwidths (FWHM)
  are 26, 30, 36, 39, 39, 40, 50, 52 nm respectively.
