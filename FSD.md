# AS7341 RPi Light Meter — Functional Specification (Short)

## 1. Purpose
A Raspberry Pi-based spectral light meter that publishes calibrated lux and
9-band relative spectral composition (415–680 nm visible + ~910 nm NIR) for
trend analysis and dashboarding. Supports both fixed indoor monitoring and
portable field measurements where network connectivity is intermittent.

## 2. Hardware
- Raspberry Pi (any model with I²C)
- Adafruit AS7341 10-channel spectral sensor (I²C @ 0x39)
- Cosine diffuser (recommended for accurate lux)
- Light-tight cap (used during dark calibration)

## 3. Inputs / Outputs
- **Sensor inputs:** 8 visible channels (415, 445, 480, 515, 555, 590, 630,
  680 nm), CLEAR (broadband), NIR (~910 nm).
- **Per-cycle outputs:**
  - `LIGHT` measurement — 9 points with `rel_intensity` field, tagged by
    `Device` and `wavelength_nm`.
  - `LIGHT_LUX` measurement — `lux` (calibrated) and `clear` (raw) fields,
    tagged by `Device` and `method`.
- **Sink:** InfluxDB v1 HTTP line protocol; multi-endpoint fan-out.
- **Local artefacts:** `as7341_dark.json` (per-channel offsets + meta),
  `as7341_lux_cal.json` (linear regression model + meta).

## 4. Operating Modes
- **M1 — Fixed station:** Continuous, networked. Live writes to InfluxDB.
- **M2 — Field / portable:** May be offline for hours to days. Buffers
  measurements locally and back-fills with original timestamps on reconnect.

## 5. Functional Requirements
| ID  | Requirement |
|-----|-------------|
| F1  | Configurable measurement period and frame averaging. |
| F2  | Sensor settings (gain, ATIME, ASTEP) must match calibration; mismatch surfaces a warning at startup for both dark and lux cal. |
| F3  | Dark-frame correction with metadata-validated offsets per channel (incl. NIR). |
| F4  | Lux from VIS8 linear/ridge regression with intercept; non-negative option. |
| F5  | Spectral composition: VIS8 normalized to 1.0 with responsivity correction; NIR reported as fraction of VIS+NIR. |
| F6  | **Auto-range** gain (and optionally ASTEP) to keep peak signal between 0.3 % and 87.5 % of ADC full-scale, with hysteresis. |
| F7  | Multi-endpoint InfluxDB fan-out with parallel writes. |
| F8  | **Persistent buffering** — failed writes survive process and OS restart, replay in chronological order on reconnect, bounded by configurable size and age. |
| F9  | Per-device tag (`Device`) so multiple Pis share a database. |
| F10 | Saturation and underflow warnings tied to ADC full-scale. |

## 6. Non-Functional Requirements
- Idle CPU: < 5 % on Pi Zero 2W at 60 s cadence.
- Disk for offline buffer: bounded (default 7 days at 60 s ≈ ~50 MB worst case).
- Crash-safe: a hard reboot must not lose buffered samples.
- Single-host install: Python venv + systemd unit.

## 7. Calibration
- **Dark capture:** ≥ 40 samples (warmup discarded), median per channel,
  written with full metadata (gain, ATIME, ASTEP, timestamp).
- **Lux model:** ≥ 8 diverse scenes against a reference lux meter; OLS or
  ridge; K-fold CV report; optional non-negativity constraint.
- **Validity:** Both files carry sensor-setting metadata. Runtime refuses to
  apply mismatched dark offsets and warns on mismatched lux cal.

## 8. Out of Scope
- UV (< 415 nm) — sensor lacks coverage.
- CCT / CRI estimation (possible future work).
- Grafana dashboards (separate repo).
- InfluxDB v2 / v3 line protocol (current version targets v1).

## 9. Open Items (see PLAN)
- Persistent on-disk buffer for field operation (F8).
- Re-introduce auto-range (F6) into the simplified main script.
- Deduplicate `as7341_influx_nir.py` vs legacy `RPi-as7341-InfluxDB.py`.
- Optional `.env`-driven configuration (sample present, not yet loaded).
- Fix `systemd/as7341.service` paths (currently point to `as7341-spectral`).
