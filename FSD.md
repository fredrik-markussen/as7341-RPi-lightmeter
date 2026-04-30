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
- **Local artefacts:**
  - `as7341_dark_{hi,lo}.json` — per-channel dark offsets + metadata, one per sensitivity preset.
  - `as7341_lux_cal_{hi,lo}.json` — VIS8 linear regression model + metadata, one per preset.
  - `as7341_responsivity.json` — empirical per-channel responsivity corrections (optional; overrides datasheet defaults).

## 4. Operating Modes
- **M1 — Fixed station:** Continuous, networked. Live writes to InfluxDB.
- **M2 — Field / portable:** May be offline for hours to days. Buffers
  measurements locally and back-fills with original timestamps on reconnect.

## 5. Functional Requirements
| ID  | Requirement |
|-----|-------------|
| F1  | Configurable measurement period and frame averaging. |
| F2  | Sensor settings (gain, ATIME, ASTEP) must match calibration; mismatch surfaces a warning at startup and on every sensitivity switch, for both dark and lux cal. |
| F3  | Dark-frame correction with metadata-validated offsets per channel (incl. NIR). |
| F4  | Lux from VIS8 OLS or ridge regression with intercept, MAD outlier rejection, K-fold CV, optional non-negative weights (`--lux-nnls`). |
| F5  | Spectral composition: VIS8 normalised to 1.0 with responsivity correction; NIR (~910 nm) reported as fraction of VIS+NIR using a datasheet-default correction (overridable via `as7341_responsivity.json`'s `nir` key, since the C-7000 reference does not reach 910 nm). |
| F6  | **Auto-sensitivity** — two fixed presets (HI: high gain/long IT for dim; LO: low gain/short IT for bright). Hysteresis-counted switches on saturation/underflow. Each preset has its own dark and lux cal files. |
| F7  | Multi-endpoint InfluxDB fan-out with parallel writes; per-endpoint connect/read timeouts and a wall-clock budget so a single hung endpoint cannot stall the loop. |
| F8  | **Buffering** — failed writes are held in a per-endpoint in-memory retry queue (bounded by `MAX_RETRY_QUEUE`) and replayed when the endpoint recovers, in chronological order. When **all** endpoints fail simultaneously, samples are also written to a daily CSV archive in `~/Documents/Lightmeter_csv_out/` (rotation: 10-min tmp files merged into per-day files; startup recovery merges any leftovers). The CSV is archive-only — it is **not** automatically replayed to InfluxDB. Process restart drops the in-memory queue, so prolonged outages are recoverable from CSV but require manual back-fill. |
| F9  | Per-device tag (`Device`) so multiple Pis share a database. |
| F10 | Saturation and underflow warnings tied to ADC full-scale. |

## 6. Non-Functional Requirements
- Idle CPU: < 5 % on Pi Zero 2W at 60 s cadence.
- Disk for offline buffer: bounded (default 7 days at 60 s ≈ ~50 MB worst case).
- Crash-safe: a hard reboot must not lose buffered samples.
- Single-host install: Python venv + systemd unit.

## 7. Calibration

Guided by `src/as7341_calibrate.py`. Reference equipment: Seconic C-7000
spectroradiometer + CoolLED pE-4000 + dome diffusers.

- **Phase 1 — Dark capture:** 100 samples per preset (warmup discarded),
  median per channel, written with full metadata (gain, ATIME, ASTEP, timestamp).
  Run for both HI and LO presets → `as7341_dark_hi.json`, `as7341_dark_lo.json`.
- **Phase 2 — Spectral responsivity (VIS8):** AS7341 + C-7000 side-by-side at
  5 CoolLED intensity levels (default 10/25/50/75/90 %). C-7000 SPD input via
  CSV export (sorted automatically) or manual entry at the 8 channel
  wavelengths. Computes empirical per-channel correction factors normalised to
  555 nm = 1.0 → `as7341_responsivity.json`. Loaded automatically by the main
  script at startup if present; falls back to datasheet values otherwise.
  NIR (~910 nm) is **not** measured in Phase 2 because the C-7000 covers
  380–780 nm only; the script keeps the datasheet default for NIR. To override,
  add a `nir` entry under `corrections` in `as7341_responsivity.json`.
- **Phase 3 — Lux model:** ≥ 10 diverse scenes per preset against C-7000 lux
  readings (default 12, recommended 12–20). VIS8 with intercept; default fit is
  ridge (α=0.01) for stability with small samples; OLS available with
  `--lux-ridge 0`; non-negative weights via `--lux-nnls`. MAD outlier rejection
  and K-fold CV report goodness-of-fit. → `as7341_lux_cal_hi.json`,
  `as7341_lux_cal_lo.json`.
- **Validity:** All cal files carry sensor-setting metadata. At startup and on
  every sensitivity switch, the runtime checks both dark and lux cal metadata
  against the active preset's gain/ATIME/ASTEP and warns on mismatch (dark
  offsets are zeroed; lux coefficients are still applied with a loud warning so
  the user can investigate).

## 8. Out of Scope
- UV (< 415 nm) — sensor lacks coverage.
- CCT / CRI estimation (possible future work).
- Grafana dashboards (separate repo).
- InfluxDB v2 / v3 line protocol (current version targets v1).

## 9. Open Items

### Resolved (2026-04-29)
- ~~Persistent on-disk buffer for field operation (F8).~~ CSV archive fallback in `as7341_influx_nir.py`; FSD F8 reworded to match (archive-only, not auto-replayed).
- ~~Re-introduce auto-range (F6) into the simplified main script.~~ 2-step HI/LO sensitivity switching with hysteresis.
- ~~Deduplicate `as7341_influx_nir.py` vs legacy `RPi-as7341-InfluxDB.py`.~~ Legacy deleted.
- ~~Optional `.env`-driven configuration.~~ `_apply_env()` reads `BASE_DIR/.env`; `config/sample.env` updated.
- ~~Fix `systemd/as7341.service` paths.~~ Now points to `as7341-RPi-lightmeter`.
- ~~`as7341_dark_capture.py` hardcoded ATIME/ASTEP~~ — now CLI-driven (`--integration-time-ms`, `--gain`, `--samples`, `--out`).
- ~~Guided calibration script missing~~ — `src/as7341_calibrate.py` implements full 3-phase guided flow.
- ~~Datasheet responsivity corrections~~ — Phase 2 produces empirical `as7341_responsivity.json`; main script loads it automatically.
- ~~Lux cal metadata not validated at runtime (F2 gap)~~ — `apply_sensitivity()` now warns when cal meta does not match current gain/ATIME/ASTEP.
- ~~NIR responsivity bias in `rel_nir`~~ — main script applies a separate `NIR_RESPONSIVITY_CORRECTION` (datasheet default 2.5, override via JSON) before computing the VIS+NIR fraction.
- ~~Phase 3 default rank-deficient with 8 scenes~~ — default bumped to 12 (≥ 10 enforced for fit); ridge α=0.01 default; `--lux-nnls` added for non-negative weights.
- ~~`as_completed` could crash the loop on an endpoint hang~~ — wrapped in `FuturesTimeoutError` handler that queues the payload for retry on each endpoint.
- ~~Old `as7341_dark.json` / `as7341_lux_cal.json` artefacts~~ — removed from repo root.
- ~~README out of sync with current code~~ — rewritten for the HI/LO preset model and `.env` configuration.

### Remaining (hardware required)
- **Calibration files missing** — `as7341_dark_hi/lo.json`, `as7341_lux_cal_hi/lo.json`, and `as7341_responsivity.json` must be generated on hardware using `src/as7341_calibrate.py`. Without `as7341_lux_cal_hi.json` the script fails at startup; without `as7341_lux_cal_lo.json` it fails at the first HI→LO sensitivity switch.

### Known limitations (by design)
- **NIR not radiometrically calibrated** — the C-7000 reference does not reach 910 nm, so NIR uses a datasheet default correction. Users with an NIR-capable reference can override via the `nir` key in `as7341_responsivity.json`.
- **CSV archive is not auto-replayed** — for prolonged outages exceeding the in-memory retry queue (`MAX_RETRY_QUEUE` × cadence), recovery requires manual import of the daily CSVs into InfluxDB.
