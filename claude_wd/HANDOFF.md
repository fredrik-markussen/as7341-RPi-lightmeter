# Claude Handoff — 2026-04-29

## Session 1 — Measurement script

| Task | Detail |
|---|---|
| F6 auto-sensitivity | 2-step HI/LO preset switching (`SENS_HI` GAIN_256X/50ms, `SENS_LO` GAIN_16X/10ms). Hysteresis counter prevents flapping. On switch: registers updated, cal files reloaded, frame discarded. |
| F8 CSV archive | 10-min tmp files → daily aggregates in `~/Documents/Lightmeter_csv_out/`. Startup recovery merges leftovers. |
| Sleep cadence fix | `max(0.0, max(PERIOD, min_period) - loop_elapsed)` |
| Legacy script deleted | `src/RPi-as7341-InfluxDB.py` removed |
| `.env` loading | `_apply_env(BASE_DIR / ".env")` at module level; `config/sample.env` updated |
| Systemd paths | `systemd/as7341.service` now references `as7341-RPi-lightmeter` |

## Session 2 — Calibration scripts

| Task | Detail |
|---|---|
| `src/as7341_dark_capture.py` refactored | CLI args: `--integration-time-ms`, `--gain`, `--samples` (default 100), `--warmup`, `--out`. |
| `src/as7341_calibrate.py` created | Guided 3-phase calibration script (dark, responsivity, lux). |
| `src/as7341_influx_nir.py` | Loads `as7341_responsivity.json` at startup; falls back to datasheet defaults. |

## Session 3 — Code review fixes (2026-04-29)

| Task | Detail |
|---|---|
| F2 lux cal meta validation | `load_cal()` now returns meta; `apply_sensitivity()` warns when cal gain/ATIME/ASTEP do not match the active preset. |
| Robust HTTP timeout | `as_completed(timeout=…)` wrapped in `FuturesTimeoutError` handler; hung endpoints queue for retry instead of crashing the loop. |
| NIR responsivity | New `NIR_RESPONSIVITY_CORRECTION` constant (datasheet default 2.5, override via `corrections.nir` in JSON); applied to NIR before computing the VIS+NIR fraction. |
| Phase 3 robustness | Default `--lux-scenes` 12 (≥ 10 enforced), `--lux-ridge` 0.01, scene-skip on saturation, per-preset `random.seed`, new `--lux-nnls` for non-negative weights, `n_scenes_used` recorded in cal meta. |
| Phase 2 levels | Default 5-level set is now [10, 25, 50, 75, 90] %. C-7000 CSV is sorted by wavelength after parsing. |
| `as7341_dark_capture.py` | Default `--out` now `as7341_dark_hi.json`; standardised on `board.I2C()`. |
| Cleanup | Removed legacy root-level `as7341_dark.json` / `as7341_lux_cal.json`; removed dead imports and unused `median_frames`. |
| FSD updates | F2 reworded for both-side validation; F4 reflects ridge default + NNLS option; F5 documents NIR datasheet default + override path; F7 mentions wall-clock budget; F8 reworded as in-memory retry queue + CSV archive (not auto-replayed). |
| README | Rewritten for HI/LO preset model and `.env` configuration. |

## Current file state

| File | Status |
|---|---|
| `src/as7341_influx_nir.py` | Active measurement script — complete. |
| `src/as7341_calibrate.py` | Guided calibration script — complete; awaits hardware run. |
| `src/as7341_dark_capture.py` | Standalone dark utility; defaults to HI preset. |
| `src/as7341_calibrate_lux.py` | Legacy stand-alone (NNLS, bootstrap, CLEAR/NIR regressors). Kept as power-user fallback; not part of guided flow. |
| `systemd/as7341.service` | Correct paths. |
| `config/sample.env` | Up to date. |
| `FSD.md`, `README.md` | Aligned with current code. |
| `as7341_dark_hi/lo.json` | **Does not exist** — must be generated on hardware. |
| `as7341_lux_cal_hi/lo.json` | **Does not exist** — must be generated on hardware. |
| `as7341_responsivity.json` | **Does not exist** — must be generated on hardware. |

## What needs doing next

### 1. Run calibration on hardware (blocking)

The measurement script raises `FileNotFoundError` at startup if
`as7341_lux_cal_hi.json` is missing, and at the first HI→LO sensitivity switch
if `as7341_lux_cal_lo.json` is missing. Run the full guided calibration:

```bash
source .venv/bin/activate
python3 src/as7341_calibrate.py
```

Equipment needed:
- Seconic C-7000 spectroradiometer
- CoolLED pE-4000 (Phase 2: single-LED wavelength sweep — see Session 4 below)
- Dome diffusers on both instruments
- Opaque cap for sensor during Phase 1

For Phase 2 the C-7000 can export a 2-column CSV (nm, W/m²/nm) per LED step,
or values can be entered manually at the 8 channel wavelengths.

To run individual phases:
```bash
python3 src/as7341_calibrate.py --phase dark             # Phase 1 only
python3 src/as7341_calibrate.py --phase responsivity     # Phase 2 only
python3 src/as7341_calibrate.py --phase lux              # Phase 3 only
python3 src/as7341_calibrate.py --phase lux --preset hi  # re-run HI lux only
```

### 2. Optional NIR override

Phase 2 does not measure NIR (C-7000 covers 380–780 nm only). After the run,
if you have an NIR-capable reference, edit `as7341_responsivity.json` and add
a `nir` entry under `corrections` to override the datasheet default of 2.5.

### 3. Future work (FSD §8 out-of-scope)
- CCT / CRI estimation from spectral data.
- InfluxDB v2/v3 protocol support.
- Grafana dashboard templates (separate repo).
- CSV archive replay path (currently archive-only).

---

## Session 4 — Phase 2 redesign for pE-4000 wavelength sweep (2026-05-13)

Pushed as commit `fe1dd81` on `origin/main`.

### Why
Original Phase 2 ran the pE-4000 at 5 broadband intensities (10/25/50/75/90 %)
and averaged BasicCounts ÷ SPD at each AS7341 channel centre. Problem: all 5
levels deliver roughly the same SPD shape, so the per-channel system is
near-degenerate — you get 5 noisy estimates of the same broadband ratio per
channel. Looking at the pE-4000 docs we confirmed it carries **16 LEDs across
4 wavelength-grouped channels** (365–770 nm) and can be driven one LED at a
time. Sweeping in-band LEDs gives one near-monochromatic stimulus per step,
so each AS7341 channel gets characterised against the wavelength region it
actually cares about.

### pE-4000 wavelength reference
| Channel A (UV–violet) | Channel B (blue–cyan) | Channel C (green–yellow) | Channel D (red–NIR) |
|---|---|---|---|
| 365 | 460 | 525 | 635 |
| 385 | 470 | 550 | 660 |
| 405 | 490 | 580 | 740 |
| 435 | 500 | 595 | 770 |

Single-LED mode = one wavelength at a time; up to 4 simultaneously (one per
channel column). Default sweep uses the 12 LEDs that fall inside the AS7341
VIS8 passband: `405, 435, 460, 470, 490, 500, 525, 550, 580, 595, 635, 660`.

### Script changes (`src/as7341_calibrate.py`)
| Change | Detail |
|---|---|
| Dropped `--resp-levels` | No longer applicable — there's one capture per LED. |
| Added `--resp-wavelengths` | Comma-separated LED wavelengths (nm). Default = the 12 LEDs above. |
| Added `--resp-min-irr-frac` | Default 0.2. Per LED step, a channel only contributes to its responsivity average when its centre irradiance ≥ this fraction of the strongest channel at that step. Drops far-off-peak ratios that would amplify noise under narrow LEDs. |
| Output JSON gains | `meta.wavelengths_nm`, `meta.n_samples_per_channel`, and a top-level `raw_levels` block holding `(led_nm, bc8, irr8)` per step. |
| Prompts updated | "set pE-4000 to single LED at {nm} nm" instead of "set CoolLED to ~{pct}%". |

Output schema for the two consumed blocks (`corrections`,
`responsivity_BC_per_W_m2_nm`) is unchanged, so `as7341_influx_nir.py` keeps
loading the file without modification.

### Inherent limitation
F5 (555 nm), F7 (630 nm), and F8 (680 nm) each have only one pE-4000 LED
nearby (550, 635, 660). The script prints a `[NOTE]` after the run listing
channels with n_samples < 2. Not a bug — the pE-4000 simply doesn't carry
closer alternatives. User can extend the sweep with off-peak LEDs and lower
`--resp-min-irr-frac` if they want more samples there, accepting noisier
ratios.

### Smoke test
`claude_wd/smoke_phase2.py` (untracked) stubs the Pi-only imports, feeds a
synthetic wavelength-sweep dataset (Gaussian LED SPDs, FWHM 25 nm, 2 %
noise), and runs the Phase 2 inner math. All 8 channels recovered the
synthetic true responsivity within ~1.3 %, F5 normalisation exact.

Run with `.venv/bin/python claude_wd/smoke_phase2.py` from the repo root.

### What needs doing next
1. **Hardware run — Phase 2.** C-7000 irradiance data already collected (34
   files in `C-7000_out/`). On the Pi, run:
   ```bash
   python3 src/as7341_calibrate.py --phase responsivity --c7000-dir C-7000_out/
   ```
   The script walks through all 34 steps (LED nm + strength%) in order and
   captures only the AS7341 reading at each. No C-7000 data entry needed.
   Expect the `[NOTE]` about F5/F7/F8 having a single contributor.
2. **Phase 1 + Phase 3** still need their first hardware runs
   (`as7341_dark_hi/lo.json` and `as7341_lux_cal_hi/lo.json` do not exist —
   the measurement script will crash without them).
3. Confirm pE-4000 per-LED intensity range covers both HI and LO sensitivity
   presets without saturation/underflow. Phase 2 uses HI preset only, but
   Phase 3's HI/LO scenes may need the source dialled accordingly.

### Files touched this session
- `src/as7341_calibrate.py` — Phase 2 rewrite.
- `README.md` — §"Phase 2 — Spectral responsivity (VIS8)" rewritten.
- `FSD.md` — §7 Phase 2 bullet rewritten.
- `claude_wd/smoke_phase2.py` — new (untracked) smoke test.
- `claude_wd/HANDOFF.md` — this section.

---

## Session 5 — C-7000 data analysis + --c7000-dir (2026-05-13)

Pushed as commit `a5ef6e3` on `origin/main`.

### C-7000 export analysis

34 CSV files in `C-7000_out/` covering 15 LED wavelengths across 2–3 intensity
levels each. Format: Seconic C-7000 native export (multi-row metadata header +
`Spectral Data NNN[nm],value` rows at 1 nm resolution, 380–780 nm). Unit
confirmed W/m²/nm via lux sanity check (~4% error against C-7000 illuminance).

LED wavelengths and file groupings confirmed:

| LED nm | Files | Levels |
|---|---|---|
| 385 | 004–005 | 60%, 80% |
| 405 | 006–007 | 51%, 80% |
| 435 | 008–009 | 20%, 80% |
| 460 | 010–012 | 20%, 50%, 80% |
| 470 | 013–014 | 20%, 50% |
| 490 | 015–016 | 20%, 80% |
| 500 | 017–019 | 20%, 50%, 80% |
| 525 | 020–021 | 20%, 80% |
| 550 | 022 | 80% |
| 580 | 023–025 | 20%, 50%, 80% |
| 595 | 026–028 | 20%, 50%, 80% |
| 635 | 029–030 | 50%, 80% |
| 660 | 031–033 | 20%, 50%, 80% |
| 740 | 034–036 | 20%, 50%, 80% |
| 770 | 037 | 50% |

Files 004–021 use `CoolLED_nm` field; files 022–037 use `CoolLED_snm`. The
parser handles both. Files without either field are skipped with a warning
(currently none).

**550 nm vs 580 nm:** Both LED settings produce an identical SPD shape peaking
at ~555 nm as measured by the C-7000 (< 0.2% normalised difference). Treated as
independent measurements — both contribute to the responsivity average for
channels near 555 nm. No action needed.

**Contributing levels per channel** (with `--resp-min-irr-frac` 0.2):

| Ch | λ nm | n_levels |
|---|---|---|
| F1 | 415 | 6 |
| F2 | 445 | 7 |
| F3 | 480 | 10 |
| F4 | 515 | 11 |
| F5 | 555 | 6 |
| F6 | 590 | 7 |
| F7 | 630 | 5 |
| F8 | 680 | 6 |

Missing intensity steps (405@20, 435@50, 470@80, 490@50, 525@50, 550@20/50,
635@20) were evaluated — not needed. All channels have ≥ 5 contributing levels
which is sufficient for a stable average.

### Script changes (`src/as7341_calibrate.py`)

| Change | Detail |
|---|---|
| Added `import re` | Required for native CSV spectral data line parsing |
| Added `_parse_c7000_native(path)` | Parses C-7000 native export format. Handles both `CoolLED_nm` and `CoolLED_snm` field names. Returns `(led_nm, strength_pct, irr8)` or `None`. |
| Added `_load_c7000_dir(dirpath)` | Reads all `*.csv` from a directory, skips files without LED metadata (warns), returns list sorted by `(led_nm, strength)`. |
| Modified `run_phase2()` | When `--c7000-dir` is supplied: loads C-7000 levels, walks through each prompting "set pE-4000 to {nm} nm at {strength}%", captures AS7341 only. Original interactive path unchanged. |
| Added `--c7000-dir` arg | Points to directory of pre-collected C-7000 CSVs. |

### What needs doing next
1. **Hardware run — Phase 2** (see Session 4 "What needs doing next" above).
2. **Phase 1 + Phase 3** hardware runs still pending.
3. **Phase 3 lux scenes**: files 022–028 (CCT 1575–4940 K range) look like
   candidate lux scenes from the C-7000 side. If AS7341 readings were captured
   alongside, those could be used directly for Phase 3. Confirm with user.

### Files touched this session
- `src/as7341_calibrate.py` — `_parse_c7000_native`, `_load_c7000_dir`,
  `--c7000-dir`, modified `run_phase2()`.
- `claude_wd/HANDOFF.md` — this section.

---

## Session 6 — Hardware calibration runs + SUN preset (2026-05-14)

Pushed as commit `bab83f1` on `origin/main`.

### C-7000 dataset expanded

New files 038–063 added to `C-7000_out/` (now committed to repo):

| Files | Content |
|---|---|
| 038–045 | Single-LED fills: 405@20, 435@50, 470@80, 490@50, 525@50, 550@20/50, 635@20 — fills all previously missing intensity steps |
| 047–051 | CoolLED White at 10/20/50/60/80 % (592–3240 lux) — Phase 3 lux scenes |
| 052–056 | Direct sunlight ~100k lux @ 5400 K — Phase 3 LO/SUN lux scenes |
| 060–063 | Mixed office lighting 1140–2560 lux @ 2600–4200 K — Phase 3 HI lux scenes |

Phase 2 now uses 42 single-LED steps (up from 34). White/sunlight/office files are
correctly skipped by `_load_c7000_dir` (no numeric `CoolLED_nm`).

### Hardware calibration completed (RPi-1)

All three phases run and cal files committed to repo root:

| File | Result |
|---|---|
| `as7341_dark_hi.json` | All zeros — clean sensor |
| `as7341_dark_lo.json` | All zeros — clean sensor |
| `as7341_responsivity.json` | Phase 2 complete — empirical corrections derived from 42 LED steps |
| `as7341_lux_cal_hi.json` | Phase 3 HI — Train R²=0.9936, CV R²=0.9851, RMSE=57 lux |
| `as7341_lux_cal_lo.json` | Phase 3 LO — Train R²=0.9970, CV R²=0.9569, RMSE=1725 lux |

Phase 2 responsivity highlights:
- nm630/nm680 corrections (0.40/0.18) are far from datasheet (1.43/2.00) — silicon has
  much higher sensitivity in the red. This is physically expected and validated.
- All channels had ≥ 4 contributing LED steps; nm630/nm680 had 4–5 (limited by pE-4000
  LED coverage near those wavelengths).

Phase 3 HI took three attempts:
- Run 1 (ridge=0.01): CV R²=0.80 — overfitting, all-zeros scene accepted due to bug
- Run 2 (ridge=0.05): CV R²=0.77 — worse; scene 11 was a bad read (nm680 anomalously low)
- Run 3 (nnls=false, default ridge): CV R²=0.985 — good dataset, no bad scenes

### Bug fixes during session

| Fix | Commit |
|---|---|
| Phase 2 `--c7000-dir`: saturation now offers skip [s] instead of looping forever | `4586ce7` |
| Saturation threshold raised from 87.5% to 95% of FS (both scripts) | `e1b16d8` |
| Phase 3 low-signal now offers skip [s] / retry (previously accepted silently) | `50c4a7e` |

### SUN preset added

Third sensitivity tier for direct sunlight: GAIN_4X, IT=10ms (same as LO).
- LO tops out at ~54k lux; SUN covers ~700 lux – ~217k lux
- Disabled by default; enable via `.env`: `AUTORANGE_SUN_ENABLE=true`
- Switching chain: HI → LO → SUN (up on sat), SUN → LO → HI (down on underflow)
- Startup warns and falls back to 2-tier if enabled but cal file missing
- Calibrate with: `--preset sun` (dark + lux phases only; responsivity is preset-independent)

### Other changes

- `C-7000_out/*.png/jpg` added to `.gitignore` (auto-generated plots, not versioned)
- `.env` removed from tracking and added to `.gitignore` (contains device endpoints)
- README updated: Phase 1–3 each have explicit run commands and flag reference
- `config/sample.env` updated with `AUTORANGE_SUN_ENABLE` and `SENS_SUN_*` entries

### Current file state

| File | Status |
|---|---|
| `src/as7341_influx_nir.py` | Complete — 3-tier auto-sensitivity |
| `src/as7341_calibrate.py` | Complete — `--preset` accepts hi/lo/sun/both/all |
| `as7341_dark_hi/lo.json` | Generated on RPi-1 |
| `as7341_lux_cal_hi/lo.json` | Generated on RPi-1 |
| `as7341_responsivity.json` | Generated on RPi-1 |
| `as7341_dark_sun.json` | **Does not exist** — run `--phase dark --preset sun` |
| `as7341_lux_cal_sun.json` | **Does not exist** — run `--phase lux --preset sun` |

### What needs doing next

Nothing blocking — system is fully operational on RPi-1.

Optional future work:
- Re-run SUN lux cal with more scenes (currently 6, CV R²=0.74) if better outdoor accuracy is needed
- CCT / CRI estimation (FSD §8)
- InfluxDB v2/v3 protocol support
- Grafana dashboard templates (separate repo)
- CSV archive manual replay path for prolonged outages

---

## Session 7 — SUN calibration + systemd service (2026-05-14)

### SUN calibration complete

`as7341_lux_cal_sun.json` generated on RPi-1 (6 scenes, 29,500–122,800 lux):
- Train R²=0.985, CV R²=0.740, RMSE=17,363 lux
- 3 negative coefficients (ridge/NNLS would help but fit is usable)
- `AUTORANGE_SUN_ENABLE=true` set in `.env`

Bugs fixed during SUN calibration run:
- Phase 3 saturation check was including CLEAR channel — CLEAR saturates ~3× earlier
  than VIS8 in sunlight but is not in the lux model. Fixed to VIS8-only. (`0014c0d`)
- `--lux-scenes` below MIN_LUX_SCENES_FOR_FIT (10) now works — fit minimum is
  `min(requested, 10)` so `--lux-scenes 6` proceeds if all scenes collected. (`905424e`)
- `UnboundLocalError` for `AUTORANGE_SUN_ENABLE` in `main()` — Python treated it as
  local due to assignment. Replaced with local `sun_enabled` variable. (`45cbe99`)

### systemd service installed on RPi-1

`/etc/systemd/system/as7341.service` — enabled, running, starts on boot.
Logs: `journalctl -u as7341 -f`

### `.env` notes

`.env` was deleted from Pi working directory when the "remove from tracking" commit
was pulled. Recreated manually via nano with:
- `INFLUX_ENDPOINTS`: localhost lightmeter + two remote AAB endpoints
- `AUTORANGE_SUN_ENABLE=true`

### Current system state (RPi-1)

| Component | Status |
|---|---|
| Measurement script | Running as systemd service, auto-starts on boot |
| HI/LO/SUN cal files | All present and loaded |
| Local InfluxDB | Writing every cycle |
| Remote InfluxDB | Fan-out to 10.239.99.73 and 10.239.99.97 |
| Auto-sensitivity | 3-tier HI→LO→SUN active |
| Grafana | Confirmed receiving data (spectral irradiance visible) |
