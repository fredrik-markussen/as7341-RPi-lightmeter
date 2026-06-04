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

---

## Session 8 — PFD output, network fixes, responsivity diagnosis (2026-05-24)

### WiFi watchdog fixed

Old watchdog (`/usr/local/bin/wifi_watchdog.sh`) pinged 8.8.8.8 as internet fallback and rebooted the Pi on failure. Combined with `fake-hwclock` restoring a stale timestamp on boot, this caused the Pi to reboot repeatedly when offline — all log timestamps showed the same clock value. User replaced watchdog with a version that does soft reconnect only (no reboot). Disabled old watchdog.

Persistent journald logging was also set up so service logs survive reboots:
```
/etc/systemd/journald.conf → Storage=persistent
sudo systemctl restart systemd-journald
```
Logs now accumulate in `/var/log/journal/`. View with `journalctl -u as7341 -f`.

### PFD added as primary output

`src/as7341_influx_nir.py` now computes and writes photon flux density (µmol/m²/s, 400–700 nm) — the primary measurement for circadian/seasonal biology work. Lux is retained as secondary reference only.

Key additions:
- Constants: `BANDWIDTHS_NM = [26, 30, 36, 39, 39, 40, 50, 52]` (AS7341 FWHM per VIS8 channel) and `_PFD_SCALE = 119.7` (h × c × Nₐ in J·nm/µmol)
- `compute_pfd(irr_vis8)` — sums `irr × λ × Δλ / 119.7` over VIS8 channels
- PFD written to `LIGHT_LUX` measurement in InfluxDB as `pfd=` field
- CSV expanded to 22 columns (added `pfd`)
- Bug fixed during session: scale was initially `119700.0` (1000× too large), giving ~1.7 µmol/m²/s in direct sun. Correct value is `119.7`.

### NIR channel handling clarified

NIR (910 nm) is written to InfluxDB as its own data point but is **excluded from the VIS8 normalisation** used for `rel_intensity`. Previously it was being included in the VIS8 sum, skewing relative values.

### README — Maths section added

New section between "Software prerequisites" and "Installation" explains the full signal chain:
- Sensor layout (Adafruit board photo + AS7341 die schematic, side by side)
- Photoelectric effect / ADC basics (ATIME, ASTEP, BasicCounts formula)
- Phase 1 dark correction
- Phase 2 responsivity procedure
- `rel_intensity` normalisation
- Absolute irradiance
- PFD formula (correctly shows `/119.7`)
- Phase 3 lux model
- Illustration: `Illustrations/PE-effect.png` with link to thescienceandmathszone.com

Lux reliability warning also added: lux readings are currently unreliable — per-preset calibration models disagree at handoff points. PFD is the recommended output.

### Network: static IP → DHCP

Attempted to set static IP `10.115.217.56` on `lightmeter-hotspot` profile so Windows hosts file entry `10.115.217.56 rpi-1.local` would work without mDNS. This broke internet routing on the Pi (wrong gateway assumed `.1`; phone hotspot's actual gateway was `.50`).

Reverted to DHCP — the phone's DHCP server assigns the same IP consistently via MAC binding, so internet/routing works automatically with any phone. Windows 10/11 resolves `rpi-1.local` via built-in mDNS without Bonjour. No hosts file entry needed.

**Do not set a static IP on `lightmeter-hotspot`.** The gateway address varies between phones.

### Grafana white screen fixed

Grafana JWT token validation failed (`token issued in the future`) because the Pi clock was stuck at 2026-04-21. Root cause: `fake-hwclock` restoring a stale timestamp and chrony only having `ntp-public.uit.no` (UiT internal, unreachable on hotspot). Fixed by:
1. Adding `pool.ntp.org iburst` to `/etc/chrony/chrony.conf`
2. Switching to DHCP (restored internet routing → NTP reachable)
3. Clock now syncs on boot; `timedatectl` shows `System clock synchronized: yes`

### Responsivity recalibration — diagnosis

Field measurements against known solar irradiance show the current `as7341_responsivity.json` has errors:
- **630 nm and 680 nm underreading** — root cause is inadequate pE-4000 LED coverage. The 680 nm channel (52 nm FWHM, 654–706 nm) has no LED in the calibration sweep past 660 nm, so its responsivity coefficient is extrapolated rather than measured.
- **590 nm overreading** — likely a C-7000 CSV interpolation artifact that inflated the reference irradiance at that step.

### What needs doing next

1. **Redo Phase 2 responsivity calibration** — repeat indoor sweep carefully with C-7000 + pE-4000. A 680–700 nm LED source should be added if available to anchor the red end properly. Run:
   ```bash
   python3 src/as7341_calibrate.py --phase responsivity --c7000-dir C-7000_out/
   ```
2. **Outdoor validation** — after new responsivity file is generated, take Pi + C-7000 outside in stable direct sunlight, compare per-channel irradiance, and derive empirical correction multipliers for any remaining offsets. No tooling for this step yet — needs a new script or phase.
3. **SUN lux recalibration** — CV R²=0.740 with 6 scenes. Re-run with more scenes if lux accuracy in sunlight matters (currently lower priority than spectral accuracy).

---

## Session 9 — Band-integrated C-7000 irradiance for Phase 2 (2026-06-04)

### Real root cause of the red error

Session 8 blamed the red underreporting on "inadequate pE-4000 LED coverage."
That is only half of it. The dominant cause was in the **reference-irradiance
extraction**: `_parse_c7000_native()` took the reference for each AS7341 channel
as a single **point sample** of the C-7000 SPD at the channel center
(`spd.get(w, 0.0) for w in WLS8`). But each channel integrates over a wide
passband (FWHM 26–52 nm). When a narrow LED sits off-center — e.g. the 660 nm LED
driving the 680 nm-center / 52 nm-FWHM channel — the channel's BasicCounts are
driven by the 660 nm peak (~0.09 W/m²/nm) while the reference was read at 680 nm
where the SPD has fallen off (~0.0099). That inflates `responsivity = BC/irr`,
which makes the measurement script's `irr = BC/responsivity` under-report.

### Fix — `src/as7341_calibrate.py`

Replaced the point sample with a **Gaussian band average** over each channel's
FWHM (the C-7000 already exports the full 1 nm SPD, 380–780 nm).

| Change | Detail |
|---|---|
| `import math`; `BANDWIDTHS_NM = [26,30,36,39,39,40,50,52]` | FWHM per VIS8 channel (mirrors `as7341_influx_nir.py`). |
| New `_band_average(spd, center_nm, fwhm_nm)` | Channel-response-weighted mean of the SPD; normalised over available wavelengths so the truncated tail past 780 nm doesn't bias low. Sanity-checked: flat SPD → exact mean; off-center 660 peak at the 680 channel → 0.237 vs point-sample 0.044 (5× more of the real stimulus captured). |
| `_parse_c7000_native` | Returns `[_band_average(spd,c,fw) …]` instead of `[spd.get(w,0.0) …]`. |
| `meta.irradiance_extraction = "band_average_gaussian_fwhm"` | Provenance flag in the output cal file. |

Measurement script and output schema unchanged — band **average** (not sum)
preserves the W/m²/nm units `irr = BC/responsivity` / `compute_pfd` expect.

### Offline preview (no hardware) — `claude_wd/preview_band_integration.py`

The committed `as7341_responsivity.json` stores the live AS7341 `bc8` per step in
`raw_levels`, and its stored `irr8` IS the old point sample. The preview matches
each stored level to its C-7000 file by that point-sample fingerprint (robust to
file-set drift — the JSON has 39 levels, `C-7000_out/` now has 42 single-LED
files), recomputes irradiance with band averaging, and re-derives corrections.
Faithfulness verified: feeding the old point samples back through the preview's
recompute reproduces the committed corrections to 5 dp.

Preview result (corrections, normalised 555 nm = 1.0; all 39 levels matched):

| Ch | old | new | datasheet |
|---|---|---|---|
| nm415 | 2.67 | 5.70 | 2.00 |
| nm445 | 1.70 | 2.33 | 1.67 |
| nm480 | 1.41 | 1.57 | 1.33 |
| nm515 | 0.89 | 1.17 | 1.11 |
| nm555 | 1.00 | 1.00 | 1.00 |
| nm590 | 1.19 | 0.90 | 1.11 |
| nm630 | **0.40 → 0.72** | | 1.43 |
| nm680 | **0.18 → 0.31** | | 2.00 |

Interpretation:
- **Red improves substantially and for the right physical reason** (nm630 nearly
  doubles, nm680 ~+70%). This is the stated goal.
- "Moving toward datasheet" is **not** the success criterion — the datasheet is
  nominal and empirical silicon legitimately deviates (per Session 6). nm415 rises
  to 5.70 because the 405 nm LED peak sits just inside the narrow 415 band, so the
  channel genuinely receives more light than the 415 nm point value implied. Band
  averaging is the more physically correct reference regardless of direction.
- **nm680 is still far below where outdoor data suggests it should be (0.31).**
  Band integration cannot fully fix it because no pE-4000 LED peaks in 670–700 nm;
  the 680 channel's reference is still dominated by the 660 LED's falling tail.
  Session 8's "add a 680–700 nm source" recommendation still stands — both root
  causes are real; this fix removes the larger, free one.
- **Caveat:** edge-driven channels (415 ← 405, 590 ← 595) are the most sensitive
  to the Gaussian channel-shape assumption. Still strictly better than point
  sampling, which assumes the channel sees a single wavelength.

### What needs doing next

1. **Hardware rerun** (reuses existing C-7000 files) to regenerate
   `as7341_responsivity.json` with live captures + the new extraction:
   ```bash
   python3 src/as7341_calibrate.py --phase responsivity --c7000-dir C-7000_out/
   ```
   The committed JSON has 39 levels but `C-7000_out/` now holds 42 single-LED
   files — the rerun will use all 42.
2. **Outdoor validation** of the red channels against the C-7000 in stable
   sunlight (unchanged from Session 8 item 2).
3. **Optional 680–700 nm source** to properly anchor nm680, then re-sweep.

### Files touched this session
- `src/as7341_calibrate.py` — `import math`, `BANDWIDTHS_NM`, `_band_average()`,
  `_parse_c7000_native` band integration, `meta.irradiance_extraction` flag.
- `claude_wd/preview_band_integration.py` — new (untracked) offline preview.
- `claude_wd/HANDOFF.md` — this section.

---

## Session 10 — calRE2 dataset + 770 nm leakage fix (2026-06-04)

### New C-7000 dataset (calRE2), metadata backfilled

User re-shot the full sweep: 45 files `C-7000_out/AS7341-calRE2_001..045`, 15
pE-4000 LEDs (385–770 nm) × 25/50/100 %, replacing the old `004–063` set (which
was removed). The new exports had no CoolLED metadata; it was backfilled from the
measured Peak Wavelength + ascending illuminance per file (group = file order in
pE-4000 channel order; 550 & 580 both peak ~555 nm). `CoolLED_nm,<nm>` +
`CoolLED_strength,<pct>` inserted after each Title line. Committed as the data swap
(45 added, 56 removed).

### First hardware rerun exposed a real bug: 770 nm out-of-band leakage

The Phase 2 rerun produced unphysical corrections (blue collapsed:
nm415=0.16, nm445=0.12, nm480=0.19; should be ~datasheet ≥1.3). Root cause:

- At 770 nm the AS7341 interference filters break down — **every** channel
  responds with near-equal out-of-band leakage (step 45 BC ≈ flat 0.06–0.11
  across all 8 channels).
- The C-7000's in-band irradiance at the blue centers under a 770 nm LED is
  near-zero noise floor (~4e-4). `responsivity = BC/irr` of (real leakage)/(≈0)
  gave ratios of 150–270 that swamped the legitimate ~1.5–7 contributions.
- These slipped past `--resp-min-irr-frac` because that filter is **relative
  within a step**, and the 770 nm spectrum is flat across channels, so every
  channel passes the "≥20 % of this step's max" test.
- The OLD point-sampled cal never hit this: the 770 nm point samples were exactly
  0 at every center, so the step was auto-skipped (`max_ir <= 0`). Band
  integration turned those zeros into tiny non-zeros → no longer skipped.

### Fix — `src/as7341_calibrate.py`

| Change | Detail |
|---|---|
| `_compute_responsivity(level_data, frac, chan_frac)` | Factored out; adds a **per-channel absolute floor** `chan_frac`: a level contributes to a channel only if that channel's irradiance ≥ `chan_frac × (channel's strongest irradiance across the sweep)`. Drops leakage-only levels (770 nm excluded from every channel) while keeping legit stimuli (660/740 still feed nm680). |
| `--resp-min-chan-frac` (default 0.1) | New CLI knob for the floor. |
| `_finalize_responsivity(...)` | Shared write+print, records `min_chan_frac`. |
| `run_recompute()` / `--recompute` [`--recompute-from`] | Re-derive corrections from an existing cal file's `raw_levels` **with no hardware** — re-tune filters without repeating the 45-step sweep. Runs before sensor init in `main()`. |

### Corrected result (floor=0.1, verified offline from the run's raw_levels)

| Ch | old(point) | broken(band) | fixed(band+floor) | datasheet |
|---|---|---|---|---|
| nm415 | 2.67 | 0.16 | 6.21 | 2.00 |
| nm445 | 1.70 | 0.12 | 2.15 | 1.67 |
| nm480 | 1.41 | 0.19 | 1.47 | 1.33 |
| nm515 | 0.89 | 1.17 | 1.17 | 1.11 |
| nm555 | 1.00 | 1.00 | 1.00 | 1.00 |
| nm590 | 1.19 | 0.81 | 0.81 | 1.11 |
| nm630 | **0.40** | 0.72 | **0.71** | 1.43 |
| nm680 | **0.18** | 0.21 | **0.42** | 2.00 |

- **Red improved (the goal):** nm630 0.40→0.71, nm680 0.18→0.42.
- **Blue edge nm415 = 6.21 is high** vs datasheet 2.0 / old 2.67. Band integration
  is exact only when the Gaussian-FWHM channel model matches the true response;
  nm415 is the narrowest channel and its only LEDs (385/405/435) all peak at/below
  its band, so the model error shows up there. nm590 (0.81) and nm680 (0.42, still
  below datasheet 2.0) are limited by one-sided LED coverage — no LED peaks in
  670–700 nm. These edge channels remain the weak point; **outdoor validation
  against the C-7000 is the arbiter** before trusting blue.

### Auto-preset-drop on saturation (added same session)

The first run skipped 14 bright steps (50/100 %) that saturated the HI preset,
forcing the thin red channels onto the dimmer, noisier 25 % levels (no bias —
`BC/irr` is linear across intensities, confirmed ×1.00–1.06 spread — but lower
SNR). Since BasicCounts = (raw − dark)/(gain × IT) is **preset-independent**, a
bright LED that rails HI can be captured at LO/SUN at the same scale.

`run_phase2` (`--c7000-dir` path) now **auto-drops HI→LO→SUN on saturation** and
re-captures instead of skipping:
- `_preset_ctx(s, preset, out_dir)` applies a preset, loads its dark file, returns
  dark/denom/thresholds, and discards one settle frame after the register change.
- Each step restarts at HI (re-applied only if a prior step's drop left the sensor
  at LO/SUN — tracked via `sensor_preset`); on saturation it steps down the chain.
- Per-level `preset` recorded in `raw_levels`; `meta.presets_used` lists them.
- Only if SUN still saturates does it offer skip/retry.

LO gives ~80× headroom over HI (16× gain × 5× IT), so most drops resolve at LO in
one step. Re-run on hardware to recover the skipped steps:
```bash
python3 src/as7341_calibrate.py --phase responsivity --c7000-dir C-7000_out/
```

### What needs doing next
1. **On the Pi, regenerate** — either re-sweep (above, now recovers all 45 steps)
   or, if not re-sweeping, recompute from the saved captures (no hardware):
   ```bash
   git pull
   python3 src/as7341_calibrate.py --recompute
   sudo systemctl restart as7341
   ```
2. **Outdoor validation** of per-channel irradiance vs the C-7000 in stable
   sunlight — especially nm415 (over-correct?) and nm630/nm680 (under-correct?).
3. The 770 nm (and marginally 740 nm) LEDs add little to VIS8; the floor makes
   them harmless, so no need to drop them from future sweeps.

### Files touched this session
- `C-7000_out/` — old set removed, 45 calRE2 files added + CoolLED metadata.
- `src/as7341_calibrate.py` — `_compute_responsivity`, `_finalize_responsivity`,
  `run_recompute`, `--resp-min-chan-frac`, `--recompute[-from]`.
- `claude_wd/HANDOFF.md` — this section.
