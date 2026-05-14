#!/usr/bin/env python3
"""
AS7341 guided calibration — three phases:
  Phase 1  Dark capture for HI and/or LO sensitivity preset
  Phase 2  Spectral responsivity using Seconic C-7000 + CoolLED pE-4000
  Phase 3  Lux calibration for HI and/or LO preset against C-7000 lux readings

Run with --phase all (default) to complete the full calibration in one session,
or run individual phases separately.

Output files (written to --out-dir, default: project root):
  as7341_dark_hi.json        Phase 1 HI
  as7341_dark_lo.json        Phase 1 LO
  as7341_responsivity.json   Phase 2
  as7341_lux_cal_hi.json     Phase 3 HI
  as7341_lux_cal_lo.json     Phase 3 LO
"""

import argparse, csv, datetime, json, random, re, time
from pathlib import Path
from statistics import median

import board
import numpy as np
from adafruit_as7341 import AS7341, Gain

# ============================
# Constants
# ============================
BANDS8 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680"]
BANDS9 = BANDS8 + ["nir"]
WLS8   = [415, 445, 480, 515, 555, 590, 630, 680]

GAIN_MULT = {
    Gain.GAIN_0_5X:0.5,  Gain.GAIN_1X:1.0,   Gain.GAIN_2X:2.0,   Gain.GAIN_4X:4.0,
    Gain.GAIN_8X:8.0,    Gain.GAIN_16X:16.0, Gain.GAIN_32X:32.0, Gain.GAIN_64X:64.0,
    Gain.GAIN_128X:128.0,Gain.GAIN_256X:256.0,Gain.GAIN_512X:512.0,
}

# Default preset settings — match SENS_HI / SENS_LO / SENS_SUN in as7341_influx_nir.py
PRESETS = {
    "hi":  {"integration_time_ms": 50,  "gain_str": "GAIN_256X"},
    "lo":  {"integration_time_ms": 10,  "gain_str": "GAIN_16X"},
    "sun": {"integration_time_ms": 10,  "gain_str": "GAIN_4X"},
}

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================
# Shared helpers
# ============================
def ms_to_atime_astep(target_ms):
    total = target_ms / 2.78e-3
    for atime in range(256):
        astep = total / (atime + 1) - 1
        if 0 <= astep <= 65534:
            return atime, int(round(astep))
    raise ValueError(f"Cannot achieve {target_ms}ms integration time")

def integration_time_ms(atime, astep):
    return (atime + 1) * (astep + 1) * 2.78e-3

def adc_fullscale(atime, astep):
    fs = (atime + 1) * (astep + 1)
    return min(fs, 65535)

def current_gain_mult(s):
    return float(GAIN_MULT.get(s.gain, 1.0))

def read_once(s):
    vis = [float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm),
           float(s.channel_515nm), float(s.channel_555nm), float(s.channel_590nm),
           float(s.channel_630nm), float(s.channel_680nm)]
    return vis, float(s.channel_clear), float(s.channel_nir)

def avg_frames(s, n):
    acc = [0.0]*8; acc_c = 0.0; acc_n = 0.0
    for _ in range(max(1, n)):
        vis, c, nir = read_once(s)
        for i in range(8): acc[i] += vis[i]
        acc_c += c; acc_n += nir
    return [x/n for x in acc], acc_c/n, acc_n/n

def apply_preset(s, preset_name):
    p = PRESETS[preset_name]
    atime, astep = ms_to_atime_astep(p["integration_time_ms"])
    s.atime = atime; s.astep = astep
    s.gain = getattr(Gain, p["gain_str"])
    it_ms = integration_time_ms(atime, astep)
    fs    = adc_fullscale(atime, astep)
    gnum  = current_gain_mult(s)
    return atime, astep, it_ms, fs, gnum

def load_dark(path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)

def dark_offsets(darkJ, s, atime, astep):
    if darkJ is None:
        return [0]*8, 0, 0
    meta = darkJ.get("meta")
    ok = (meta and str(meta.get("gain","")) == str(s.gain)
          and int(meta.get("atime",-1)) == atime
          and int(meta.get("astep",-1)) == astep)
    if not ok:
        print("[WARN] Dark file meta does not match current settings — using zero offsets.")
    dv = [int(darkJ.get(b, 0)) if ok else 0 for b in BANDS8]
    dc = int(darkJ.get("clear", 0)) if ok else 0
    dn = int(darkJ.get("nir",   0)) if ok else 0
    return dv, dc, dn

def header(text):
    bar = "=" * (len(text) + 4)
    print(f"\n{bar}\n  {text}\n{bar}")

def prompt(msg, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val if val else (str(default) if default is not None else "")

# ============================
# Fitting helpers (from as7341_calibrate_lux.py)
# ============================
def fit_ols(Phi, y):
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return beta

def fit_ridge(Phi, y, alpha):
    A = Phi.T @ Phi + alpha * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ y)

def fit_nnls(Phi, y, iters=4000):
    """Projected-gradient NNLS with free intercept.

    Phi[:,0] is assumed all-ones (intercept column). The intercept is updated
    jointly with the weights but is not constrained; only Phi[:,1:] coefficients
    are projected to be non-negative.

    The weight columns are normalised to unit standard deviation before solving
    so a single global step size works across both the gentle intercept gradient
    and the much steeper weight gradients (lux ~10–100k against BasicCounts
    ~1–1000 makes the un-normalised problem ill-conditioned).
    """
    scales = Phi[:, 1:].std(axis=0, ddof=0)
    scales = np.where(scales < 1e-12, 1.0, scales)
    Phi_n = Phi.copy()
    Phi_n[:, 1:] = Phi[:, 1:] / scales

    PtP = Phi_n.T @ Phi_n
    Pty = Phi_n.T @ y
    eigvals = np.linalg.eigvalsh(PtP)
    L = 2.0 * float(eigvals[-1])
    lr = 1.0 / max(L, 1e-12)

    beta = np.zeros(Phi_n.shape[1])
    for _ in range(iters):
        grad = 2.0 * (PtP @ beta - Pty)
        beta -= lr * grad
        np.maximum(beta[1:], 0.0, out=beta[1:])

    beta[1:] /= scales
    return beta

def fit_lux(Phi, y, ridge, nnls):
    if nnls:
        return fit_nnls(Phi, y)
    if ridge > 0:
        return fit_ridge(Phi, y, ridge)
    return fit_ols(Phi, y)

def kfold_cv(Phi, y, k, ridge, nnls):
    n = len(y)
    idx = list(range(n)); random.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    preds = np.zeros(n)
    for fold in range(k):
        val = folds[fold]
        train = [i for i in range(n) if i not in val]
        b = fit_lux(Phi[train], y[train], ridge, nnls)
        preds[val] = Phi[val] @ b
    rmse = float(np.sqrt(np.mean((y - preds)**2)))
    ss   = float(np.sum((y - np.mean(y))**2))
    r2   = 1.0 - float(np.sum((y - preds)**2)) / ss if ss > 0 else 1.0
    return rmse, r2

def mad_filter(Phi, y, k):
    beta = fit_ols(Phi, y)
    resid = y - Phi @ beta
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)) * 1.4826)
    if mad == 0:
        return Phi, y
    keep = np.abs(resid - med) <= k * mad
    if np.count_nonzero(keep) < max(4, Phi.shape[1] + 1):
        return Phi, y
    n_removed = int(np.count_nonzero(~keep))
    if n_removed:
        print(f"  MAD filter removed {n_removed} outlier(s).")
    return Phi[keep], y[keep]

# ============================
# Phase 1 — Dark capture
# ============================
def run_phase1(s, args):
    header("PHASE 1: Dark Calibration")
    print("You will capture dark offsets for each sensitivity preset.")
    print("Keep the sensor covered throughout both captures.")

    presets = {"both": ["hi", "lo"], "all": ["hi", "lo", "sun"]}.get(args.preset, [args.preset])
    for preset_name in presets:
        p = PRESETS[preset_name]
        atime, astep, it_ms, fs, _ = apply_preset(s, preset_name)
        out_path = Path(args.out_dir) / f"as7341_dark_{preset_name}.json"

        print(f"\n--- Preset {preset_name.upper()}: gain={s.gain}, IT={it_ms:.1f}ms "
              f"(ATIME={atime}, ASTEP={astep}) ---")
        input("Cover the sensor completely and press Enter when ready...")

        print(f"Warmup ({args.dark_warmup} frames)...", end="", flush=True)
        for _ in range(args.dark_warmup):
            read_once(s)
            time.sleep(0.01)
        print(" done.")

        buckets = [[] for _ in range(8)]; cb = []; nb = []
        print(f"Collecting {args.dark_samples} frames:", end="", flush=True)
        for i in range(args.dark_samples):
            vis, c, nir = read_once(s)
            for j in range(8): buckets[j].append(vis[j])
            cb.append(c); nb.append(nir)
            if (i + 1) % 20 == 0:
                print(f" {i+1}", end="", flush=True)
            time.sleep(0.005)
        print(" done.")

        vis_med   = [int(round(median(b))) for b in buckets]
        clear_med = int(round(median(cb)))
        nir_med   = int(round(median(nb)))

        out = {
            "meta": {
                "gain": str(s.gain), "atime": atime, "astep": astep,
                "tint_ms": it_ms, "samples": args.dark_samples,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            "clear": clear_med,
        }
        for name, val in zip(BANDS9, vis_med + [nir_med]):
            out[name] = val

        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved {out_path.name}  clear={clear_med} nir={nir_med} "
              f"vis={vis_med}")

# ============================
# Phase 2 — Spectral responsivity
# ============================
def _read_c7000_csv(path):
    """Parse a 2-column CSV (wavelength_nm, irradiance_W_m2_nm). Returns (wls, irr) arrays sorted by wavelength."""
    wls = []; irr = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                w = float(row[0]); v = float(row[1])
                wls.append(w); irr.append(v)
            except ValueError:
                continue  # skip header rows
    if len(wls) < 2:
        raise ValueError("CSV must have at least 2 numeric rows (wavelength_nm, irradiance)")
    wls, irr = (list(t) for t in zip(*sorted(zip(wls, irr))))
    return wls, irr

def _interpolate(wls, irr, target_nm):
    """Linear interpolation of irr at target_nm from sorted (wls, irr) lists."""
    for i in range(len(wls) - 1):
        if wls[i] <= target_nm <= wls[i+1]:
            t = (target_nm - wls[i]) / (wls[i+1] - wls[i])
            return irr[i] + t * (irr[i+1] - irr[i])
    raise ValueError(f"Wavelength {target_nm}nm is outside CSV range "
                     f"({wls[0]}–{wls[-1]}nm)")

DEFAULT_RESP_WAVELENGTHS = [405, 435, 460, 470, 490, 500, 525, 550, 580, 595, 635, 660]

def _parse_c7000_native(path):
    """Parse a Seconic C-7000 native export CSV (multi-row header format).

    Recognises both 'CoolLED_nm' and 'CoolLED_snm' field names.
    Returns (led_nm, strength_pct, irr8) where irr8 is W/m²/nm at WLS8,
    or None if the file carries no CoolLED LED-wavelength metadata.
    """
    led_nm = None
    strength = None
    spd = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        for field in ("CoolLED_nm,", "CoolLED_snm,"):
            if line.startswith(field):
                val = line.split(",")[1].strip()
                if val:
                    try:
                        led_nm = int(round(float(val)))
                    except ValueError:
                        pass
        if line.startswith("CoolLED_strength,"):
            val = line.split(",")[1].strip()
            if val:
                try:
                    strength = int(round(float(val)))
                except ValueError:
                    pass
        m = re.match(r"Spectral Data (\d+)\[nm\],([\d.eE+\-]+)", line)
        if m:
            spd[int(m.group(1))] = float(m.group(2))
    if led_nm is None or not spd:
        return None
    return (led_nm, strength, [spd.get(w, 0.0) for w in WLS8])

def _load_c7000_dir(dirpath):
    """Load all C-7000 native CSVs from dirpath.

    Returns list of (led_nm, strength_pct, irr8) sorted by (led_nm, strength).
    Files without CoolLED LED-wavelength metadata are skipped with a warning.
    """
    levels = []
    skipped = []
    for f in sorted(Path(dirpath).glob("*.csv")):
        result = _parse_c7000_native(f)
        if result is None:
            skipped.append(f.name)
        else:
            levels.append(result)
    if skipped:
        print(f"[WARN] Skipped {len(skipped)} CSV(s) with no CoolLED metadata: "
              f"{', '.join(skipped)}")
    levels.sort(key=lambda x: (x[0], x[1] or 0))
    return levels

def _get_irradiance_for_level(level_num, n_levels, led_nm):
    """Prompt for C-7000 spectral data with the source set to a single pE-4000 LED.

    Returns the irradiance (W/m²/nm) at the 8 AS7341 channel centers.
    """
    print(f"\n  C-7000 spectral data for level {level_num}/{n_levels} (LED {led_nm} nm):")
    print("  Option A — provide a CSV file exported from the C-7000")
    print("             (2 columns: wavelength_nm, irradiance_W_m2_nm)")
    print("  Option B — enter values manually for each channel wavelength")
    choice = input("  CSV file path (or press Enter for manual): ").strip()

    if choice:
        try:
            wls, irr = _read_c7000_csv(choice)
            vals = [_interpolate(wls, irr, nm) for nm in WLS8]
            print("  Interpolated irradiance (W/m²/nm):")
            for nm, v in zip(WLS8, vals):
                print(f"    {nm}nm: {v:.6e}")
            return vals
        except Exception as e:
            print(f"  [ERR] Could not read CSV: {e}")
            print("  Falling back to manual entry.")

    vals = []
    print("  Enter C-7000 irradiance (W/m²/nm) at each wavelength")
    print("  (channels far from the LED will be near zero — enter 0 if below floor):")
    for nm in WLS8:
        while True:
            try:
                v = float(input(f"    {nm}nm: ").strip())
                vals.append(v)
                break
            except ValueError:
                print("    Please enter a number.")
    return vals

def _parse_wavelengths(spec):
    out = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(round(float(tok))))
    if not out:
        raise ValueError("--resp-wavelengths must list at least one wavelength in nm")
    return out

def run_phase2(s, args):
    header("PHASE 2: Spectral Responsivity")
    print("Equipment: AS7341 with dome diffuser + Seconic C-7000 side-by-side.")
    print("Light source: CoolLED pE-4000 in single-LED mode (one wavelength at a time).")

    if args.c7000_dir:
        c7000_levels = _load_c7000_dir(args.c7000_dir)
        if not c7000_levels:
            raise RuntimeError(f"No valid C-7000 CSV files found in '{args.c7000_dir}'")
        wavelengths = sorted(set(lv[0] for lv in c7000_levels))
        print(f"\nLoaded {len(c7000_levels)} C-7000 measurements from '{args.c7000_dir}'")
        print(f"LED wavelengths: {', '.join(str(w) for w in wavelengths)} nm")
        print("C-7000 irradiance pre-loaded — only AS7341 captures needed.")
    else:
        wavelengths = _parse_wavelengths(args.resp_wavelengths)
        print(f"Wavelength sweep ({len(wavelengths)} LEDs): "
              f"{', '.join(str(w) for w in wavelengths)} nm")
        print("The C-7000 data can be entered as a CSV export or typed manually.")

    atime, astep, it_ms, fs, gnum = apply_preset(s, "hi")
    print(f"\nUsing HI preset: gain={s.gain}, IT={it_ms:.1f}ms")

    dark_path = Path(args.out_dir) / "as7341_dark_hi.json"
    darkJ = load_dark(dark_path)
    if darkJ is None:
        print(f"[WARN] {dark_path.name} not found — dark correction inactive for this phase.")
    dv, dc, _ = dark_offsets(darkJ, s, atime, astep)
    denom = max(1e-9, gnum * it_ms)
    sat_th = 0.95 * fs; lo_th = 0.003 * fs

    level_data = []  # list of dicts: {"led_nm", "bc8", "irr8"}

    if args.c7000_dir:
        n = len(c7000_levels)
        for k, (led_nm, strength, irr8) in enumerate(c7000_levels):
            str_label = f" at {strength}%" if strength is not None else ""
            print(f"\n--- Step {k+1}/{n}: set pE-4000 to {led_nm} nm{str_label} ---")
            skip = False
            while True:
                input("  Press Enter to capture AS7341 reading...")
                vis_raw, _, _ = avg_frames(s, args.resp_avg)
                peak = max(vis_raw)
                if peak >= sat_th:
                    print(f"  [WARN] Saturating (peak={int(peak)}, FS={fs}) — "
                          "cannot use this reading (C-7000 irradiance was at the original intensity).")
                    choice = input("  Skip this step [s] or retry after adjusting [Enter]: ").strip().lower()
                    if choice == "s":
                        skip = True
                        break
                    continue
                if peak <= lo_th:
                    print(f"  [WARN] Signal too low (peak={int(peak)}) — increase intensity.")
                    continue
                break
            if skip:
                print(f"  Step skipped.")
                continue
            vis = [max(0.0, v - d) for v, d in zip(vis_raw, dv)]
            bc8 = [v / denom for v in vis]
            print(f"  BasicCounts: { {b: round(x,4) for b,x in zip(BANDS8, bc8)} }")
            level_data.append({"led_nm": led_nm, "bc8": bc8, "irr8": irr8})
    else:
        n = len(wavelengths)
        for k, led_nm in enumerate(wavelengths):
            print(f"\n--- Level {k+1}/{n}: set pE-4000 to single LED at {led_nm} nm ---")
            print("  Dial intensity so the AS7341 peak channel sits well below saturation")
            print("  but clearly above the noise floor. Take your time.")
            while True:
                input("  Press Enter to capture AS7341 reading...")
                vis_raw, _, _ = avg_frames(s, args.resp_avg)
                peak = max(vis_raw)
                if peak >= sat_th:
                    print(f"  [WARN] Saturating (peak={int(peak)}, FS={fs}) — reduce CoolLED intensity.")
                    continue
                if peak <= lo_th:
                    print(f"  [WARN] Signal too low (peak={int(peak)}) — increase CoolLED intensity.")
                    continue
                break
            vis = [max(0.0, v - d) for v, d in zip(vis_raw, dv)]
            bc8 = [v / denom for v in vis]
            print(f"  BasicCounts: { {b: round(x,4) for b,x in zip(BANDS8, bc8)} }")
            irr8 = _get_irradiance_for_level(k+1, n, led_nm)
            level_data.append({"led_nm": led_nm, "bc8": bc8, "irr8": irr8})

    # Per-channel responsivity: average BC/E only over levels where the channel
    # actually sees significant LED power. Channels far from the LED have near-zero
    # SPD at their center and the ratio would amplify noise, so filter by relative
    # SPD strength at this level.
    frac = float(args.resp_min_irr_frac)
    raw_resp = [None] * 8
    n_used   = [0] * 8

    for i in range(8):
        vals = []
        for lvl in level_data:
            bc = lvl["bc8"][i]
            ir = lvl["irr8"][i]
            max_ir = max(lvl["irr8"])
            if max_ir <= 0 or ir <= 0 or bc <= 0:
                continue
            if (ir / max_ir) < frac:
                continue
            vals.append(bc / ir)
        if not vals:
            raise ValueError(
                f"No valid responsivity data for channel {BANDS8[i]} "
                f"(λ={WLS8[i]} nm). Add an LED near this wavelength or lower "
                f"--resp-min-irr-frac (currently {frac})."
            )
        raw_resp[i] = sum(vals) / len(vals)
        n_used[i]   = len(vals)

    # Normalise to F5/555 nm = 1.0
    ref = raw_resp[WLS8.index(555)]
    corrections = [ref / r for r in raw_resp]

    out_path = Path(args.out_dir) / "as7341_responsivity.json"
    result = {
        "corrections": {b: round(c, 6) for b, c in zip(BANDS8, corrections)},
        "responsivity_BC_per_W_m2_nm": {b: r for b, r in zip(BANDS8, raw_resp)},
        "meta": {
            "gain": str(s.gain), "it_ms": it_ms,
            "wavelengths_nm": wavelengths,
            "n_samples_per_channel": {b: n for b, n in zip(BANDS8, n_used)},
            "min_irr_frac": frac,
            "resp_avg_frames": args.resp_avg,
            "instrument": "Seconic C-7000",
            "source": "CoolLED pE-4000, single-LED mode (wavelength sweep)",
            "units": "responsivity_BC_per_W_m2_nm in BasicCounts per (W/m^2/nm)",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
        "raw_levels": level_data,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\nEmpirical responsivity corrections (normalised to 555nm = 1.0):")
    datasheet = [2.0, 1.67, 1.33, 1.11, 1.0, 1.11, 1.43, 2.0]
    print(f"  {'Channel':>8}  {'Measured':>10}  {'Datasheet':>10}  {'Diff':>8}  {'n':>4}")
    for b, meas, ds, n in zip(BANDS8, corrections, datasheet, n_used):
        diff = meas - ds
        print(f"  {b:>8}  {meas:>10.4f}  {ds:>10.4f}  {diff:>+8.4f}  {n:>4}")

    print("\nAbsolute responsivity (BasicCounts per W/m^2/nm):")
    for b, r in zip(BANDS8, raw_resp):
        print(f"  {b:>8}  {r:>12.4e}")
    short = [b for b, n in zip(BANDS8, n_used) if n < 2]
    if short:
        print(f"\n[NOTE] Channels with < 2 contributing LEDs: {', '.join(short)}. "
              "Consider adding LEDs nearer those wavelengths to your sweep.")
    print(f"\nSaved {out_path.name}")
    print("Restart as7341_influx_nir.py to apply the new corrections "
          "and emit absolute irradiance per channel.")

# ============================
# Phase 3 — Lux calibration
# ============================
MIN_LUX_SCENES_FOR_FIT = 10  # 9 unknowns + 1 redundancy is the floor for a stable fit

def run_phase3(s, args):
    header("PHASE 3: Lux Calibration")
    print("Equipment: AS7341 with dome diffuser + Seconic C-7000 side-by-side,")
    print("           same angle, no shadows. Use diverse light scenes.")
    print(f"Scenes per preset: {args.lux_scenes}  |  Frames averaged: {args.lux_avg}")
    if args.lux_scenes < MIN_LUX_SCENES_FOR_FIT:
        print(f"[WARN] {args.lux_scenes} scenes is below the recommended minimum "
              f"({MIN_LUX_SCENES_FOR_FIT}). Consider --lux-scenes {MIN_LUX_SCENES_FOR_FIT}+.")

    presets = {"both": ["hi", "lo"], "all": ["hi", "lo", "sun"]}.get(args.preset, [args.preset])

    for preset_name in presets:
        random.seed(42)  # per-preset deterministic shuffle for k-fold
        p = PRESETS[preset_name]
        atime, astep, it_ms, fs, gnum = apply_preset(s, preset_name)
        print(f"\n--- Preset {preset_name.upper()}: gain={s.gain}, IT={it_ms:.1f}ms ---")

        dark_path = Path(args.out_dir) / f"as7341_dark_{preset_name}.json"
        darkJ = load_dark(dark_path)
        if darkJ is None:
            print(f"[WARN] {dark_path.name} not found — dark correction inactive.")
        dv, dc, _ = dark_offsets(darkJ, s, atime, astep)
        denom = max(1e-9, gnum * it_ms)
        sat_th = 0.95 * fs; lo_th = 0.003 * fs

        rows = []
        for i in range(1, args.lux_scenes + 1):
            print(f"\n  Scene {i}/{args.lux_scenes}")
            scene_skipped = False
            while True:
                input("  Position instruments for this scene, then press Enter to capture...")
                vis_raw, clear_raw, _ = avg_frames(s, args.lux_avg)
                peak = max(vis_raw)  # VIS8 only — CLEAR is not in the lux model
                if peak >= sat_th:
                    print(f"  [WARN] Near saturation (peak={int(peak)}, FS={fs}) — "
                          "reduce intensity or use a lower-sensitivity preset.")
                    choice = input("  Retry [Y], skip scene [s]? ").strip().lower()
                    if choice == "s":
                        scene_skipped = True
                        break
                    continue  # default = retry
                if peak <= lo_th:
                    print(f"  [WARN] Very low signal (peak={int(peak)}) — "
                          "increase intensity or use a higher-sensitivity preset.")
                    choice = input("  Retry [Y], skip scene [s]? ").strip().lower()
                    if choice == "s":
                        scene_skipped = True
                    break
                break

            if scene_skipped:
                print(f"  Scene {i} skipped (saturated).")
                continue

            vis = [max(0.0, v - d) for v, d in zip(vis_raw, dv)]
            bc8 = [max(0.0, v) / denom for v in vis]
            print(f"  BasicCounts: { {b: round(x,3) for b,x in zip(BANDS8, bc8)} }")

            while True:
                try:
                    lux = float(input("  Enter C-7000 lux reading: ").strip())
                    break
                except ValueError:
                    print("  Please enter a number.")
            rows.append({"bc8": bc8, "lux": lux})

        fit_min = min(args.lux_scenes, MIN_LUX_SCENES_FOR_FIT)
        if len(rows) < fit_min:
            print(f"\n[ERROR] Only {len(rows)} usable scene(s) collected for preset {preset_name.upper()}; "
                  f"need ≥ {fit_min}. Skipping fit — re-run this preset with more scenes.")
            continue

        X = np.array([r["bc8"] for r in rows])
        y = np.array([r["lux"] for r in rows])
        Phi = np.column_stack([np.ones(len(y)), X])

        Phi_f, y_f = mad_filter(Phi, y, args.lux_madk)
        beta = fit_lux(Phi_f, y_f, args.lux_ridge, args.lux_nnls)

        yhat = Phi_f @ beta
        rmse_train = float(np.sqrt(np.mean((y_f - yhat)**2)))
        ss = float(np.sum((y_f - np.mean(y_f))**2))
        r2_train = 1.0 - float(np.sum((y_f - yhat)**2)) / ss if ss > 0 else 1.0

        rmse_cv = r2_cv = None
        if args.lux_kfold >= 2 and len(y_f) >= args.lux_kfold:
            rmse_cv, r2_cv = kfold_cv(Phi_f, y_f, args.lux_kfold, args.lux_ridge, args.lux_nnls)

        b0 = float(beta[0]); w = list(map(float, beta[1:]))
        negs = sum(1 for c in w if c < 0)

        print(f"\n  Fit results (preset {preset_name.upper()}):")
        print(f"    Train:  RMSE={rmse_train:.3f} lux  R²={r2_train:.4f}")
        if rmse_cv is not None:
            print(f"    CV k={args.lux_kfold}: RMSE={rmse_cv:.3f} lux  R²={r2_cv:.4f}")
        if negs and not args.lux_nnls:
            print(f"    [NOTE] {negs} negative coefficient(s). "
                  "Consider --lux-ridge 0.05 or --lux-nnls to suppress.")

        out_path = Path(args.out_dir) / f"as7341_lux_cal_{preset_name}.json"
        cal = {
            "b0": b0, "w": w, "bands": BANDS8,
            "meta": {
                "gain": str(s.gain), "atime": atime, "astep": astep, "it_ms": it_ms,
                "features": BANDS8, "ridge_alpha": float(args.lux_ridge),
                "nnls": bool(args.lux_nnls),
                "mad_k": float(args.lux_madk), "kfold": args.lux_kfold,
                "n_scenes_used": len(rows),
                "rmse_train": rmse_train, "r2_train": r2_train,
                "rmse_cv": rmse_cv, "r2_cv": r2_cv,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        }
        with open(out_path, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"  Saved {out_path.name}")

# ============================
# Main
# ============================
def main():
    ap = argparse.ArgumentParser(
        description="AS7341 guided calibration (dark + responsivity + lux)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--phase",   choices=["all","dark","responsivity","lux"], default="all")
    ap.add_argument("--preset",  choices=["hi","lo","sun","both","all"], default="both",
                    help="Preset(s) to calibrate: hi, lo, sun, both (hi+lo), all (hi+lo+sun)")
    ap.add_argument("--out-dir", default=str(BASE_DIR),
                    help="Directory to write output JSON files")

    # Phase 1
    ap.add_argument("--dark-samples", type=int,   default=100)
    ap.add_argument("--dark-warmup",  type=int,   default=10)

    # Phase 2
    ap.add_argument("--resp-wavelengths", type=str,
                    default=",".join(str(w) for w in DEFAULT_RESP_WAVELENGTHS),
                    help="Comma-separated pE-4000 LED wavelengths (nm) to sweep, "
                         "one LED at a time")
    ap.add_argument("--resp-avg",     type=int,   default=20,
                    help="Frames averaged per wavelength step")
    ap.add_argument("--resp-min-irr-frac", type=float, default=0.2,
                    help="Per level, include a channel only if its irradiance at the "
                         "channel center is at least this fraction of the max channel "
                         "irradiance at that level (filters out far-off-peak noise)")
    ap.add_argument("--c7000-dir", type=str, default=None,
                    help="Directory of pre-collected Seconic C-7000 native CSV exports. "
                         "Irradiance is loaded from these files; only AS7341 captures "
                         "are needed interactively. Files must contain CoolLED_nm or "
                         "CoolLED_snm metadata.")

    # Phase 3
    ap.add_argument("--lux-scenes",   type=int,   default=12,
                    help=f"Number of calibration scenes (≥ {MIN_LUX_SCENES_FOR_FIT} recommended)")
    ap.add_argument("--lux-avg",      type=int,   default=10)
    ap.add_argument("--lux-ridge",    type=float, default=0.01,
                    help="L2 regularisation α; 0.01 stabilises small-sample fits")
    ap.add_argument("--lux-nnls",     action="store_true",
                    help="Constrain weights to be non-negative (projected gradient)")
    ap.add_argument("--lux-kfold",    type=int,   default=5)
    ap.add_argument("--lux-madk",     type=float, default=3.5)

    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("=== AS7341 Calibration Guide ===")
    print(f"Phase: {args.phase}  |  Preset: {args.preset}  |  Output: {args.out_dir}")

    i2c = board.I2C()
    s = AS7341(i2c)
    try:
        s.flicker_detection_enabled = False
    except Exception:
        pass

    if args.phase in ("all", "dark"):
        run_phase1(s, args)

    if args.phase in ("all", "responsivity"):
        run_phase2(s, args)

    if args.phase in ("all", "lux"):
        run_phase3(s, args)

    print("\n=== Calibration complete ===")

if __name__ == "__main__":
    main()
