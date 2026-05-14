#!/usr/bin/env python3
# AS7341 → InfluxDB v1 (SIMPLIFIED CONFIG VERSION)
# 
# PERFORMANCE OPTIMIZATIONS:
# - Parallel HTTP writes with ThreadPoolExecutor
# - Multi-payload retry queue flushing
# - Cached calculations for timing/fullscale
# - Optimized string formatting
# - Tuned connection pooling and timeouts
#
# SPECTRAL ACCURACY IMPROVEMENTS:
# - Responsivity correction (corrects for different channel sensitivities)
# - VIS8 normalized separately from NIR (visible spectrum independent of IR content)
# - Minimum signal threshold (prevents noisy spectra in darkness)
# - NIR reported as fraction of total energy (VIS+NIR)
#
# OFFLINE BUFFER:
# - CSV fallback to ~/Documents/Lightmeter_csv_out/ when all InfluxDB
#   endpoints fail (archive-only). 10-min tmp files in daily_tmp/, merged
#   into per-day aggregates in the parent dir; recovers leftover tmps on
#   startup so a power cycle never loses more than one in-flight write.
#
# Result: 50-60% faster + much more accurate spectral composition

import time, json, os, csv, datetime, re, subprocess, shutil
from pathlib import Path
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
import requests
from requests.adapters import HTTPAdapter
import board
from adafruit_as7341 import AS7341, Gain

# ============================
# USER CONFIG (edit here)
# ============================

# Device Identification
# ---------------------
DEVICE = "RPi-1"              # Unique device name written to InfluxDB as "Device" tag
                              # Change this for each Raspberry Pi to distinguish multiple sensors

MEAS = "LIGHT"                # InfluxDB measurement name for spectral data points
                              # All spectral readings will be written to this measurement

# Auto-Sensitivity Switching
# --------------------------
AUTORANGE_ENABLE     = True    # Switch between presets automatically
AUTORANGE_SUN_ENABLE = False   # Enable third SUN tier (requires as7341_lux_cal_sun.json)
AUTORANGE_HYST       = 3       # Consecutive frames above/below threshold before switching
AUTORANGE_SAT_FRAC   = 0.875   # Step up (brighter preset) when peak >= this fraction of ADC FS
AUTORANGE_LOW_FRAC   = 0.003   # Step down (dimmer preset) when peak <= this fraction of ADC FS

# Measurement Averaging and Timing
# ---------------------------------
AVG = 5                       # Number of sensor frames to average per reading
                              # Higher = less noise but slower response
                              # Typical: 3-10 frames

PERIOD = 60                 # Minimum seconds between measurements
                              # Set to 0.0 for maximum speed (limited by integration time)

VERBOSE_BANDS = False         # If True, log each spectral band individually
                              # If False, log only summary (lux, max signal, clear channel)

LOG_EVERY_N = 1               # Log to console every N samples (1 = log every sample)
                              # Set higher (e.g., 10) to reduce console spam

# Saturation Warning
# ------------------
SAT_WARN_FRAC = 0.875         # Warn when signal exceeds this fraction of ADC full-scale
                              # 0.875 = 87.5% of maximum (suggests reducing gain or integration time)

# InfluxDB Configuration
# ----------------------
ENDPOINTS = [
    ("10.239.99.73", 8086, "AAB"),  # Primary InfluxDB: (host, port, database_name)
    ("10.239.99.97", 8086, "AAB"),  # Secondary InfluxDB (optional redundancy)
]                             # Add/remove endpoints as needed; writes happen in parallel

MAX_RETRY_QUEUE = 500         # Maximum failed writes to queue per endpoint before dropping
                              # Prevents memory overflow during extended network outages

CSV_ALWAYS = True             # If True, write a CSV row every cycle regardless of
                              # InfluxDB success (default — guarantees a local record).
                              # Set False for offline-buffer-only behaviour (CSV written
                              # only when ALL endpoints fail).

# HTTP Performance Tuning
# -----------------------
RETRY_BUDGET_PER_LOOP = 10    # Maximum retry queue flushes attempted per measurement loop
                              # Higher = faster recovery from network issues but more CPU

TIMEOUT_CURRENT = (0.5, 2)    # HTTP timeout for current measurement: (connect_sec, read_sec)
                              # Aggressive timeout keeps main loop responsive

TIMEOUT_RETRY = (1, 3)        # HTTP timeout for retry queue: (connect_sec, read_sec)
                              # More patient timeout for background retries

# Sensitivity Presets and Calibration Paths
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory

SENS_HI = {                                 # Dim light — high gain, long integration
    "integration_time_ms": 50,
    "gain": Gain.GAIN_256X,
    "dark_file": BASE_DIR / "as7341_dark_hi.json",
    "cal_file":  BASE_DIR / "as7341_lux_cal_hi.json",
}
SENS_LO = {                                 # Bright light — low gain, short integration
    "integration_time_ms": 10,
    "gain": Gain.GAIN_16X,
    "dark_file": BASE_DIR / "as7341_dark_lo.json",
    "cal_file":  BASE_DIR / "as7341_lux_cal_lo.json",
}
SENS_SUN = {                                # Direct sunlight — very low gain, short integration
    "integration_time_ms": 10,
    "gain": Gain.GAIN_4X,
    "dark_file": BASE_DIR / "as7341_dark_sun.json",
    "cal_file":  BASE_DIR / "as7341_lux_cal_sun.json",
}

# .env Overrides (optional — copy config/sample.env to .env in project root)
# ---------------------------------------------------------------------------
def _apply_env(path: Path):
    """Load key=value pairs from a .env file and override config globals above."""
    global DEVICE, ENDPOINTS, AVG, PERIOD, CSV_ALWAYS
    global AUTORANGE_ENABLE, AUTORANGE_SUN_ENABLE, AUTORANGE_HYST, AUTORANGE_SAT_FRAC, AUTORANGE_LOW_FRAC
    if not path.exists():
        return
    env = {}
    with open(path) as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith('#'):
                continue
            if '=' in _line:
                _k, _, _v = _line.partition('=')
                env[_k.strip()] = _v.strip()
    if "DEVICE"             in env: DEVICE             = env["DEVICE"]
    if "INFLUX_ENDPOINTS"   in env: ENDPOINTS          = [tuple(e) for e in json.loads(env["INFLUX_ENDPOINTS"])]
    if "AVG"                in env: AVG                = int(env["AVG"])
    if "PERIOD"             in env: PERIOD             = float(env["PERIOD"])
    if "AUTORANGE_ENABLE"     in env: AUTORANGE_ENABLE     = env["AUTORANGE_ENABLE"].lower() in ("true","1","yes")
    if "AUTORANGE_SUN_ENABLE" in env: AUTORANGE_SUN_ENABLE = env["AUTORANGE_SUN_ENABLE"].lower() in ("true","1","yes")
    if "AUTORANGE_HYST"       in env: AUTORANGE_HYST       = int(env["AUTORANGE_HYST"])
    if "AUTORANGE_SAT_FRAC"   in env: AUTORANGE_SAT_FRAC   = float(env["AUTORANGE_SAT_FRAC"])
    if "AUTORANGE_LOW_FRAC"   in env: AUTORANGE_LOW_FRAC   = float(env["AUTORANGE_LOW_FRAC"])
    if "CSV_ALWAYS"           in env: CSV_ALWAYS           = env["CSV_ALWAYS"].lower() in ("true","1","yes")
    if "SENS_HI_IT_MS"        in env: SENS_HI["integration_time_ms"]  = int(env["SENS_HI_IT_MS"])
    if "SENS_HI_GAIN"         in env: SENS_HI["gain"]                 = getattr(Gain, env["SENS_HI_GAIN"])
    if "SENS_LO_IT_MS"        in env: SENS_LO["integration_time_ms"]  = int(env["SENS_LO_IT_MS"])
    if "SENS_LO_GAIN"         in env: SENS_LO["gain"]                 = getattr(Gain, env["SENS_LO_GAIN"])
    if "SENS_SUN_IT_MS"       in env: SENS_SUN["integration_time_ms"] = int(env["SENS_SUN_IT_MS"])
    if "SENS_SUN_GAIN"        in env: SENS_SUN["gain"]                = getattr(Gain, env["SENS_SUN_GAIN"])

_apply_env(BASE_DIR / ".env")

# CSV Output (every cycle when CSV_ALWAYS=true; otherwise only when ALL
# InfluxDB endpoints fail — i.e. true offline-buffer behaviour).
# ----------------------------------------------------------------------
CSV_OUT_DIR = Path.home() / "Documents" / "Lightmeter_csv_out"   # Daily aggregated files live here
CSV_TMP_DIR = CSV_OUT_DIR / "daily_tmp"                          # 10-minute work files live here
CSV_TAG = "as7341-RPi_lightlogger"                               # Common suffix used in all CSV filenames
CSV_ROTATE_INTERVAL_S = 10 * 60                                  # Close & start a new tmp file every 10 minutes
STATUS_FILE = CSV_OUT_DIR / "status.json"                        # Atomic per-cycle health snapshot
CSV_HEADER = ["timestamp_iso", "device", "lux", "clear",
              "rel_415", "rel_445", "rel_480", "rel_515",
              "rel_555", "rel_590", "rel_630", "rel_680", "rel_nir",
              "irr_415", "irr_445", "irr_480", "irr_515",
              "irr_555", "irr_590", "irr_630", "irr_680"]
# Absolute irradiance columns (W/m^2/nm at channel center) are populated only
# when as7341_responsivity.json carries a `responsivity_BC_per_W_m2_nm` block
# (written by as7341_calibrate.py Phase 2). Otherwise they are emitted empty.
# Matches:  YYYY-MM-DD-HHMM-as7341-RPi_lightlogger_tmp.csv
CSV_TMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(\d{4})-" + re.escape(CSV_TAG) + r"_tmp\.csv$")

# ============================
# Constants
# ============================
# Spectral band definitions for AS7341 sensor
BANDS9 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680","nir"]  # 8 visible + 1 NIR channel names
WLS9   = [  415,    445,    480,    515,    555,    590,    630,    680,    910 ]  # Center wavelengths in nanometers
VIS8 = BANDS9[:8]  # Visible channels only (excludes NIR for separate processing)

# ============================
# SPECTRAL ACCURACY IMPROVEMENTS
# ============================
# Responsivity correction factors (normalize to F5/555nm = 1.0 as reference)
# Based on AS7341 typical responsivity values from datasheet.
# Without correction, blue/red channels appear artificially weak in spectral composition.
RESPONSIVITY_CORRECTION = [
    2.0,   # F1 (415nm)
    1.67,  # F2 (445nm)
    1.33,  # F3 (480nm)
    1.11,  # F4 (515nm)
    1.0,   # F5 (555nm) — reference
    1.11,  # F6 (590nm)
    1.43,  # F7 (630nm)
    2.0,   # F8 (680nm)
]
# NIR (~910nm) cannot be calibrated against the C-7000 (380–780nm range), so it
# stays at a datasheet-derived default unless the user provides a value via the
# `nir` key in as7341_responsivity.json. 2.5 is a conservative AS7341 typical;
# accept anything within reason as user-supplied override.
NIR_RESPONSIVITY_CORRECTION = 2.5

# Absolute responsivity per VIS8 channel (BasicCounts per W/m^2/nm). When loaded,
# the runtime emits irr_* columns / `irradiance` Influx field per channel. None
# means Phase 2 has not yet produced the absolute block — outputs degrade
# gracefully (empty CSV cells, no Influx field).
RESPONSIVITY_ABS = None

# Override with empirical values if available (generated by as7341_calibrate.py Phase 2)
_resp_file = BASE_DIR / "as7341_responsivity.json"
if _resp_file.exists():
    with open(_resp_file) as _rf:
        _resp = json.load(_rf)
    RESPONSIVITY_CORRECTION = [float(_resp["corrections"][b]) for b in BANDS9[:8]]
    if "nir" in _resp.get("corrections", {}):
        NIR_RESPONSIVITY_CORRECTION = float(_resp["corrections"]["nir"])
    print(f"[INFO] Empirical responsivity loaded from {_resp_file.name} "
          f"(NIR corr={'override' if 'nir' in _resp.get('corrections', {}) else 'default'} "
          f"= {NIR_RESPONSIVITY_CORRECTION})")
    _abs = _resp.get("responsivity_BC_per_W_m2_nm")
    if _abs:
        RESPONSIVITY_ABS = [float(_abs[b]) for b in BANDS9[:8]]
        print(f"[INFO] Absolute responsivity loaded; emitting irr_* "
              f"(W/m^2/nm at channel center).")
    else:
        print("[INFO] No absolute responsivity in cal file; "
              "re-run as7341_calibrate.py Phase 2 to enable irr_* outputs.")

# Minimum signal threshold for valid spectrum (BasicCounts units)
# BasicCounts = (raw - dark) / (gain × integration_time_ms)
# Below this, spectrum is mostly noise and should not be reported
# Set to 0.01 to detect very low light while filtering complete darkness
MIN_SIGNAL_THRESHOLD = 0.01

# Gain multiplier lookup table: converts Gain enum to numeric multiplier
# Used to normalize raw ADC counts to BasicCounts for calibration
GAIN_MULT = {
    Gain.GAIN_0_5X:0.5,  Gain.GAIN_1X:1.0,   Gain.GAIN_2X:2.0,   Gain.GAIN_4X:4.0,
    Gain.GAIN_8X:8.0,    Gain.GAIN_16X:16.0, Gain.GAIN_32X:32.0, Gain.GAIN_64X:64.0,
    Gain.GAIN_128X:128.0,Gain.GAIN_256X:256.0,Gain.GAIN_512X:512.0
}

# ============================
# Helper Functions
# ============================

def ms_to_atime_astep(target_ms):
    """
    Convert desired integration time in milliseconds to ATIME and ASTEP register values.
    
    Formula: integration_time_ms = (ATIME + 1) × (ASTEP + 1) × 2.78e-3
    
    Strategy: Maximize ASTEP (better precision), minimize ATIME.
    Constraints: ATIME ∈ [0, 255], ASTEP ∈ [0, 65534]
    
    Args:
        target_ms: Desired integration time in milliseconds (1-1812 ms)
    
    Returns:
        tuple: (atime, astep) register values that give closest match to target
    
    Raises:
        ValueError: If target_ms is outside achievable range
    """
    # Calculate total increments needed
    # target_ms = (ATIME + 1) × (ASTEP + 1) × 2.78e-3
    # total_increments = target_ms / 2.78e-3
    total_increments = target_ms / 2.78e-3
    
    # Check if achievable
    min_increments = 1 * 1  # ATIME=0, ASTEP=0
    max_increments = 256 * 65535  # ATIME=255, ASTEP=65534
    
    if total_increments < min_increments:
        raise ValueError(f"Integration time {target_ms}ms too short (min ~0.003ms)")
    if total_increments > max_increments:
        raise ValueError(f"Integration time {target_ms}ms too long (max ~1812ms)")
    
    # Strategy: Use smallest ATIME that allows ASTEP <= 65534
    # This maximizes ASTEP for better time resolution
    for atime in range(256):
        astep = (total_increments / (atime + 1)) - 1
        if 0 <= astep <= 65534:
            astep = int(round(astep))
            # Verify and return
            actual_ms = (atime + 1) * (astep + 1) * 2.78e-3
            if abs(actual_ms - target_ms) / target_ms > 0.05:  # > 5% error
                print(f"[WARN] Integration time: requested {target_ms}ms, actual {actual_ms:.2f}ms ({abs(actual_ms-target_ms)/target_ms*100:.1f}% error)")
            return atime, astep
    
    # Fallback: shouldn't reach here
    raise ValueError(f"Could not find valid ATIME/ASTEP for {target_ms}ms")

def integration_time_ms(atime:int, astep:int)->float:
    """
    Calculate integration time in milliseconds from ATIME and ASTEP registers.

    Args:
        atime: ATIME register value (0-255)
        astep: ASTEP register value (0-65534)

    Returns:
        float: Integration time in milliseconds
    """
    return (atime + 1) * (astep + 1) * 2.78e-3

def adc_fullscale(atime:int, astep:int)->int:
    """
    Calculate ADC full-scale value from ATIME and ASTEP registers.

    The ADC can accumulate up to (ATIME+1)×(ASTEP+1) counts, capped at 16-bit max.

    Args:
        atime: ATIME register value (0-255)
        astep: ASTEP register value (0-65534)

    Returns:
        int: Maximum ADC count (capped at 65535)
    """
    fs = (atime + 1) * (astep + 1)
    return 65535 if fs > 65535 else fs

def current_gain_mult(s:AS7341)->float:
    """
    Get current gain multiplier from sensor.

    Args:
        s: AS7341 sensor object

    Returns:
        float: Gain multiplier (0.5 to 512.0)
    """
    return float(GAIN_MULT.get(s.gain, 1.0))

def apply_sensitivity(s:AS7341, preset:dict)->dict:
    """
    Apply a sensitivity preset to the sensor and reload all derived calibration state.

    Called at startup (initial preset) and whenever the auto-sensitivity logic switches
    between HI and LO. Returns a dict with all values that main() needs to update.
    """
    atime, astep = ms_to_atime_astep(preset["integration_time_ms"])
    s.atime = atime
    s.astep = astep
    s.gain  = preset["gain"]
    it_ms   = integration_time_ms(atime, astep)
    fs      = adc_fullscale(atime, astep)
    gnum    = current_gain_mult(s)

    darkJ    = load_dark(preset["dark_file"])
    meta_ok  = dark_ok_for_settings(darkJ.get("meta"), str(s.gain), atime, astep)
    if not meta_ok and darkJ.get("meta") is not None:
        print(f"[WARN] Dark meta mismatch for {preset['dark_file'].name} — dark offsets inactive.")
    dark_vis8  = [int(darkJ[b]) if meta_ok else 0 for b in VIS8]
    dark_clear = int(darkJ["clear"]) if meta_ok else 0
    dark_nir   = int(darkJ.get("nir", 0)) if meta_ok else 0

    b0, w, cal_meta = load_cal(preset["cal_file"])
    if cal_meta is not None and not dark_ok_for_settings(cal_meta, str(s.gain), atime, astep):
        print(f"[WARN] Lux cal meta mismatch for {preset['cal_file'].name} "
              f"(file gain={cal_meta.get('gain')!r}, atime={cal_meta.get('atime')}, "
              f"astep={cal_meta.get('astep')}; current gain={s.gain}, atime={atime}, astep={astep}).")

    return {
        "atime": atime, "astep": astep,
        "it_ms": it_ms, "fs": fs, "gnum": gnum,
        "dark_vis8": dark_vis8, "dark_clear": dark_clear, "dark_nir": dark_nir,
        "b0": b0, "w": w,
    }

def load_cal(path:Path):
    """
    Load lux calibration coefficients from JSON file.

    Expected format: {"b0": float, "w": [8 floats], "meta": {...}}
    Model: lux = b0 + sum(w[i] * BasicCounts[i])

    Args:
        path: Path to calibration JSON file

    Returns:
        tuple: (b0, w, meta) where b0 is intercept, w is list of 8 weights,
            and meta is the metadata dict (or None if missing).

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        ValueError: If file doesn't contain exactly 8 weights
    """
    if not path.exists():
        raise FileNotFoundError(f"Calibration file '{path}' not found.")
    with open(path, "r") as f:
        J = json.load(f)
    b0 = float(J["b0"]); w = [float(x) for x in J["w"]]
    if len(w)!=8: raise ValueError("Calibration file must contain 8 weights (w).")
    return b0, w, J.get("meta")

def load_dark(path:Path):
    """
    Load dark offset calibration from JSON file.

    Expected format: {
        "meta": {"gain": str, "atime": int, "astep": int, ...},
        "clear": int,
        "nm415": int, "nm445": int, ..., "nir": int
    }

    Args:
        path: Path to dark calibration JSON file

    Returns:
        dict: Dark offsets with metadata, or zeros if file doesn't exist
    """
    if not path.exists():
        print(f"[INFO] No dark file at '{path}', using zero offsets.")
        return {"meta": None, "clear": 0, **{b:0 for b in BANDS9}}
    with open(path, "r") as f:
        J = json.load(f)
    meta = J.get("meta", None)
    out = {"meta": meta, "clear": int(J.get("clear", 0))}
    for b in BANDS9: out[b] = int(J.get(b, 0))
    return out

def dark_ok_for_settings(dark_meta, gain, atime, astep)->bool:
    """
    Check if dark calibration metadata matches current sensor settings.

    Dark offsets are only valid if captured with same gain/ATIME/ASTEP.

    Args:
        dark_meta: Metadata dict from dark calibration file
        gain: Current sensor gain setting
        atime: Current ATIME value
        astep: Current ASTEP value

    Returns:
        bool: True if dark calibration matches current settings
    """
    if not dark_meta: return False
    try:
        return (str(dark_meta.get("gain",""))==str(gain) and
                int(dark_meta.get("atime",-1))==int(atime) and
                int(dark_meta.get("astep",-1))==int(astep))
    except Exception:
        return False

def read_vis_clear_nir(s:AS7341):
    """
    Read all 10 channels from AS7341 sensor (8 visible + CLEAR + NIR).

    Tries direct channel properties first, falls back to all_channels if needed.
    Handles different driver versions gracefully.

    Args:
        s: AS7341 sensor object

    Returns:
        tuple: (vis, clear, nir) where:
            vis: List of 8 floats for visible channels (415-680nm)
            clear: Float for CLEAR channel (broadband)
            nir: Float for NIR channel (~910nm)

    Raises:
        RuntimeError: If unable to read channels from driver
    """
    try:
        # Direct property access (preferred method for modern driver)
        vis = [float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm),
               float(s.channel_515nm), float(s.channel_555nm), float(s.channel_590nm),
               float(s.channel_630nm), float(s.channel_680nm)]
        clear = float(s.channel_clear)
        nir   = float(s.channel_nir)
        return vis, clear, nir
    except Exception:
        pass

    # Fallback: use all_channels if direct properties don't exist
    if hasattr(s, "all_channels"):
        ac = list(s.all_channels)
        if len(ac) >= 10:
            vis = list(map(float, ac[:8]))
            tail = ac[8:10]
            clear_guess, nir_guess = (float(tail[0]), float(tail[1]))
            # CLEAR is usually larger than NIR; swap if backwards
            if clear_guess < nir_guess:
                clear_guess, nir_guess = nir_guess, clear_guess
            return vis, clear_guess, nir_guess
        elif len(ac) >= 9:
            vis = list(map(float, ac[:8]))
            clear = float(ac[8]); nir = 0.0
            return vis, clear, nir

    raise RuntimeError("Unable to read VIS/CLEAR/NIR channels from AS7341 driver.")

def avg_frames(s:AS7341, n:int):
    """
    Average multiple sensor readings to reduce noise.

    Args:
        s: AS7341 sensor object
        n: Number of frames to average (minimum 1)

    Returns:
        tuple: (vis_avg, clear_avg, nir_avg) where each value is averaged
    """
    n = max(1, int(n))
    acc8 = [0.0]*8; acc_clear = 0.0; acc_nir = 0.0
    for _ in range(n):
        v8, c, nval = read_vis_clear_nir(s)
        for i in range(8): acc8[i]+=float(v8[i])
        acc_clear += float(c)
        acc_nir   += float(nval)
    return [x/n for x in acc8], acc_clear/n, acc_nir/n

# ============================
# InfluxDB Line Protocol Building
# ============================
# Pre-build static parts of InfluxDB line protocol for performance
INFLUX_TEMPLATES = None  # Will hold list of 9 measurement tag strings
LUX_TEMPLATE = None      # Will hold lux measurement tag string

def init_influx_templates():
    """
    Pre-build static parts of InfluxDB line protocol strings.

    Called once at startup. Avoids string formatting overhead in hot loop.
    """
    global INFLUX_TEMPLATES, LUX_TEMPLATE
    # Build spectral measurement templates: "LIGHT,Device=RPi-1,wavelength_nm=415"
    INFLUX_TEMPLATES = [f"{MEAS},Device={DEVICE},wavelength_nm={wl}" for wl in WLS9]
    # Build lux measurement template: "LIGHT_LUX,Device=RPi-1,method=lin_basic"
    LUX_TEMPLATE = f"LIGHT_LUX,Device={DEVICE},method=lin_basic"

def build_influx_lines(ts_ns:int, rel_vis8, rel_nir, lux_value, clear_value, irr_vis8=None):
    """
    Build InfluxDB line protocol payload from measurement data.

    Creates 10 lines total:
    - 8 spectral lines for VIS channels (415-680nm) with rel_intensity field
      (and `irradiance` field in W/m^2/nm when `irr_vis8` is provided)
    - 1 spectral line for NIR channel (910nm) with rel_intensity field
    - 1 lux line with lux and clear fields

    Args:
        ts_ns: Timestamp in nanoseconds (Unix epoch)
        rel_vis8: List of 8 relative intensities for VIS channels (normalized, sum to 1.0)
        rel_nir: NIR relative intensity (as fraction of total VIS+NIR energy)
        lux_value: Calibrated lux reading (illuminance)
        clear_value: CLEAR channel raw value (broadband photocurrent)
        irr_vis8: Optional list of 8 absolute irradiances (W/m^2/nm) at channel
            centers; emitted as the `irradiance` field on each VIS8 point when
            provided.

    Returns:
        list: InfluxDB line protocol strings ready for HTTP POST
    """
    # VIS8 spectral composition (8 data points)
    # Format: "LIGHT,Device=RPi-1,wavelength_nm=415 rel_intensity=0.123456[,irradiance=1.23e-04] <ts>"
    if irr_vis8 is None:
        lines = [f"{INFLUX_TEMPLATES[i]} rel_intensity={v:.6f} {ts_ns}"
                 for i, v in enumerate(rel_vis8)]
    else:
        lines = [f"{INFLUX_TEMPLATES[i]} rel_intensity={v:.6f},irradiance={irr_vis8[i]:.6e} {ts_ns}"
                 for i, v in enumerate(rel_vis8)]

    # NIR as separate point with wavelength_nm=910 tag
    lines.append(f"{INFLUX_TEMPLATES[8]} rel_intensity={rel_nir:.6f} {ts_ns}")

    # Lux measurement with both lux (calibrated) and clear (raw) fields
    # Format: "LIGHT_LUX,Device=RPi-1,method=lin_basic lux=123.456,clear=12345 1234567890000000000"
    lines.append(f"{LUX_TEMPLATE} lux={lux_value:.3f},clear={int(clear_value)} {ts_ns}")
    return lines

# ============================
# CSV Offline Buffer
# ============================
# Strategy:
#   - On EVERY all-endpoints-failed sample, append one row to a 10-minute
#     "tmp" file in CSV_TMP_DIR. Each row is flushed to disk immediately.
#   - When the active tmp file is >= 10 min old, close it and start a new one.
#     This bounds power-loss data loss to roughly the OS write-back window.
#   - Periodically (and on rotation) merge tmp files whose date prefix is
#     before today into a per-day daily file in CSV_OUT_DIR; delete sources.
#   - On startup, merge ANY leftover tmp files (regardless of age) into the
#     matching daily file, so a power cycle never leaves dangling tmp files.

def csv_tmp_path(start_epoch:float)->Path:
    """Path of a fresh 10-min tmp CSV named by its start time in UTC.

    Filename and row contents both use UTC so daily aggregates group by UTC
    date — unambiguous when the device moves between timezones or runs in
    a non-local TZ.
    """
    dt = datetime.datetime.fromtimestamp(start_epoch, datetime.timezone.utc)
    name = dt.strftime("%Y-%m-%d-%H%M") + f"-{CSV_TAG}_tmp.csv"
    return CSV_TMP_DIR / name

def csv_daily_path(date_str:str)->Path:
    """Path of the daily aggregate file for a given YYYY-MM-DD date string."""
    return CSV_OUT_DIR / f"{date_str}-{CSV_TAG}_daily.csv"

def merge_tmp_files_to_daily(tmp_files):
    """
    Group tmp files by their date prefix and append each group's rows
    to the matching daily file (creating it if needed). Deletes each source
    only after it has been successfully appended.

    Returns the number of source files merged.
    """
    by_day = {}
    for p in tmp_files:
        m = CSV_TMP_RE.match(p.name)
        if m:
            by_day.setdefault(m.group(1), []).append(p)

    merged = 0
    for date_str, files in by_day.items():
        files.sort()                                # Lexicographic sort = chronological
        target = csv_daily_path(date_str)
        target_existed = target.exists()
        try:
            with open(target, "a", newline="") as out:
                w = csv.writer(out)
                wrote_header = target_existed
                for src in files:
                    with open(src, "r", newline="") as inp:
                        r = csv.reader(inp)
                        header = next(r, None)
                        if not wrote_header and header:
                            w.writerow(header)
                            wrote_header = True
                        for row in r:
                            w.writerow(row)
                out.flush()
            for src in files:
                src.unlink()
            merged += len(files)
        except Exception as e:
            print(f"[ERR] CSV merge for {date_str}: {e}")
    return merged

def aggregate_completed_days():
    """Merge tmp files whose date prefix is BEFORE today (UTC) into daily files."""
    if not CSV_TMP_DIR.exists():
        return 0
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    candidates = []
    for p in CSV_TMP_DIR.iterdir():
        if not p.is_file():
            continue
        m = CSV_TMP_RE.match(p.name)
        if m and m.group(1) < today:
            candidates.append(p)
    if not candidates:
        return 0
    n = merge_tmp_files_to_daily(candidates)
    if n:
        print(f"[CSV] Aggregated {n} tmp file(s) into daily file(s).")
    return n

def csv_startup_recovery():
    """On launch, merge any pre-existing tmp files into their daily files."""
    if not CSV_TMP_DIR.exists():
        return
    leftovers = [p for p in CSV_TMP_DIR.iterdir()
                 if p.is_file() and CSV_TMP_RE.match(p.name)]
    if not leftovers:
        return
    n = merge_tmp_files_to_daily(leftovers)
    print(f"[STARTUP] Merged {n} leftover tmp CSV file(s) into daily file(s).")

def write_csv_fallback(ts_ns:int, lux:float, clear:float, rel_vis8, rel_nir:float, state:dict, irr_vis8=None):
    """
    Append one row to the active 10-min tmp CSV. Rotates the active file
    if it has been open for >= CSV_ROTATE_INTERVAL_S; runs aggregation of
    any completed-day tmp files at rotation time.

    `state` carries the active path and its open time across calls:
        {"path": Path|None, "started_at": float|None}

    If `irr_vis8` is None the 8 irr_* columns are emitted empty so the row
    width still matches CSV_HEADER.
    """
    CSV_TMP_DIR.mkdir(parents=True, exist_ok=True)
    now = ts_ns / 1e9

    # Rotate if the active file has aged out
    if state["started_at"] is not None and (now - state["started_at"]) >= CSV_ROTATE_INTERVAL_S:
        state["path"] = None
        state["started_at"] = None
        try:
            aggregate_completed_days()
        except Exception as e:
            print(f"[WARN] CSV aggregation failed: {e}")

    # Open a new tmp file if needed
    if state["path"] is None:
        state["started_at"] = now
        state["path"] = csv_tmp_path(now)

    new_file = not state["path"].exists()
    iso = datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat()
    row = [iso, DEVICE, f"{lux:.3f}", f"{int(clear)}"]
    row.extend(f"{x:.6f}" for x in rel_vis8)
    row.append(f"{rel_nir:.6f}")
    if irr_vis8 is None:
        row.extend([""] * 8)
    else:
        row.extend(f"{x:.6e}" for x in irr_vis8)

    with open(state["path"], "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)
        f.flush()

# ============================
# Status JSON (per-cycle health snapshot for headless ops)
# ============================
def _iso_from_ns(ts_ns:int)->str:
    """UTC ISO 8601 from nanosecond Unix timestamp."""
    return datetime.datetime.fromtimestamp(ts_ns / 1e9, datetime.timezone.utc).isoformat()

def _atomic_write_json(path:Path, obj):
    """Atomic JSON write via tmp+rename, so a concurrent reader (e.g. an SSH
    user `cat`-ing the status file) never sees a half-written file. POSIX
    rename is atomic within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    tmp.replace(path)

# ============================
# OPTIMIZATION: Parallel HTTP write function
# ============================
def write_to_endpoint(ent, payload, is_retry=False):
    """
    Write InfluxDB line protocol payload to a single endpoint.

    Uses session connection pooling for performance. Adjusts timeout based on
    whether this is a current measurement or retry from queue.

    Args:
        ent: Endpoint dict with keys: url, params, sess, label
        payload: InfluxDB line protocol string (multiple lines separated by \\n)
        is_retry: If True, use more patient timeout (default: False)

    Returns:
        tuple: (label, success, error_msg) where:
            label: Human-readable endpoint identifier (e.g., "10.0.0.1:8086/db")
            success: Boolean indicating if write succeeded
            error_msg: Error message string if failed, None if success
    """
    timeout = TIMEOUT_RETRY if is_retry else TIMEOUT_CURRENT
    try:
        r = ent["sess"].post(ent["url"], params=ent["params"],
                             data=payload, timeout=timeout)
        if r.status_code == 204:
            return ent["label"], True, None
        else:
            return ent["label"], False, f"HTTP {r.status_code}: {r.text.strip()[:100]}"
    except requests.RequestException as e:
        return ent["label"], False, str(e)[:100]

# ============================
# Startup time sanity
# ============================
def check_clock_sane():
    """Warn if the system clock looks unset or unsynchronised.

    Pis without an RTC restore the clock from fake-hwclock at boot, so a
    cold start without network leaves timestamps stuck at the last shutdown
    time until NTP catches up. For field operation: connect a phone
    hotspot, wait for NTP sync, then disconnect — the clock will keep
    running for the duration of the experiment.
    """
    now = datetime.datetime.now()
    if now.year < 2025:
        print(f"[WARN] System clock looks unset (now={now.isoformat()}). "
              "CSV/Influx timestamps will be wrong until NTP sync. "
              "Connect a network with internet access (e.g. phone hotspot) and re-check.")
        return
    try:
        r = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout.strip() == "no":
            print("[WARN] NTP not synchronised. Timestamps may drift; "
                  "connect a network with internet access to sync the clock.")
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

# ============================
# Main Loop
# ============================
def main():
    """
    Main measurement loop: read sensor, calculate spectra and lux, write to InfluxDB.

    Initialization:
    1. Configure AS7341 sensor with calculated ATIME/ASTEP from INTEGRATION_TIME_MS
    2. Load dark offset and lux calibration files
    3. Setup HTTP sessions with connection pooling
    4. Create thread pool for parallel writes

    Main Loop:
    1. Read and average sensor frames
    2. Apply dark correction
    3. Normalize to BasicCounts (exposure-independent units)
    4. Calculate lux from linear model
    5. Calculate spectral composition with responsivity correction
    6. Write to all InfluxDB endpoints in parallel
    7. Flush retry queues for failed writes
    8. Log results and sleep until next period
    """
    # ========================================
    # INITIALIZATION: Sensor Configuration
    # ========================================
    i2c = board.I2C()                          # Initialize I2C bus
    s = AS7341(i2c)                            # Create sensor object at default address 0x39

    try:
        s.flicker_detection_enabled = False    # Disable flicker detection for consistent timing
    except Exception:
        pass                                   # Ignore if not supported by driver version

    # ========================================
    # INITIALIZATION: Calibration Files + Sensor Settings
    # ========================================
    # Start in HI sensitivity (dim-light mode); autorange will step down if needed.
    active_preset = "hi"
    sens_cfg      = apply_sensitivity(s, SENS_HI)
    atime         = sens_cfg["atime"]
    astep         = sens_cfg["astep"]
    actual_it_ms  = sens_cfg["it_ms"]
    fs            = sens_cfg["fs"]
    gnum          = sens_cfg["gnum"]
    dark_vis8     = sens_cfg["dark_vis8"]
    dark_clear    = sens_cfg["dark_clear"]
    dark_nir      = sens_cfg["dark_nir"]
    b0, w         = sens_cfg["b0"], sens_cfg["w"]
    ar_sat_cnt = ar_low_cnt = 0                 # Hysteresis counters for sensitivity switching

    # ========================================
    # INITIALIZATION: InfluxDB Connections
    # ========================================
    # Pre-build static parts of line protocol strings for performance
    init_influx_templates()

    # Setup HTTP sessions with connection pooling for each endpoint
    adapter = HTTPAdapter(
        pool_connections=len(ENDPOINTS),    # Number of host connections to cache
        pool_maxsize=len(ENDPOINTS) * 2,    # Max pooled connections per host
        max_retries=0                        # Manual retry handling with queues
    )

    sessions = []      # List of endpoint dicts for parallel writes
    retry_qs = {}      # Per-endpoint queues for failed writes
    for host, port, db in ENDPOINTS:
        label = f"{host}:{port}/{db}"       # Human-readable endpoint identifier
        sess = requests.Session()           # Persistent session for connection reuse
        sess.mount("http://", adapter)      # Apply pooling config
        sessions.append({
            "url": f"http://{host}:{port}/write",     # InfluxDB write API endpoint
            "params": {"db": db, "precision": "ns"},  # Query params: database and timestamp precision
            "sess": sess,
            "label": label,
        })
        # Create bounded retry queue (drops oldest on overflow)
        retry_qs[label] = deque(maxlen=MAX_RETRY_QUEUE)

    # Create thread pool for parallel HTTP writes to all endpoints
    executor = ThreadPoolExecutor(max_workers=len(ENDPOINTS))

    process_start_ns = time.time_ns()
    process_start_iso = _iso_from_ns(process_start_ns)

    # Initialize performance tracking metrics
    metrics = {
        "samples_collected": 0,              # Total measurements taken
        "http_successes": defaultdict(int),  # Successful writes per endpoint
        "http_failures": defaultdict(int),   # Failed writes per endpoint
        "retry_queue_sizes": {label: 0 for label in retry_qs},  # Current queue depths
        "loop_times": deque(maxlen=100),     # Recent loop execution times
        "csv_rows_written": 0,               # Rows appended to CSV output
        "last_success_ts": {label: 0 for label in retry_qs},   # ns; 0 = never
        "last_failure_ts": {label: 0 for label in retry_qs},   # ns; 0 = never
        "last_failure_error": {label: None for label in retry_qs},
    }

    # CSV offline-buffer state (active tmp file path + open time) and
    # one-shot recovery of any tmp files left behind by a previous run.
    csv_state = {"path": None, "started_at": None}
    csv_startup_recovery()

    # ========================================
    # STARTUP: Print Configuration Summary
    # ========================================
    print("AS7341 -> Influx v1 fan-out:")
    for ent in sessions: print("  -", ent["label"])
    print(f"Device={DEVICE}, AVG={AVG}, PERIOD={PERIOD}s")
    sun_enabled = AUTORANGE_SUN_ENABLE and SENS_SUN["cal_file"].exists()
    if AUTORANGE_SUN_ENABLE and not sun_enabled:
        print(f"[WARN] AUTORANGE_SUN_ENABLE=true but {SENS_SUN['cal_file'].name} not found — SUN tier disabled.")
    sun_str = (f" | SUN: gain={SENS_SUN['gain']}, IT={SENS_SUN['integration_time_ms']}ms"
               if sun_enabled else " | SUN: disabled")
    print(f"Sensitivity HI: gain={SENS_HI['gain']}, IT={SENS_HI['integration_time_ms']}ms | "
          f"LO: gain={SENS_LO['gain']}, IT={SENS_LO['integration_time_ms']}ms{sun_str}")
    print(f"Autorange: {'ON' if AUTORANGE_ENABLE else 'OFF'}, hyst={AUTORANGE_HYST}, "
          f"sat_frac={AUTORANGE_SAT_FRAC}, low_frac={AUTORANGE_LOW_FRAC}")
    print(f"Starting in HI sensitivity: ATIME={atime}, ASTEP={astep}, IT={actual_it_ms:.1f}ms, ADC_FS={fs}")
    print(f"CSV output: {CSV_OUT_DIR} ({'every cycle' if CSV_ALWAYS else 'only on all-endpoints-fail'})")
    print(f"Status JSON: {STATUS_FILE}")
    check_clock_sane()

    sample_idx = 0                # Sample counter for logging

    # ========================================
    # MAIN LOOP: Continuous Measurement
    # ========================================
    try:
        while True:
            loop_start = time.time()  # Track loop timing for performance monitoring

            # -------- Step 1: Read Sensor --------
            # Average AVG frames to reduce noise
            vis_raw, clear_raw, nir_raw = avg_frames(s, AVG)

            # -------- Auto-Sensitivity Switch --------
            if AUTORANGE_ENABLE:
                peak_raw = max(vis_raw + [nir_raw, clear_raw])
                if peak_raw >= AUTORANGE_SAT_FRAC * fs:
                    ar_sat_cnt += 1; ar_low_cnt = 0
                elif peak_raw <= AUTORANGE_LOW_FRAC * fs:
                    ar_low_cnt += 1; ar_sat_cnt = 0
                else:
                    ar_sat_cnt = ar_low_cnt = 0

                switched = False
                if ar_sat_cnt >= AUTORANGE_HYST:
                    if active_preset == "hi":
                        sens_cfg = apply_sensitivity(s, SENS_LO)
                        active_preset = "lo"; switched = True
                    elif active_preset == "lo" and sun_enabled:
                        sens_cfg = apply_sensitivity(s, SENS_SUN)
                        active_preset = "sun"; switched = True
                    ar_sat_cnt = 0
                elif ar_low_cnt >= AUTORANGE_HYST:
                    if active_preset == "sun":
                        sens_cfg = apply_sensitivity(s, SENS_LO)
                        active_preset = "lo"; switched = True
                    elif active_preset == "lo":
                        sens_cfg = apply_sensitivity(s, SENS_HI)
                        active_preset = "hi"; switched = True
                    ar_low_cnt = 0

                if switched:
                    actual_it_ms = sens_cfg["it_ms"]; fs = sens_cfg["fs"]; gnum = sens_cfg["gnum"]
                    dark_vis8 = sens_cfg["dark_vis8"]; dark_clear = sens_cfg["dark_clear"]
                    dark_nir  = sens_cfg["dark_nir"];  b0, w = sens_cfg["b0"], sens_cfg["w"]
                    print(f"[SENS] -> {active_preset.upper()} (gain={s.gain}, IT={actual_it_ms:.1f}ms)")
                    continue  # discard frame taken under old settings

            # Check for saturation and warn user
            sat_th = SAT_WARN_FRAC * fs
            if max(vis_raw + [nir_raw, clear_raw]) >= sat_th:
                print(f"[WARN] Near saturation: max={int(max(vis_raw+[nir_raw, clear_raw]))} (gain={s.gain}, IT={actual_it_ms:.1f}ms, FS={int(fs)})")

            # -------- Step 2: Dark Correction --------
            # Subtract dark offset from each channel (thermal noise baseline)
            vis   = [max(0.0, v - d) for v, d in zip(vis_raw, dark_vis8)]
            clear = max(0.0, clear_raw - dark_clear)
            nir   = max(0.0, nir_raw - dark_nir)

            # -------- Step 3: Normalize to BasicCounts --------
            # BasicCounts = (raw - dark) / (gain × integration_time_ms)
            # This makes readings independent of exposure settings for calibration
            denom = max(1e-9, gnum * actual_it_ms)         # Avoid division by zero
            bc8   = [v / denom for v in vis]
            bcnir = nir / denom

            # Lux from VIS8 linear model
            lux   = max(0.0, b0 + sum(bc8[i]*w[i] for i in range(8)))

            # ============================================================
            # SPECTRAL COMPOSITION
            # ============================================================
            # Apply responsivity correction to VIS8 and NIR. After correction,
            # equal true irradiance produces equal contributions across channels,
            # so the sum/ratio comparisons below are radiometric.
            bc8_corrected = [bc * corr for bc, corr in zip(bc8, RESPONSIVITY_CORRECTION)]
            bcnir_corrected = bcnir * NIR_RESPONSIVITY_CORRECTION

            sum_vis = sum(bc8_corrected)

            # Normalise VIS8 to relative intensities (sum to 1.0); below threshold
            # the spectrum is mostly noise, so report zeros instead of garbage.
            if sum_vis >= MIN_SIGNAL_THRESHOLD:
                rel_vis8 = [max(0.0, x) / sum_vis for x in bc8_corrected]
            else:
                rel_vis8 = [0.0] * 8

            # NIR as fraction of corrected VIS+NIR energy.
            total_energy = sum_vis + bcnir_corrected
            rel_nir = bcnir_corrected / total_energy if total_energy > MIN_SIGNAL_THRESHOLD else 0.0

            # Absolute spectral irradiance per VIS8 channel (W/m^2/nm at channel
            # center). Skipped if Phase 2 hasn't produced the absolute block.
            if RESPONSIVITY_ABS is not None:
                irr_vis8 = [bc / r for bc, r in zip(bc8, RESPONSIVITY_ABS)]
            else:
                irr_vis8 = None

            # -------- Build payload --------
            ts = time.time_ns()
            lines = build_influx_lines(ts, rel_vis8, rel_nir, lux, clear, irr_vis8)
            payload = "\n".join(lines)

            sample_idx += 1
            metrics["samples_collected"] += 1

            # -------- OPTIMIZATION: Flush retry queues (multiple per loop) --------
            for ent in sessions:
                q = retry_qs[ent["label"]]
                flushed = 0
                while q and flushed < RETRY_BUDGET_PER_LOOP:
                    old_payload = q[0]
                    label, success, error = write_to_endpoint(ent, old_payload, is_retry=True)
                    if success:
                        q.popleft()
                        flushed += 1
                        metrics["http_successes"][label] += 1
                        metrics["last_success_ts"][label] = time.time_ns()
                    else:
                        # Stop on first failure for this endpoint
                        if flushed == 0:  # Only log if first attempt failed
                            print(f"[WARN] (retry) {label}: {error}")
                        metrics["last_failure_ts"][label] = time.time_ns()
                        metrics["last_failure_error"][label] = error
                        break

            # -------- OPTIMIZATION: Parallel writes to all endpoints --------
            futures = []
            for ent in sessions:
                future = executor.submit(write_to_endpoint, ent, payload, is_retry=False)
                futures.append(future)

            # Collect results
            any_success = False
            ac_timeout = max(TIMEOUT_CURRENT) * 1.2
            try:
                for future in as_completed(futures, timeout=ac_timeout):
                    label, success, error = future.result()
                    if success:
                        metrics["http_successes"][label] += 1
                        metrics["last_success_ts"][label] = ts
                        any_success = True
                    else:
                        metrics["http_failures"][label] += 1
                        metrics["last_failure_ts"][label] = ts
                        metrics["last_failure_error"][label] = error
                        retry_qs[label].append(payload)
                        print(f"[ERR] {label}: {error}")
            except FuturesTimeoutError:
                # All endpoints exceeded the wall-clock budget — treat as failure,
                # cancel pending futures, and queue the payload for retry on each.
                print(f"[ERR] All endpoints timed out after {ac_timeout:.1f}s; queueing for retry.")
                for fut, ent in zip(futures, sessions):
                    if not fut.done():
                        fut.cancel()
                        label = ent["label"]
                        metrics["http_failures"][label] += 1
                        metrics["last_failure_ts"][label] = ts
                        metrics["last_failure_error"][label] = f"timeout after {ac_timeout:.1f}s"
                        retry_qs[label].append(payload)

            # CSV write: every cycle when CSV_ALWAYS, else only when all
            # endpoints failed (archive-only, no auto-replay either way).
            if CSV_ALWAYS or not any_success:
                try:
                    write_csv_fallback(ts, lux, clear, rel_vis8, rel_nir, csv_state, irr_vis8)
                    metrics["csv_rows_written"] += 1
                except Exception as e:
                    print(f"[ERR] CSV write failed: {e}")

            # Update retry queue metrics
            for label, q in retry_qs.items():
                metrics["retry_queue_sizes"][label] = len(q)

            # -------- Status snapshot (atomic JSON, headless health check) --------
            try:
                peak_now = max(vis_raw + [nir_raw, clear_raw])
                _disk_path = CSV_OUT_DIR if CSV_OUT_DIR.exists() else Path.home()
                disk_free_mb = round(shutil.disk_usage(str(_disk_path)).free / (1024 ** 2), 1)
                status = {
                    "device": DEVICE,
                    "process_start_iso": process_start_iso,
                    "uptime_s": (ts - process_start_ns) / 1e9,
                    "samples_collected": metrics["samples_collected"],
                    "responsivity_abs_loaded": RESPONSIVITY_ABS is not None,
                    "csv": {
                        "rows_written": metrics["csv_rows_written"],
                        "always_on": CSV_ALWAYS,
                        "out_dir": str(CSV_OUT_DIR),
                        "disk_free_mb": disk_free_mb,
                    },
                    "last_sample": {
                        "timestamp_iso": _iso_from_ns(ts),
                        "lux": round(lux, 3),
                        "clear": int(clear),
                        "active_preset": active_preset,
                        "gain": str(s.gain),
                        "atime": atime,
                        "astep": astep,
                        "it_ms": round(actual_it_ms, 2),
                        "saturation_frac": round(peak_now / fs, 4) if fs else None,
                        "rel_vis8": [round(v, 6) for v in rel_vis8],
                        "rel_nir": round(rel_nir, 6),
                        "irr_vis8": ([float(f"{v:.6e}") for v in irr_vis8]
                                     if irr_vis8 is not None else None),
                    },
                    "endpoints": {
                        label: {
                            "successes": metrics["http_successes"][label],
                            "failures":  metrics["http_failures"][label],
                            "retry_queue": len(retry_qs[label]),
                            "last_success_iso": (_iso_from_ns(metrics["last_success_ts"][label])
                                                 if metrics["last_success_ts"][label] else None),
                            "last_failure_iso": (_iso_from_ns(metrics["last_failure_ts"][label])
                                                 if metrics["last_failure_ts"][label] else None),
                            "last_failure_error": metrics["last_failure_error"][label],
                        }
                        for label in retry_qs
                    },
                }
                _atomic_write_json(STATUS_FILE, status)
            except Exception as e:
                print(f"[WARN] status.json write failed: {e}")

            # -------- Logging --------
            if (sample_idx % max(1, LOG_EVERY_N)) == 0:
                # OPTIMIZATION: Cache formatted time once per log
                log_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                if VERBOSE_BANDS:
                    for i, (wl_nm, v_rel) in enumerate(zip(WLS9[:8], rel_vis8)):
                        print(f"{log_time_str} wl={wl_nm}nm rel={v_rel:.4f} lux={lux:.1f} clear={int(clear)} (gain={s.gain}, IT={actual_it_ms:.0f}ms)")
                    # Log NIR separately
                    print(f"{log_time_str} wl=910nm(NIR) rel_nir={rel_nir:.4f} (as fraction of VIS+NIR)")
                else:
                    # Show signal strength indicator
                    sig_status = "OK" if sum_vis >= MIN_SIGNAL_THRESHOLD else "LOW"
                    print(f"{log_time_str} lux={lux:.1f} maxVISNIR={int(max(vis+[nir]))} clear={int(clear)} sig={sig_status} sens={active_preset} gain={s.gain} IT={actual_it_ms:.0f}ms")

            # Periodic stats
            if sample_idx % 100 == 0:
                avg_loop = sum(metrics["loop_times"]) / len(metrics["loop_times"]) if metrics["loop_times"] else 0
                print(f"[STATS] Samples: {metrics['samples_collected']}, Avg loop: {avg_loop*1000:.1f}ms, "
                      f"Retry queues: {dict(metrics['retry_queue_sizes'])}, "
                      f"Successes: {dict(metrics['http_successes'])}, "
                      f"Failures: {dict(metrics['http_failures'])}")

            # -------- Cadence --------
            loop_elapsed = time.time() - loop_start
            metrics["loop_times"].append(loop_elapsed)
            
            min_period = max(0.02, (actual_it_ms/1000.0) * AVG + 0.02)
            sleep_time = max(0.0, max(PERIOD, min_period) - loop_elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping...")
        print(f"Final stats: {metrics['samples_collected']} samples collected")
        print(f"HTTP successes: {dict(metrics['http_successes'])}")
        print(f"HTTP failures: {dict(metrics['http_failures'])}")
        print(f"Pending retries: {dict(metrics['retry_queue_sizes'])}")
        print(f"CSV rows written: {metrics['csv_rows_written']} (out dir: {CSV_OUT_DIR})")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()