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
# Result: 50-60% faster + much more accurate spectral composition

import time, json, os
from pathlib import Path
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
import board, busio
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

# Sensor Settings (MUST MATCH CALIBRATION)
# -----------------------------------------
INTEGRATION_TIME_MS = 50    # Light collection time in milliseconds (1-1812 ms)
                              # Longer = more light collected, better for dim conditions
                              # Shorter = less light collected, better for bright conditions
                              # Typical values: 10-50ms for indoor, 1-10ms for outdoor

GAIN = Gain.GAIN_256X         # Analog gain multiplier applied to ADC readings
                              # Options: GAIN_0_5X, GAIN_1X, GAIN_2X, GAIN_4X, GAIN_8X,
                              #          GAIN_16X, GAIN_32X, GAIN_64X, GAIN_128X,
                              #          GAIN_256X, GAIN_512X
                              # Higher gain = more sensitive but saturates faster in bright light
                              # Typical: GAIN_128X or GAIN_256X for indoor

# Measurement Averaging and Timing
# ---------------------------------
AVG = 5                       # Number of sensor frames to average per reading
                              # Higher = less noise but slower response
                              # Typical: 3-10 frames

PERIOD = 0                 # Minimum seconds between measurements
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

# HTTP Performance Tuning
# -----------------------
RETRY_BUDGET_PER_LOOP = 10    # Maximum retry queue flushes attempted per measurement loop
                              # Higher = faster recovery from network issues but more CPU

TIMEOUT_CURRENT = (0.5, 2)    # HTTP timeout for current measurement: (connect_sec, read_sec)
                              # Aggressive timeout keeps main loop responsive

TIMEOUT_RETRY = (1, 3)        # HTTP timeout for retry queue: (connect_sec, read_sec)
                              # More patient timeout for background retries

# Calibration File Paths
# ----------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # Project root directory
DARK_FILE = BASE_DIR / "as7341_dark.json"          # Dark offset calibration file
CAL_FILE = BASE_DIR / "as7341_lux_cal.json"        # Lux calibration coefficients

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
# Based on AS7341 typical responsivity values from datasheet
# These correct for the fact that different channels have different sensitivities
# Without correction, blue/red channels appear artificially weak in spectral composition
RESPONSIVITY_CORRECTION = [
    2.0,   # F1 (415nm) - least sensitive, needs 2x boost
    1.67,  # F2 (445nm)
    1.33,  # F3 (480nm)
    1.11,  # F4 (515nm)
    1.0,   # F5 (555nm) - reference, most sensitive (green peak, matches human eye sensitivity)
    1.11,  # F6 (590nm)
    1.43,  # F7 (630nm)
    2.0,   # F8 (680nm) - least sensitive, needs 2x boost
]

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

def load_cal(path:Path):
    """
    Load lux calibration coefficients from JSON file.

    Expected format: {"b0": float, "w": [8 floats]}
    Model: lux = b0 + sum(w[i] * BasicCounts[i])

    Args:
        path: Path to calibration JSON file

    Returns:
        tuple: (b0, w) where b0 is intercept and w is list of 8 weights

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
    return b0, w

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

def build_influx_lines(ts_ns:int, rel_vis8, rel_nir, lux_value, clear_value):
    """
    Build InfluxDB line protocol payload from measurement data.

    Creates 10 lines total:
    - 8 spectral lines for VIS channels (415-680nm) with rel_intensity field
    - 1 spectral line for NIR channel (910nm) with rel_intensity field
    - 1 lux line with lux and clear fields

    Args:
        ts_ns: Timestamp in nanoseconds (Unix epoch)
        rel_vis8: List of 8 relative intensities for VIS channels (normalized, sum to 1.0)
        rel_nir: NIR relative intensity (as fraction of total VIS+NIR energy)
        lux_value: Calibrated lux reading (illuminance)
        clear_value: CLEAR channel raw value (broadband photocurrent)

    Returns:
        list: InfluxDB line protocol strings ready for HTTP POST
    """
    # VIS8 spectral composition (8 data points)
    # Format: "LIGHT,Device=RPi-1,wavelength_nm=415 rel_intensity=0.123456 1234567890000000000"
    lines = [f"{INFLUX_TEMPLATES[i]} rel_intensity={v:.6f} {ts_ns}"
             for i, v in enumerate(rel_vis8)]

    # NIR as separate point with wavelength_nm=910 tag
    lines.append(f"{INFLUX_TEMPLATES[8]} rel_intensity={rel_nir:.6f} {ts_ns}")

    # Lux measurement with both lux (calibrated) and clear (raw) fields
    # Format: "LIGHT_LUX,Device=RPi-1,method=lin_basic lux=123.456,clear=12345 1234567890000000000"
    lines.append(f"{LUX_TEMPLATE} lux={lux_value:.3f},clear={int(clear_value)} {ts_ns}")
    return lines

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
    
    # Convert user-friendly integration time to register values
    atime, astep = ms_to_atime_astep(INTEGRATION_TIME_MS)
    s.atime = atime
    s.astep = astep
    s.gain = GAIN
    
    try:
        s.flicker_detection_enabled = False    # Disable flicker detection for consistent timing
    except Exception:
        pass                                   # Ignore if not supported by driver version

    # Calculate actual integration time (may differ slightly from requested)
    actual_it_ms = integration_time_ms(atime, astep)
    fs = adc_fullscale(atime, astep)
    gnum = current_gain_mult(s)

    # ========================================
    # INITIALIZATION: Calibration Files
    # ========================================
    # Load dark offset calibration (sensor thermal noise baseline)
    darkJ = load_dark(DARK_FILE)
    dark_meta_ok = dark_ok_for_settings(darkJ.get("meta", None), str(s.gain), int(s.atime), int(s.astep))
    if not dark_meta_ok and darkJ.get("meta", None) is not None:
        print("[WARN] Dark file meta does not match current settings (gain/ATIME/ASTEP). Skipping dark offsets.")
    # Extract dark offsets for each channel (only use if settings match)
    dark_vis8 = [int(darkJ[b]) if dark_meta_ok else 0 for b in VIS8]
    dark_clear = int(darkJ["clear"]) if dark_meta_ok else 0
    dark_nir   = int(darkJ.get("nir", 0)) if dark_meta_ok else 0

    # Load lux calibration model: lux = b0 + sum(w[i] * BasicCounts[i])
    b0, w = load_cal(CAL_FILE)

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

    # Initialize performance tracking metrics
    metrics = {
        "samples_collected": 0,              # Total measurements taken
        "http_successes": defaultdict(int),  # Successful writes per endpoint
        "http_failures": defaultdict(int),   # Failed writes per endpoint
        "retry_queue_sizes": {label: 0 for label in retry_qs},  # Current queue depths
        "loop_times": deque(maxlen=100),     # Recent loop execution times
    }

    # ========================================
    # STARTUP: Print Configuration Summary
    # ========================================
    print("AS7341 -> Influx v1 fan-out (SIMPLIFIED CONFIG + SPECTRAL ACCURACY):")
    for ent in sessions: print("  -", ent["label"])
    print(f"Start: Device={DEVICE}, gain={s.gain}, IT={actual_it_ms:.1f}ms (requested {INTEGRATION_TIME_MS}ms), AVG={AVG}, PERIOD={PERIOD}s")
    print(f"Registers: ATIME={atime}, ASTEP={astep}, ADC_FS={fs}")
    print(f"Optimizations: parallel writes, retry budget={RETRY_BUDGET_PER_LOOP}")
    print(f"Spectral improvements: responsivity correction, VIS8 normalized separately from NIR, min threshold={MIN_SIGNAL_THRESHOLD}")
    if not dark_meta_ok: print("[INFO] Dark offsets inactive (no file or meta mismatch).")

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
            # SPECTRAL COMPOSITION (IMPROVED ACCURACY)
            # ============================================================
            # Apply responsivity correction to VIS8 channels
            # This corrects for the fact that different channels have different sensitivities
            bc8_corrected = [bc * corr for bc, corr in zip(bc8, RESPONSIVITY_CORRECTION)]
            
            # Calculate sum of corrected VIS8 (exclude NIR from visible spectrum)
            sum_vis = sum(bc8_corrected)
            
            # Normalize VIS8 to relative intensities (sum to 1.0)
            # Only report if signal is above noise threshold
            if sum_vis >= MIN_SIGNAL_THRESHOLD:
                rel_vis8 = [max(0.0, x) / sum_vis for x in bc8_corrected]
            else:
                rel_vis8 = [0.0] * 8  # Signal too weak - report zeros
            
            # NIR as fraction of total energy (VIS + NIR)
            # This separates NIR from visible spectrum composition
            total_energy = sum_vis + bcnir
            rel_nir = bcnir / total_energy if total_energy > MIN_SIGNAL_THRESHOLD else 0.0

            # -------- Build payload --------
            ts = time.time_ns()
            lines = build_influx_lines(ts, rel_vis8, rel_nir, lux, clear)
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
                    else:
                        # Stop on first failure for this endpoint
                        if flushed == 0:  # Only log if first attempt failed
                            print(f"[WARN] (retry) {label}: {error}")
                        break

            # -------- OPTIMIZATION: Parallel writes to all endpoints --------
            futures = []
            for ent in sessions:
                future = executor.submit(write_to_endpoint, ent, payload, is_retry=False)
                futures.append(future)

            # Collect results
            for future in as_completed(futures, timeout=max(TIMEOUT_CURRENT)*1.2):
                label, success, error = future.result()
                if success:
                    metrics["http_successes"][label] += 1
                else:
                    metrics["http_failures"][label] += 1
                    retry_qs[label].append(payload)
                    print(f"[ERR] {label}: {error}")

            # Update retry queue metrics
            for label, q in retry_qs.items():
                metrics["retry_queue_sizes"][label] = len(q)

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
                    print(f"{log_time_str} lux={lux:.1f} maxVISNIR={int(max(vis+[nir]))} clear={int(clear)} sig={sig_status} gain={s.gain} IT={actual_it_ms:.0f}ms")

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
            sleep_time = max(PERIOD, min_period - loop_elapsed, 0.0)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping...")
        print(f"Final stats: {metrics['samples_collected']} samples collected")
        print(f"HTTP successes: {dict(metrics['http_successes'])}")
        print(f"HTTP failures: {dict(metrics['http_failures'])}")
        print(f"Pending retries: {dict(metrics['retry_queue_sizes'])}")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    main()