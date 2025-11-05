#!/usr/bin/env python3
# Capture dark offsets for AS7341 (cover sensor completely).
# - Records VIS8 + NIR + CLEAR
# - Saves metadata: gain, atime, astep, timestamp
# - Uses warm-up frames, then median to resist outliers
# - Must be run with the SAME timing/gain as the measurement script

import time, json, os, datetime
import board, busio
from statistics import median
from adafruit_as7341 import AS7341, Gain

# ============================
# Config (match your measurement script)
# ============================
OUT_PATH   = "as7341_dark.json"
ATIME_D    = 99          # same as main (ATIME)
ASTEP_D    = 999         # same as main (ASTEP)
GAIN_D     = Gain.GAIN_64X
WARMUP_N   = 10          # frames to discard
SAMPLE_N   = 40          # frames to keep for median

BANDS9 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680","nir"]

def read_once(s: AS7341):
    # Explicit properties are robust; the driver handles SMUX internally.
    v8 = [float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm),
          float(s.channel_515nm), float(s.channel_555nm), float(s.channel_590nm),
          float(s.channel_630nm), float(s.channel_680nm)]
    clear = float(s.channel_clear)
    nir   = float(s.channel_nir)
    return v8, clear, nir

def main():
    print("=== AS7341 Dark Capture ===")
    print("* Cover the sensor completely (opaque cap or thick tape).")
    print("* Ensure the SAME settings (gain/ATIME/ASTEP) as your measurement pipeline.")
    i2c = busio.I2C(board.SCL, board.SDA)
    s   = AS7341(i2c)

    # Apply canonical settings
    try:
        s.flicker_detection_enabled = False
    except Exception:
        pass

    # These attributes exist in recent Adafruit driver; if not, fall back to integration_time
    try:
        s.atime = ATIME_D
        s.astep = ASTEP_D
    except Exception:
        # Fallback: integration_time in ms ~ (ATIME+1)(ASTEP+1)*2.78e-3
        tint_ms = (ATIME_D + 1) * (ASTEP_D + 1) * 2.78e-3
        s.integration_time = tint_ms

    s.gain  = GAIN_D

    # Warm-up frames (let SMUX/ADC settle)
    for _ in range(WARMUP_N):
        _ = read_once(s)
        time.sleep(0.01)

    # Collect frames
    vis_frames   = [[] for _ in range(8)]
    clear_frames = []
    nir_frames   = []

    for _ in range(SAMPLE_N):
        v8, clear, nir = read_once(s)
        for i in range(8):
            vis_frames[i].append(v8[i])
        clear_frames.append(clear)
        nir_frames.append(nir)
        time.sleep(0.005)

    # Median per-channel (robust to outliers & tiny leaks)
    vis_med = [int(round(median(ch))) for ch in vis_frames]
    clear_med = int(round(median(clear_frames)))
    nir_med   = int(round(median(nir_frames)))

    # Build JSON
    meta = {
        "gain": str(s.gain),
        "atime": int(getattr(s, "atime", ATIME_D)),
        "astep": int(getattr(s, "astep", ASTEP_D)),
        "tint_ms": float(getattr(s, "integration_time", (ATIME_D + 1) * (ASTEP_D + 1) * 2.78e-3)),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
    out = {"meta": meta, "clear": clear_med}
    for name, val in zip(BANDS9, vis_med + [nir_med]):
        out[name] = int(val)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved dark offsets to", OUT_PATH)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
