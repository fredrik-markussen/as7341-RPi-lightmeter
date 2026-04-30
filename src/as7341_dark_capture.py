#!/usr/bin/env python3
# Capture dark offsets for AS7341. Cover the sensor completely before running.
# Saves per-channel medians + full metadata (gain, ATIME, ASTEP, timestamp).
# Run with the SAME settings as your measurement script for each preset.
#
# For the standard HI/LO 2-preset workflow, prefer src/as7341_calibrate.py which
# captures both presets in one guided session. This script is a standalone
# utility for ad-hoc captures with custom settings.

import argparse, time, json, datetime
import board
from statistics import median
from adafruit_as7341 import AS7341, Gain

BANDS9 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680","nir"]

GAIN_MULT = {
    Gain.GAIN_0_5X:0.5,  Gain.GAIN_1X:1.0,   Gain.GAIN_2X:2.0,   Gain.GAIN_4X:4.0,
    Gain.GAIN_8X:8.0,    Gain.GAIN_16X:16.0, Gain.GAIN_32X:32.0, Gain.GAIN_64X:64.0,
    Gain.GAIN_128X:128.0,Gain.GAIN_256X:256.0,Gain.GAIN_512X:512.0,
}

def ms_to_atime_astep(target_ms):
    total = target_ms / 2.78e-3
    for atime in range(256):
        astep = total / (atime + 1) - 1
        if 0 <= astep <= 65534:
            return atime, int(round(astep))
    raise ValueError(f"Cannot achieve {target_ms}ms integration time")

def read_once(s):
    vis = [float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm),
           float(s.channel_515nm), float(s.channel_555nm), float(s.channel_590nm),
           float(s.channel_630nm), float(s.channel_680nm)]
    return vis, float(s.channel_clear), float(s.channel_nir)

def main():
    ap = argparse.ArgumentParser(description="AS7341 dark offset capture")
    ap.add_argument("--integration-time-ms", type=float, default=50,
                    help="Integration time in ms (must match measurement script)")
    ap.add_argument("--gain", default="GAIN_256X",
                    help="Gain setting, e.g. GAIN_256X (must match measurement script)")
    ap.add_argument("--samples", type=int, default=100,
                    help="Frames to keep for median (warmup frames are additional)")
    ap.add_argument("--warmup", type=int, default=10,
                    help="Frames to discard at startup for ADC settle")
    ap.add_argument("--out", default="as7341_dark_hi.json",
                    help="Output JSON path. Use as7341_dark_hi.json or as7341_dark_lo.json "
                         "to match the measurement script's HI/LO presets.")
    args = ap.parse_args()

    gain = getattr(Gain, args.gain)
    atime, astep = ms_to_atime_astep(args.integration_time_ms)
    actual_ms = (atime + 1) * (astep + 1) * 2.78e-3

    print("=== AS7341 Dark Capture ===")
    print(f"Settings: gain={gain}, IT={actual_ms:.1f}ms (ATIME={atime}, ASTEP={astep})")
    print(f"Samples: {args.warmup} warmup + {args.samples} kept")
    print("Cover the sensor completely (opaque cap or thick tape), then press Enter.")
    input()

    i2c = board.I2C()
    s = AS7341(i2c)
    try:
        s.flicker_detection_enabled = False
    except Exception:
        pass
    s.atime = atime
    s.astep = astep
    s.gain  = gain

    print(f"Warming up ({args.warmup} frames)...", end="", flush=True)
    for _ in range(args.warmup):
        read_once(s)
        time.sleep(0.01)
    print(" done.")

    vis_frames   = [[] for _ in range(8)]
    clear_frames = []
    nir_frames   = []

    print(f"Collecting {args.samples} frames...", end="", flush=True)
    for i in range(args.samples):
        vis, clear, nir = read_once(s)
        for j in range(8):
            vis_frames[j].append(vis[j])
        clear_frames.append(clear)
        nir_frames.append(nir)
        if (i + 1) % 20 == 0:
            print(f" {i+1}", end="", flush=True)
        time.sleep(0.005)
    print(" done.")

    vis_med   = [int(round(median(ch))) for ch in vis_frames]
    clear_med = int(round(median(clear_frames)))
    nir_med   = int(round(median(nir_frames)))

    out = {
        "meta": {
            "gain":      str(s.gain),
            "atime":     int(s.atime),
            "astep":     int(s.astep),
            "tint_ms":   actual_ms,
            "samples":   args.samples,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        },
        "clear": clear_med,
    }
    for name, val in zip(BANDS9, vis_med + [nir_med]):
        out[name] = val

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved to {args.out}")
    print(f"  clear={clear_med}  nir={nir_med}")
    print("  VIS8:", {b: v for b, v in zip(BANDS9[:8], vis_med)})

if __name__ == "__main__":
    main()
