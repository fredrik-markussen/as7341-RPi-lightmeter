#!/usr/bin/env python3
"""Offline preview: how band-integrated C-7000 irradiance changes Phase 2 corrections.

No hardware needed. Reuses the *live* AS7341 BasicCounts already stored in
as7341_responsivity.json (`raw_levels[*].bc8`) and re-derives the reference
irradiance by re-parsing C-7000_out/ with the new band-average extraction, then
recomputes per-channel responsivity exactly as run_phase2() does.

The stored `irr8` in the JSON is the OLD point-sample (the C-7000 SPD at each
channel center), so each stored level is matched to its C-7000 file by that exact
signature — robust to file-set drift and ordering (the committed JSON has fewer
levels than C-7000_out/ now holds).

Run from repo root:  python3 claude_wd/preview_band_integration.py
"""
import sys, types, json, re
from pathlib import Path

# --- Stub Pi-only / heavy imports so we can import the real parser functions ---
# _parse_c7000_native / _load_c7000_dir / _band_average are pure-python and don't
# touch these, but as7341_calibrate.py imports them at module load.
for name in ("board", "numpy"):
    sys.modules.setdefault(name, types.ModuleType(name))
_af = types.ModuleType("adafruit_as7341")
class _Gain:  # only the names referenced at import time matter
    GAIN_0_5X=GAIN_1X=GAIN_2X=GAIN_4X=GAIN_8X=GAIN_16X=GAIN_32X=GAIN_64X=0
    GAIN_128X=GAIN_256X=GAIN_512X=0
_af.AS7341 = object
_af.Gain = _Gain
sys.modules.setdefault("adafruit_as7341", _af)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import as7341_calibrate as cal  # noqa: E402

C7000_DIR = REPO / "C-7000_out"
RESP_JSON = REPO / "as7341_responsivity.json"
DATASHEET = [2.0, 1.67, 1.33, 1.11, 1.0, 1.11, 1.43, 2.0]


def read_spd(path):
    """Minimal re-read of a C-7000 file's SPD dict {int nm: W/m^2/nm} and led_nm.

    Mirrors as7341_calibrate._parse_c7000_native's parsing; kept separate here so
    we can derive BOTH the old point-sample (for fingerprint matching) and the new
    band average from the same SPD.
    """
    led_nm = None
    spd = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        for field in ("CoolLED_nm,", "CoolLED_snm,"):
            if line.startswith(field):
                v = line.split(",")[1].strip()
                if v:
                    try:
                        led_nm = int(round(float(v)))
                    except ValueError:
                        pass
        m = re.match(r"Spectral Data (\d+)\[nm\],([\d.eE+\-]+)", line)
        if m:
            spd[int(m.group(1))] = float(m.group(2))
    return led_nm, spd


def recompute(level_data, frac):
    """Replicates run_phase2()'s per-channel responsivity averaging + 555nm norm."""
    raw_resp = [None] * 8
    n_used = [0] * 8
    for i in range(8):
        vals = []
        for lvl in level_data:
            bc = lvl["bc8"][i]; ir = lvl["irr8"][i]; max_ir = max(lvl["irr8"])
            if max_ir <= 0 or ir <= 0 or bc <= 0:
                continue
            if (ir / max_ir) < frac:
                continue
            vals.append(bc / ir)
        if not vals:
            raise ValueError(f"No valid data for {cal.BANDS8[i]}")
        raw_resp[i] = sum(vals) / len(vals)
        n_used[i] = len(vals)
    ref = raw_resp[cal.WLS8.index(555)]
    corrections = [ref / r for r in raw_resp]
    return raw_resp, corrections, n_used


def main():
    old = json.loads(RESP_JSON.read_text())
    raw_levels = old["raw_levels"]
    frac = old["meta"].get("min_irr_frac", 0.2)

    # Fingerprint every C-7000 file: (led_nm, point8) -> band8, where point8 is
    # the old point sample used to match against stored irr8.
    files = []
    for f in sorted(C7000_DIR.glob("*.csv")):
        led_nm, spd = read_spd(f)
        if led_nm is None or not spd:
            continue
        point8 = tuple(spd.get(w, 0.0) for w in cal.WLS8)
        band8 = [cal._band_average(spd, c, fw)
                 for c, fw in zip(cal.WLS8, cal.BANDWIDTHS_NM)]
        files.append((f.name, led_nm, point8, band8))

    new_levels = []
    unmatched = []
    for stored in raw_levels:
        sig = tuple(stored["irr8"])
        hit = None
        for name, led_nm, point8, band8 in files:
            if led_nm != stored["led_nm"]:
                continue
            if all(abs(a - b) <= 1e-9 + 1e-6 * abs(b)
                   for a, b in zip(sig, point8)):
                hit = (name, band8)
                break
        if hit is None:
            unmatched.append(stored["led_nm"])
            continue
        new_levels.append({"led_nm": stored["led_nm"],
                           "bc8": stored["bc8"], "irr8": hit[1]})

    if unmatched:
        print(f"[WARN] {len(unmatched)} stored level(s) had no matching C-7000 "
              f"file (led_nm={unmatched}); excluded from preview.")
    if not new_levels:
        print("[ERR] No stored levels could be matched — aborting.")
        return 1

    old_corr = [old["corrections"][b] for b in cal.BANDS8]
    _, new_corr, n_used = recompute(new_levels, frac)

    print(f"Matched {len(new_levels)}/{len(raw_levels)} stored steps "
          f"(min_irr_frac={frac}).\n")
    print(f"  {'Ch':>6}  {'old':>9}  {'new':>9}  {'datasheet':>9}  "
          f"{'new-ds':>8}  {'n':>4}")
    for b, o, nw, ds, n in zip(cal.BANDS8, old_corr, new_corr, DATASHEET, n_used):
        print(f"  {b:>6}  {o:>9.4f}  {nw:>9.4f}  {ds:>9.4f}  {nw-ds:>+8.4f}  {n:>4}")
    print("\nExpect nm630/nm680 to climb toward datasheet (1.43/2.00) and "
          "nm590 to ease toward ~1.1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
