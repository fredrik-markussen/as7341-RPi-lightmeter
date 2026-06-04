#!/usr/bin/env python3
"""Apply broadband-reference validation corrections to as7341_responsivity.json.

After a Phase 2 calibration, a side-by-side broadband comparison (RPi vs Seconic
C-7000) may reveal a channel reading off — e.g. nm415 over-reports because its
only calibration LEDs (385/405/435) all peak at/below its passband, so the
band-integration model gave it too-low responsivity. This scales the affected
channels' responsivities so the RPi matches the C-7000 reference, band-integrated
over each channel's FWHM (the same model Phase 2 uses).

Per channel it computes  ratio = RPi_irr / C7000_band_irr  (geometric mean across
all --pair references), then:

  --fix-channels 415,445   scale ONLY these to the median ratio of the other
                           channels — corrects spectral SHAPE, leaves the
                           well-behaved channels and the absolute scale untouched.
  --fix-channels all       scale EVERY channel by its own ratio — RPi matches the
                           C-7000 absolutely (needs trustworthy references).

Each --pair is  REF_C7000.csv:RPI_export.csv  (repeatable). The RPi export is the
Grafana "wavelength_nm,last" CSV. Writes --resp in place and appends provenance
under meta.validation. Use --dry-run to preview without writing.

Examples:
  # tonight: flatten the two blue channels against one indoor reference
  python3 src/as7341_apply_validation.py \\
      --pair "C-7000_out/comparisons/AS7341-calRE2_046_02°_7405K.csv:C-7000_out/comparisons/RPi_Light Spectrum (W_m2_nm)-data-2026-06-04 19_51_31.csv"

  # morning: add direct-sun references and re-derive across all of them
  python3 src/as7341_apply_validation.py --pair "sun_c7000.csv:sun_rpi.csv" --pair ...
"""
import argparse, json, math, re, sys
from pathlib import Path
from statistics import median

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
import as7341_calibrate as cal  # _band_average, WLS8, BANDWIDTHS_NM, BANDS8, DATASHEET_CORR


def read_c7000_spd(path):
    """C-7000 native export -> {int nm: W/m^2/nm} (dense 1 nm block wins)."""
    spd = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"Spectral Data (\d+)\[nm\],([\d.eE+\-]+)", line)
        if m:
            spd[int(m.group(1))] = float(m.group(2))
    if not spd:
        raise SystemExit(f"No 'Spectral Data' rows in {path}")
    return spd


def read_rpi_csv(path):
    """Grafana export (Time,wavelength_nm,last) -> irr8 at WLS8 (mean per wavelength)."""
    acc = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            wl = int(float(parts[-2])); v = float(parts[-1])
        except ValueError:
            continue  # header / blank
        acc.setdefault(wl, []).append(v)
    missing = [w for w in cal.WLS8 if w not in acc]
    if missing:
        raise SystemExit(f"RPi CSV {path} missing wavelengths {missing}")
    return [sum(acc[w]) / len(acc[w]) for w in cal.WLS8]


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--resp", default=str(BASE / "as7341_responsivity.json"),
                    help="responsivity JSON to correct (in place)")
    ap.add_argument("--pair", action="append", required=True, metavar="REF.csv:RPI.csv",
                    help="C-7000 reference CSV : RPi export CSV (repeatable)")
    ap.add_argument("--fix-channels", default="415,445",
                    help="comma-separated channel centers to correct, or 'all'")
    ap.add_argument("--dry-run", action="store_true", help="preview, do not write")
    args = ap.parse_args()

    # Per-channel ratio = RPi / C7000(band), geometric mean across pairs.
    logsum = [0.0] * 8
    pair_meta = []
    for pair in args.pair:
        if ":" not in pair:
            raise SystemExit(f"--pair must be REF.csv:RPI.csv, got {pair!r}")
        ref, rpi = pair.rsplit(":", 1)
        spd = read_c7000_spd(ref)
        c7 = [cal._band_average(spd, c, fw) for c, fw in zip(cal.WLS8, cal.BANDWIDTHS_NM)]
        r = read_rpi_csv(rpi)
        for i in range(8):
            if c7[i] <= 0 or r[i] <= 0:
                raise SystemExit(f"Non-positive irradiance at {cal.WLS8[i]}nm in {pair}")
            logsum[i] += math.log(r[i] / c7[i])
        pair_meta.append({"ref": Path(ref).name, "rpi": Path(rpi).name})
    npair = len(args.pair)
    ratio = [math.exp(logsum[i] / npair) for i in range(8)]

    all_mode = args.fix_channels.strip().lower() == "all"
    if all_mode:
        fix_idx = list(range(8))
        mult = list(ratio)            # absolute match: new_irr = old_irr/ratio
        target = None
    else:
        fix_nm = {int(float(x)) for x in args.fix_channels.split(",") if x.strip()}
        fix_idx = [i for i, w in enumerate(cal.WLS8) if w in fix_nm]
        if not fix_idx:
            raise SystemExit(f"--fix-channels {args.fix_channels!r} matched no channel of {cal.WLS8}")
        others = [ratio[i] for i in range(8) if i not in fix_idx]
        if not others:
            raise SystemExit("Cannot flatten: every channel is in --fix-channels (use 'all').")
        target = median(others)
        mult = [(ratio[i] / target) if i in fix_idx else 1.0 for i in range(8)]

    resp = json.loads(Path(args.resp).read_text())
    Rabs = resp["responsivity_BC_per_W_m2_nm"]
    new_Rabs = {b: Rabs[b] * mult[i] for i, b in enumerate(cal.BANDS8)}
    ref555 = new_Rabs["nm555"]
    new_corr = {b: ref555 / new_Rabs[b] for b in cal.BANDS8}

    mode = "absolute match (all channels)" if all_mode else \
           f"flatten {[cal.WLS8[i] for i in fix_idx]} to median ratio {target:.3f}"
    print(f"Validation correction — {mode}; {npair} reference pair(s).\n")
    print(f"  {'ch':>5} {'ratio':>7} {'mult':>6} {'corr old':>9} {'corr new':>9} {'datasheet':>9}")
    for i, b in enumerate(cal.BANDS8):
        old = resp["corrections"].get(b, float('nan'))
        mark = "  *" if i in fix_idx and not all_mode else ""
        print(f"  {cal.WLS8[i]:>5} {ratio[i]:>7.3f} {mult[i]:>6.3f} "
              f"{old:>9.4f} {new_corr[b]:>9.4f} {cal.DATASHEET_CORR[i]:>9.2f}{mark}")
    if not all_mode:
        print(f"\n  (* corrected; others left as-is. Residual ratio ~{target:.2f} is a "
              "uniform absolute offset that does not distort spectral shape / PFD.)")

    if args.dry_run:
        print("\n[dry-run] no file written.")
        return

    resp["responsivity_BC_per_W_m2_nm"] = new_Rabs
    resp["corrections"] = {b: round(new_corr[b], 6) for b in cal.BANDS8}
    resp.setdefault("meta", {}).setdefault("validation", []).append({
        "mode": "all" if all_mode else args.fix_channels,
        "target_ratio": target,
        "pairs": pair_meta,
        "ratio": {b: round(ratio[i], 5) for i, b in enumerate(cal.BANDS8)},
        "multiplier": {b: round(mult[i], 5) for i, b in enumerate(cal.BANDS8)},
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
    })
    Path(args.resp).write_text(json.dumps(resp, indent=2))
    print(f"\nSaved {Path(args.resp).name}. Restart as7341_influx_nir.py to apply.")


if __name__ == "__main__":
    main()
