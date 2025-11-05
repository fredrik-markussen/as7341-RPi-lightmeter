#!/usr/bin/env python3
"""
AS7341 lux calibration
----------------------
Model: lux ≈ b0 + Σ w[i]*BasicCounts[i]
where BasicCounts[i] = max(0, raw_i - dark_i) / (gain * t_int_ms)

Improvements over baseline:
- Uses VIS8 by default (optionally include CLEAR/NIR as regressors with flags).
- Multiple captures per scene (--captures) with median aggregation.
- K-fold cross‑validation RMSE and R^2.
- One‑pass MAD outlier rejection (configurable).
- Optional ridge regression (--ridge alpha).
- Optional non‑negative coefficients constraint via projected gradient (--nnls-lite).
- Bootstrap confidence intervals for coefficients (--boot N).
- Dynamic warnings for saturation/underflow tied to ADC full‑scale.
- Saves rich metadata for runtime compatibility checks.

Run with the SAME gain/ATIME/ASTEP used in measurement.
"""

import argparse, json, os, sys, time, math, random
from statistics import median
import numpy as np

import board, busio
from adafruit_as7341 import AS7341, Gain

# ---------------- Constants ----------------
BANDS8 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680"]
WLS8   = [415,    445,    480,    515,    555,    590,    630,    680   ]

GAIN_MULT = {
    Gain.GAIN_0_5X: 0.5,  Gain.GAIN_1X:   1.0,  Gain.GAIN_2X:   2.0,  Gain.GAIN_4X:   4.0,
    Gain.GAIN_8X:   8.0,  Gain.GAIN_16X: 16.0,  Gain.GAIN_32X: 32.0,  Gain.GAIN_64X: 64.0,
    Gain.GAIN_128X:128.0, Gain.GAIN_256X:256.0, Gain.GAIN_512X:512.0,
}

# ---------------- Helpers ----------------
def integration_time_ms(atime:int, astep:int) -> float:
    # Datasheet: t = (ATIME+1)*(ASTEP+1)*2.78 µs
    return (atime + 1) * (astep + 1) * 2.78e-3

def adc_fullscale(atime:int, astep:int)->int:
    fs = (atime + 1) * (astep + 1)
    return 65535 if fs > 65535 else fs

def current_gain_mult(s: AS7341) -> float:
    return float(GAIN_MULT.get(s.gain, 1.0))

def read_vis_clear_nir(s: AS7341):
    """Return (vis8 list, clear, nir)."""
    vis = [
        float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm), float(s.channel_515nm),
        float(s.channel_555nm), float(s.channel_590nm), float(s.channel_630nm), float(s.channel_680nm),
    ]
    clear = float(s.channel_clear)
    nir   = float(s.channel_nir)
    return vis, clear, nir

def avg_captures(s: AS7341, n: int):
    """Median of n captures to reduce flicker and spikes."""
    vis_buckets = [[] for _ in range(8)]
    clear_b = []
    nir_b = []
    for _ in range(max(1, n)):
        v8, c, nval = read_vis_clear_nir(s)
        for i in range(8): vis_buckets[i].append(v8[i])
        clear_b.append(c); nir_b.append(nval)
        time.sleep(0.01)
    vis_med = [median(ch) for ch in vis_buckets]
    return vis_med, median(clear_b), median(nir_b)

def load_dark(path: str, expect_nir: bool = True):
    if not (path and os.path.exists(path)):
        print(f"[INFO] No dark file at '{path}', using zero offsets.")
        base = {"meta": None, "clear": 0}
        base.update({b:0 for b in BANDS8})
        if expect_nir: base["nir"] = 0
        return base
    with open(path, "r") as f:
        J = json.load(f)
    return J

def dark_ok_for_settings(dark_meta, gain, atime, astep) -> bool:
    if not dark_meta: return False
    try:
        return (
            str(dark_meta.get("gain","")) == str(gain) and
            int(dark_meta.get("atime",-1)) == int(atime) and
            int(dark_meta.get("astep",-1)) == int(astep)
        )
    except Exception:
        return False

def design_matrix(X, use_clear=False, use_nir=False):
    cols = X["vis"][:]
    names = BANDS8[:]
    if use_clear:
        cols.append(X["clear"])
        names.append("clear")
    if use_nir:
        cols.append(X["nir"])
        names.append("nir")
    return np.array(cols, dtype=float), names

def fit_ridge(Phi, y, alpha):
    A = Phi.T @ Phi
    A += alpha * np.eye(A.shape[0])
    beta = np.linalg.solve(A, Phi.T @ y)
    return beta

def fit_ols(Phi, y):
    beta, *_ = np.linalg.lstsq(Phi, y, rcond=None)
    return beta

def nnls_projected_gradient(Phi, y, iters=2000, lr=1e-3):
    """Lightweight NNLS-like: gradient steps on squared loss with non-negativity projection."""
    d = Phi.shape[1]
    beta = np.maximum(0.0, np.random.randn(d) * 1e-3)
    for _ in range(iters):
        grad = 2 * Phi.T @ (Phi @ beta - y)
        beta -= lr * grad
        beta = np.maximum(0.0, beta)
    return beta

def kfold_indices(n, k):
    idx = list(range(n))
    random.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    return folds

def rmse(y, yhat):
    e = y - yhat
    return float(np.sqrt(np.mean(e*e)))

def r2(y, yhat):
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="AS7341 lux calibration (VIS8 default)")
    ap.add_argument("--avg", type=int, default=10, help="captures per scene (median aggregated)")
    ap.add_argument("--samples", type=int, default=8, help="number of calibration scenes")
    ap.add_argument("--dark-file", default="as7341_dark.json")
    ap.add_argument("--out", default="as7341_lux_cal.json")
    ap.add_argument("--gain", default="GAIN_64X", help="AS7341 gain enum, e.g., GAIN_64X")
    ap.add_argument("--atime", type=int, default=99, help="ATIME register value")
    ap.add_argument("--astep", type=int, default=999, help="ASTEP register value")
    ap.add_argument("--ridge", type=float, default=0.0, help="L2 regularization α (0=OLS)")
    ap.add_argument("--madk", type=float, default=3.5, help="Reject outliers with |res| > madk*MAD (one pass)")
    ap.add_argument("--kfold", type=int, default=5, help="K-fold CV (>=2 to enable)")
    ap.add_argument("--boot", type=int, default=0, help="Bootstrap samples for coefficient CIs (0=off)")
    ap.add_argument("--use-clear", action="store_true", help="Include CLEAR as a regressor")
    ap.add_argument("--use-nir", action="store_true", help="Include NIR as a regressor")
    ap.add_argument("--nnls-lite", action="store_true", help="Enforce non-negative weights via projected gradient")
    args = ap.parse_args()

    random.seed(12345)

    # I2C & sensor
    i2c = board.I2C()
    s = AS7341(i2c)

    # Apply timing & gain
    try:
        s.gain = getattr(Gain, args.gain)
    except AttributeError:
        raise ValueError(f"Unknown gain '{args.gain}'.")
    s.atime = int(args.atime)
    s.astep = int(args.astep)
    try:
        s.flicker_detection_enabled = False
    except Exception:
        pass

    it_ms = integration_time_ms(s.atime, s.astep)
    gnum  = current_gain_mult(s)
    fs    = adc_fullscale(s.atime, s.astep)

    # Dark handling
    darkJ = load_dark(args.dark_file, expect_nir=True)
    dark_meta_ok = dark_ok_for_settings(darkJ.get("meta", None), str(s.gain), int(s.atime), int(s.astep))
    if not dark_meta_ok and darkJ.get("meta", None) is not None:
        print("[WARN] Dark file meta does not match current settings (gain/ATIME/ASTEP). Dark offsets will be ignored.")
    dark_vis = [float(darkJ.get(b, 0)) if dark_meta_ok else 0.0 for b in BANDS8]
    dark_clear = float(darkJ.get("clear", 0.0)) if dark_meta_ok else 0.0
    dark_nir   = float(darkJ.get("nir",   0.0)) if dark_meta_ok else 0.0

    print("Calibration start:")
    print(f"  gain={s.gain} ({gnum}x), ATIME={s.atime}, ASTEP={s.astep}, IT≈{it_ms:.1f} ms, avg/capture={args.avg}")
    print("Collect diverse scenes (daylight, shade, warm/cool LED, etc.).")
    print("Place lux meter & sensor co-located, same angle; avoid shadows/speculars.")

    rows = []
    for i in range(1, args.samples + 1):
        input(f"Scene {i}/{args.samples}: set up, then press Enter to capture…")
        v8_raw, clear_raw, nir_raw = avg_captures(s, args.avg)
        max_raw = max(v8_raw + [clear_raw, nir_raw])
        if max_raw >= 0.875*fs:
            print("[WARN] Near saturation; consider lower gain or shorter IT.")
        if max_raw <= 0.003*fs:
            print("[WARN] Very low counts; consider higher gain or longer IT.")

        # Dark-correct and exposure-normalize
        vis   = [max(0.0, v - d) for v, d in zip(v8_raw, dark_vis)]
        clear = max(0.0, clear_raw - dark_clear)
        nir   = max(0.0, nir_raw - dark_nir)
        denom = max(1e-9, gnum * it_ms)
        bc8   = [v / denom for v in vis]
        bcc   = clear / denom
        bcn   = nir   / denom

        print("BasicCounts:", {b: round(x, 3) for b, x in zip(BANDS8, bc8)}, "| clear:", round(bcc,1), "| nir:", round(bcn,1))
        lux = float(input("Enter lux meter reading (lux): ").strip())

        rows.append({"vis": bc8, "clear": bcc, "nir": bcn, "lux": lux})

    # Build design matrix
    X_cols, names = design_matrix({"vis": rows[0]["vis"], "clear": rows[0]["clear"], "nir": rows[0]["nir"]},
                                  use_clear=args.use_clear, use_nir=args.use_nir)
    # rebuild properly for all rows
    X_list = []
    for r in rows:
        Xr, _ = design_matrix(r, use_clear=args.use_clear, use_nir=args.use_nir)
        X_list.append(Xr)
    X = np.vstack(X_list)
    y = np.array([r["lux"] for r in rows], dtype=float)

    # Add intercept
    Phi = np.column_stack([np.ones(X.shape[0]), X])

    # Initial fit
    if args.nnls_lite:
        beta = np.zeros(Phi.shape[1])
        beta[0] = np.mean(y)  # intercept free
        beta[1:] = nnls_projected_gradient(Phi[:,1:], y - beta[0], iters=4000, lr=1e-4)
    elif args.ridge > 0:
        beta = fit_ridge(Phi, y, alpha=args.ridge)
    else:
        beta = fit_ols(Phi, y)

    yhat = Phi @ beta
    resid = y - yhat
    rmse0 = rmse(y, yhat)
    r20   = r2(y, yhat)
    condX = float(np.linalg.cond(X)) if X.shape[0] >= X.shape[1] and X.shape[1] > 0 else float("inf")

    print("\nInitial fit:")
    print(f"  RMSE = {rmse0:.3f} lux   R^2 = {r20:.4f}   cond(X) = {condX:.1f}")

    # MAD outlier rejection
    if len(y) >= 5 and args.madk > 0:
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med)) * 1.4826)
        if mad > 0:
            keep = np.abs(resid - med) <= (args.madk * mad)
            if np.count_nonzero(keep) >= max(4, X.shape[1]+1) and np.count_nonzero(~keep) > 0:
                Phi2 = Phi[keep]; y2 = y[keep]; X2 = X[keep]
                if args.nnls_lite:
                    beta = np.zeros(Phi2.shape[1]); beta[0] = np.mean(y2)
                    beta[1:] = nnls_projected_gradient(Phi2[:,1:], y2 - beta[0], iters=4000, lr=1e-4)
                elif args.ridge > 0:
                    beta = fit_ridge(Phi2, y2, alpha=args.ridge)
                else:
                    beta = fit_ols(Phi2, y2)
                yhat = Phi2 @ beta
                resid = y2 - yhat
                rmse0 = rmse(y2, yhat); r20 = r2(y2, yhat)
                condX = float(np.linalg.cond(X2)) if X2.shape[0] >= X2.shape[1] and X2.shape[1] > 0 else float("inf")
                print(f"After MAD({args.madk}) rejection: kept {np.count_nonzero(keep)}/{len(y)}")
                print(f"  RMSE = {rmse0:.3f} lux   R^2 = {r20:.4f}   cond(X) = {condX:.1f}")

    # K-fold CV
    rmse_cv = None; r2_cv = None
    if args.kfold and args.kfold >= 2 and len(y) >= args.kfold:
        folds = kfold_indices(len(y), args.kfold)
        preds = np.zeros_like(y)
        for k in range(args.kfold):
            val_idx = folds[k]
            train_idx = [i for i in range(len(y)) if i not in val_idx]
            Phi_tr, y_tr = Phi[train_idx], y[train_idx]
            Phi_va = Phi[val_idx]
            if args.nnls_lite:
                b = np.zeros(Phi_tr.shape[1]); b[0] = np.mean(y_tr)
                b[1:] = nnls_projected_gradient(Phi_tr[:,1:], y_tr - b[0], iters=4000, lr=1e-4)
            elif args.ridge > 0:
                b = fit_ridge(Phi_tr, y_tr, alpha=args.ridge)
            else:
                b = fit_ols(Phi_tr, y_tr)
            preds[val_idx] = Phi_va @ b
        rmse_cv = rmse(y, preds)
        r2_cv = r2(y, preds)
        print(f"K-fold CV (k={args.kfold}): RMSE = {rmse_cv:.3f} lux   R^2 = {r2_cv:.4f}")

    # Bootstrap CIs
    ci_lo = None; ci_hi = None
    if args.boot and args.boot > 0 and len(y) >= 4:
        Bs = []
        n = len(y)
        for _ in range(args.boot):
            idx = np.random.randint(0, n, size=n)
            Phi_b = Phi[idx]; y_b = y[idx]
            if args.nnls_lite:
                b = np.zeros(Phi_b.shape[1]); b[0] = np.mean(y_b)
                b[1:] = nnls_projected_gradient(Phi_b[:,1:], y_b - b[0], iters=2000, lr=2e-4)
            elif args.ridge > 0:
                b = fit_ridge(Phi_b, y_b, alpha=args.ridge)
            else:
                b = fit_ols(Phi_b, y_b)
            Bs.append(b)
        B = np.vstack(Bs)
        ci_lo = np.percentile(B, 2.5, axis=0).tolist()
        ci_hi = np.percentile(B,97.5, axis=0).tolist()
        print("Bootstrap 95% CIs computed for coefficients.")

    # Report
    b0 = float(beta[0]); w = list(map(float, beta[1:]))
    negs = sum(1 for c in w if c < 0)
    if negs:
        print(f"[NOTE] {negs} coefficient(s) are negative. Consider --ridge or --nnls-lite if undesired.")

    # Save calibration
    meta = {
        "gain": str(s.gain),
        "atime": int(s.atime),
        "astep": int(s.astep),
        "it_ms": it_ms,
        "features": names,
        "use_clear": bool(args.use_clear),
        "use_nir": bool(args.use_nir),
        "ridge_alpha": float(args.ridge),
        "mad_k": float(args.madk),
        "kfold": int(args.kfold),
        "boot": int(args.boot),
        "nnls_lite": bool(args.nnls_lite),
        "rmse_train": rmse0,
        "r2_train": r20,
        "rmse_cv": rmse_cv,
        "r2_cv": r2_cv
    }
    J = {"b0": b0, "w": w, "bands": names, "meta": meta}
    if ci_lo and ci_hi:
        J["coef_ci95_lo"] = ci_lo
        J["coef_ci95_hi"] = ci_hi

    with open(args.out, "w") as f:
        json.dump(J, f, indent=2)

    print(f"\nSaved calibration to {args.out}")
    print("Coefficients:")
    print(f"  b0 = {b0:.6f}")
    for name, c in zip(names, w):
        print(f"  w[{name}] = {c:.9f}")
    print(f"Train: RMSE = {rmse0:.3f} lux   R^2 = {r20:.4f}")
    if rmse_cv is not None:
        print(f"   CV: RMSE = {rmse_cv:.3f} lux   R^2 = {r2_cv:.4f}")
    if ci_lo and ci_hi:
        print("95% CI (coefficients) available in JSON.")

if __name__ == "__main__":
    main()
