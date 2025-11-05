#!/usr/bin/env python3
# AS7341 → InfluxDB v1 (VIS8 + NIR fractions + calibrated lux)
# - 9-bin spectral fractions (415..680 nm + ~910 nm NIR), CLEAR excluded from composition
# - Lux from 8-band linear model: lux ≈ b0 + Σ w[i]*bc[i], where bc = (raw-dark)/(gain * t_int_ms)
# - Dynamic autorange thresholds tied to ADC full-scale ((ATIME+1)*(ASTEP+1), capped 65535)
# - Batched averaging, canonical timing, per-endpoint retry queues

import time, json, os
from pathlib import Path
from collections import deque
import requests, board, busio
from adafruit_as7341 import AS7341, Gain

# ============================
# USER CONFIG (edit here)
# ============================
DEVICE = "RPi-1"                 # Tag written to Influx (Device=...)
MEAS   = "LIGHT"                 # Measurement name for spectral fractions

# Canonical integration timing (t_int(ms) = (ATIME+1)*(ASTEP+1)*2.78e-3)
ATIME_D = 99                     # AS7341 ATIME register
ASTEP_D = 999                    # AS7341 ASTEP register
GAIN_D  = Gain.GAIN_64X          # One of: 0.5X,1X,2X,...,512X

AVG     = 10                     # frames averaged per sample
PERIOD  = 0.0                    # extra sleep seconds per loop (0 = as fast as pipeline allows)
VERBOSE_BANDS = False            # log each band fraction every LOG_EVERY_N samples
LOG_EVERY_N   = 1

# Autorange (gain/astep) based on dynamic thresholds vs ADC full-scale
AUTORANGE_ENABLE = True
AUTORANGE_HYST   = 2             # consecutive hits before changing gain
ADJUST_ASTEPS    = False         # allow changing ASTEP if gain limits reached
SAT_WARN_FRAC    = 0.875         # warn/autorange when any band exceeds 87.5% of full-scale
UNDERFLOW_FRAC   = 0.003         # raise gain when all bands below 0.3% of full-scale

# InfluxDB v1 endpoints: list of (host, port, db)
ENDPOINTS = [
    ("10.239.99.73", 8086, "AAB"),
    ("10.239.99.97", 8086, "AAB"),
]
MAX_RETRY_QUEUE = 500

# Calibration and dark files (expected in PROJECT ROOT)
# This script lives in .../as7341-spectral/src/, so project root is parent of this file's dir.
BASE_DIR  = Path(__file__).resolve().parent.parent
DARK_FILE = BASE_DIR / "as7341_dark.json"      # {"meta":{"gain":"GAIN_64X","atime":..,"astep":..},"clear":int,"nm415":int,...,"nir":int}
CAL_FILE  = BASE_DIR / "as7341_lux_cal.json"   # {"b0":float,"w":[8 floats]}

# ============================
# Tables / helpers
# ============================
# VIS8 + NIR (CLEAR is separate and not part of composition)
BANDS9 = ["nm415","nm445","nm480","nm515","nm555","nm590","nm630","nm680","nir"]
WLS9   = [  415,    445,    480,    515,    555,    590,    630,    680,    910 ]  # NIR ~910 nm typical

VIS8 = BANDS9[:8]

GAIN_ORDER = [
    Gain.GAIN_0_5X, Gain.GAIN_1X,   Gain.GAIN_2X,   Gain.GAIN_4X,   Gain.GAIN_8X,
    Gain.GAIN_16X,  Gain.GAIN_32X,  Gain.GAIN_64X,  Gain.GAIN_128X, Gain.GAIN_256X, Gain.GAIN_512X,
]
GAIN_MULT = {
    Gain.GAIN_0_5X:0.5,  Gain.GAIN_1X:1.0,   Gain.GAIN_2X:2.0,   Gain.GAIN_4X:4.0,
    Gain.GAIN_8X:8.0,    Gain.GAIN_16X:16.0, Gain.GAIN_32X:32.0, Gain.GAIN_64X:64.0,
    Gain.GAIN_128X:128.0,Gain.GAIN_256X:256.0,Gain.GAIN_512X:512.0
}

def integration_time_ms(atime:int, astep:int)->float:
    # t_int = (ATIME+1)*(ASTEP+1)*2.78e-3 ms
    return (atime + 1) * (astep + 1) * 2.78e-3

def adc_fullscale(atime:int, astep:int)->int:
    # ADC full-scale counts before clipping; ((ATIME+1)*(ASTEP+1)) saturated at 65535
    fs = (atime + 1) * (astep + 1)
    return 65535 if fs > 65535 else fs

def current_gain_mult(s:AS7341)->float:
    return float(GAIN_MULT.get(s.gain, 1.0))

def load_cal(path:Path):
    if not path.exists():
        raise FileNotFoundError(f"Calibration file '{path}' not found.")
    with open(path, "r") as f:
        J = json.load(f)
    b0 = float(J["b0"]); w = [float(x) for x in J["w"]]
    if len(w)!=8: raise ValueError("Calibration file must contain 8 weights (w).")
    return b0, w

def load_dark(path:Path):
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
    if not dark_meta: return False
    try:
        return (str(dark_meta.get("gain",""))==str(gain) and
                int(dark_meta.get("atime",-1))==int(atime) and
                int(dark_meta.get("astep",-1))==int(astep))
    except Exception:
        return False

def read_vis_clear_nir(s:AS7341):
    # Return ([8 VIS floats], clear float, nir float).
    # Use explicit properties (stable across Blinka versions).
    try:
        vis = [float(s.channel_415nm), float(s.channel_445nm), float(s.channel_480nm),
               float(s.channel_515nm), float(s.channel_555nm), float(s.channel_590nm),
               float(s.channel_630nm), float(s.channel_680nm)]
        clear = float(s.channel_clear)
        nir   = float(s.channel_nir)
        return vis, clear, nir
    except Exception:
        pass

    # Fallback via all_channels, attempt to infer ordering (driver-dependent)
    if hasattr(s, "all_channels"):
        ac = list(s.all_channels)
        if len(ac) >= 10:
            vis = list(map(float, ac[:8]))
            tail = ac[8:10]
            clear_guess, nir_guess = (float(tail[0]), float(tail[1]))
            if clear_guess < nir_guess:
                # Some stacks report [F1..F8, NIR, CLEAR]
                clear_guess, nir_guess = nir_guess, clear_guess
            return vis, clear_guess, nir_guess
        elif len(ac) >= 9:
            vis = list(map(float, ac[:8]))
            clear = float(ac[8]); nir = 0.0
            return vis, clear, nir

    raise RuntimeError("Unable to read VIS/CLEAR/NIR channels from AS7341 driver.")

def avg_frames(s:AS7341, n:int):
    n = max(1, int(n))
    acc8 = [0.0]*8; acc_clear = 0.0; acc_nir = 0.0
    for _ in range(n):
        v8, c, nval = read_vis_clear_nir(s)
        for i in range(8): acc8[i]+=float(v8[i])
        acc_clear += float(c)
        acc_nir   += float(nval)
    return [x/n for x in acc8], acc_clear/n, acc_nir/n

def autorange_update(s:AS7341, max_after_dark:float, state:dict):
    if not AUTORANGE_ENABLE: return
    fs = float(adc_fullscale(s.atime, s.astep))
    sat_th = SAT_WARN_FRAC * fs
    lo_th  = UNDERFLOW_FRAC * fs
    hi = state.setdefault("hi_cnt",0); lo = state.setdefault("lo_cnt",0)
    if max_after_dark >= sat_th:
        state["hi_cnt"]=hi+1; state["lo_cnt"]=0
    elif max_after_dark <= lo_th:
        state["lo_cnt"]=lo+1; state["hi_cnt"]=0
    else:
        state["hi_cnt"]=state["lo_cnt"]=0
        return
    if state["hi_cnt"]>=AUTORANGE_HYST:
        gi = GAIN_ORDER.index(s.gain) if s.gain in GAIN_ORDER else GAIN_ORDER.index(GAIN_D)
        if gi>0:
            s.gain = GAIN_ORDER[gi-1]
            if state.get("verbose",False): print(f"[AUTO] Gain -> {s.gain} (down)")
        elif ADJUST_ASTEPS and s.astep>0:
            s.astep = max(0, s.astep//2)
            if state.get("verbose",False): print(f"[AUTO] ASTEP -> {s.astep} (down)")
        state["hi_cnt"]=0
    if state["lo_cnt"]>=AUTORANGE_HYST:
        gi = GAIN_ORDER.index(s.gain) if s.gain in GAIN_ORDER else GAIN_ORDER.index(GAIN_D)
        if gi < len(GAIN_ORDER)-1:
            s.gain = GAIN_ORDER[gi+1]
            if state.get("verbose",False): print(f"[AUTO] Gain -> {s.gain} (up)")
        elif ADJUST_ASTEPS and s.astep<65534:
            s.astep = min(65534, s.astep*2 + 1)
            if state.get("verbose",False): print(f"[AUTO] ASTEP -> {s.astep} (up)")
        state["lo_cnt"]=0

def build_influx_lines(ts_ns:int, rel9, lux_value, clear_value):
    lines = []
    for wl_nm, v_rel in zip(WLS9, rel9):
        lines.append(f"{MEAS},Device={DEVICE},wavelength_nm={wl_nm} rel_intensity={v_rel:.6f} {ts_ns}")
    lines.append(f"LIGHT_LUX,Device={DEVICE},method=lin_basic lux={lux_value:.3f},clear={int(clear_value)} {ts_ns}")
    return lines

# ============================
# Main
# ============================
def main():
    # Sensor init
    i2c = board.I2C()
    s = AS7341(i2c)
    s.atime = ATIME_D; s.astep = ASTEP_D; s.gain = GAIN_D
    try: s.flicker_detection_enabled = False
    except Exception: pass

    it_ms = integration_time_ms(s.atime, s.astep)

    # Dark & cal
    darkJ = load_dark(DARK_FILE)
    dark_meta_ok = dark_ok_for_settings(darkJ.get("meta", None), str(s.gain), int(s.atime), int(s.astep))
    if not dark_meta_ok and darkJ.get("meta", None) is not None:
        print("[WARN] Dark file meta does not match current settings (gain/ATIME/ASTEP). Skipping dark offsets.")
    dark_vis8 = [int(darkJ[b]) if dark_meta_ok else 0 for b in VIS8]
    dark_clear = int(darkJ["clear"]) if dark_meta_ok else 0
    dark_nir   = int(darkJ.get("nir", 0)) if dark_meta_ok else 0

    b0, w = load_cal(CAL_FILE)  # weights for VIS8 -> lux

    # Sessions + per-endpoint retry queues
    sessions = []
    retry_qs = {}  # label -> deque of payload strings
    for host, port, db in ENDPOINTS:
        label = f"{host}:{port}/{db}"
        sessions.append({
            "url": f"http://{host}:{port}/write",
            "params": {"db": db, "precision": "ns"},
            "sess": requests.Session(),
            "label": label,
        })
        retry_qs[label] = deque(maxlen=MAX_RETRY_QUEUE)

    print("AS7341 -> Influx v1 fan-out:")
    for ent in sessions: print("  -", ent["label"])
    print(f"Start: Device={DEVICE}, gain={s.gain}, ATIME={s.atime}, ASTEP={s.astep}, IT≈{it_ms:.1f} ms, AVG={AVG}, PERIOD={PERIOD}s")
    if not dark_meta_ok: print("[INFO] Dark offsets inactive (no file or meta mismatch).")

    ar_state = {"verbose": True}
    sample_idx = 0

    try:
        while True:
            t0 = time.time()

            vis_raw, clear_raw, nir_raw = avg_frames(s, AVG)

            # Warn near saturation using dynamic FS
            fs = float(adc_fullscale(s.atime, s.astep))
            sat_th = SAT_WARN_FRAC * fs
            if max(vis_raw + [nir_raw, clear_raw]) >= sat_th:
                print(f"[WARN] Near saturation: max={int(max(vis_raw+[nir_raw, clear_raw]))} (gain={s.gain}, IT≈{integration_time_ms(s.atime,s.astep):.1f}ms, FS={int(fs)})")

            # Dark-correct
            vis   = [max(0.0, v - d) for v, d in zip(vis_raw, dark_vis8)]
            clear = max(0.0, clear_raw - dark_clear)
            nir   = max(0.0, nir_raw - dark_nir)

            # Exposure-normalize (bc = counts / (gain * t_int_ms))
            gnum  = current_gain_mult(s)
            it_ms = integration_time_ms(s.atime, s.astep)
            denom = max(1e-9, gnum * it_ms)
            bc8   = [v / denom for v in vis]
            bcnir = nir / denom

            # Lux from VIS8 linear model
            lux   = max(0.0, b0 + sum(bc8[i]*w[i] for i in range(8)))

            # Spectral composition (9 bins: VIS8 + NIR), exclude CLEAR
            bc9   = bc8 + [bcnir]
            sum_bc = sum(x for x in bc9 if x > 0)
            rel9   = ([max(0.0, x)/sum_bc for x in bc9] if sum_bc > 0 else [0.0]*9)

            # Autorange based on max of VIS+NIR (CLEAR excluded from decision)
            autorange_update(s, max(vis + [nir]), ar_state)

            # Payload
            ts = time.time_ns()
            payload = "\n".join(build_influx_lines(ts, rel9, lux, clear))

            sample_idx += 1
            if (sample_idx % max(1, LOG_EVERY_N)) == 0:
                if VERBOSE_BANDS:
                    for wl_nm, v_rel in zip(WLS9, rel9):
                        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} wl={wl_nm}nm rel={v_rel:.4f} lux={lux:.1f} clear={int(clear)} (gain={s.gain}, IT≈{it_ms:.0f}ms)")
                else:
                    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} lux={lux:.1f} maxVISNIR={int(max(vis+[nir]))} clear={int(clear)} gain={s.gain} IT≈{it_ms:.0f}ms")

            # -------- Retry queues: flush one payload per endpoint (non-blocking) --------
            for ent in sessions:
                q = retry_qs[ent["label"]]
                if q:
                    old_payload = q[0]
                    try:
                        r = ent["sess"].post(ent["url"], params=ent["params"], data=old_payload, timeout=(2,5))
                        if r.status_code == 204:
                            q.popleft()
                        else:
                            print(f"[WARN] (retry) {ent['label']} {r.status_code}: {r.text.strip()}")
                    except requests.RequestException as e:
                        print(f"[ERR]  (retry) {ent['label']} HTTP: {e}")
                        # keep queued

            # -------- Current payload: send to ALL endpoints; enqueue only the failing ones --------
            for ent in sessions:
                try:
                    r = ent["sess"].post(ent["url"], params=ent["params"], data=payload, timeout=(2,5))
                    if r.status_code != 204:
                        print(f"[WARN] {ent['label']} write {r.status_code}: {r.text.strip()}")
                        retry_qs[ent["label"]].append(payload)
                except requests.RequestException as e:
                    print(f"[ERR]  {ent['label']} HTTP: {e}")
                    retry_qs[ent["label"]].append(payload)

            # -------- Cadence --------
            loop_elapsed = time.time() - t0
            min_period = max(0.02, (integration_time_ms(s.atime, s.astep)/1000.0) * AVG + 0.02)
            time.sleep(max(PERIOD, min_period - loop_elapsed, 0.0))

    except KeyboardInterrupt:
        print("\nStopping.")

if __name__ == "__main__":
    main()
