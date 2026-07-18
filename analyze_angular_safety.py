"""
How much angular space around a speaker is "safe"?

The direction-conditioned model is only asked to separate speakers by their
azimuth, so its resolution is limited by the array beamwidth (and by training:
sources were always >= MIN_ANGULAR_SEP_DEG apart). This script measures, on
controlled scenes, how extraction quality falls off as an interferer is moved
closer to the target in azimuth, and derives the separation angle below which
the single-model path should hand over to the separation fallback.

Method
------
For each separation angle in SWEEP_DEG and each trial:
  * place the TARGET at a random azimuth / distance,
  * place one INTERFERER exactly `sep` degrees away (random side),
  * optionally place extra speakers that are far away in azimuth from both
    (>= FAR_GUARD_DEG), so we can also see whether scene density matters,
  * simulate anechoic (max_order=0), equalize every source to unit power at the
    mics (so every interferer sits at 0 dB SIR -- no level shortcut),
  * run the model steered at the target azimuth,
  * score the output.

Metrics
-------
  SI-SDR        : overall extraction quality vs. the target at the reference mic.
  SIR_out (dB)  : how much of the *near interferer* leaks into the output.
                  The output is projected onto the target and onto the near
                  interferer; SIR_out is the ratio of those two energies.
                  This is the direct answer to "how close before speech leaks in".

Outputs
-------
  angular_safety/report.txt      human-readable table + derived thresholds
  angular_safety/results.json    per-condition curves + thresholds (machine-read)
  angular_safety/curve.png       SI-SDR and SIR_out vs. separation angle

The derived thresholds are consumed by fallback_separation.py (SAFE_SEPARATION_DEG).
"""

import json
import math
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import torch

import train as tr
from dataset import CENTER, FS, ROOM_SIZE, compute_steering_vector, get_mic_positions

# ---------------- settings ----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./angular_safety"
CKPT_CANDIDATES = [
    os.path.join(tr.CHECKPOINT_DIR, "best.pt"), os.path.join(tr.CHECKPOINT_DIR, "last.pt"),
    "./checkpoints_v4/best.pt", "./checkpoints_v4/last.pt",
]
SWEEP_DEG = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60, 90, 120]
N_TRIALS = 24                      # per separation angle, per condition
CONDITIONS = {                     # name -> number of EXTRA far-away speakers
    "2 speakers (target + near interferer)": 0,
    "6 speakers (1 near + 4 far)": 4,
}
FAR_GUARD_DEG = 45.0               # extra speakers stay this far from target/interferer
DIST_RANGE = (3.0, 30.0)           # metres from the array centre
SEG_LEN = int(round(tr.SEGMENT_SECONDS * FS))
PLATEAU_FROM_DEG = 60              # separations at/above this define "no crowding"
PLATEAU_DROP_DB = 3.0              # safe = within this much of the plateau
ASR_SISDR_DB = 10.0                # SI-SDR generally needed for reliable ASR
# ------------------------------------------


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------
def speaker_id(path):
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def load_clip(path, rng):
    """Load, energy-crop to the segment length (deterministic given rng state)."""
    from dataset import load_and_pad_audio, _energy_crop
    sig = load_and_pad_audio(path)
    sig = _energy_crop(sig, SEG_LEN)
    if len(sig) < SEG_LEN:
        sig = np.pad(sig, (0, SEG_LEN - len(sig)), mode="wrap")
    return sig[:SEG_LEN].astype(np.float32)


def simulate(signals, azimuths_deg, distances, mic_positions):
    """Anechoic sim; every source equalized to unit power at the mics.
    Returns premix (K, 16, T)."""
    room = pra.ShoeBox(ROOM_SIZE, fs=FS, max_order=0)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    for sig, az, dist in zip(signals, azimuths_deg, distances):
        a = math.radians(az)
        room.add_source([CENTER[0] + dist * math.cos(a),
                         CENTER[1] + dist * math.sin(a),
                         CENTER[2]], signal=sig)
    premix = room.simulate(return_premix=True)
    out = np.empty_like(premix)
    for k in range(premix.shape[0]):
        p = float(np.mean(premix[k] ** 2)) + 1e-12
        out[k] = premix[k] / math.sqrt(p)
    return out


def build_scene(files_by_spk, spk_ids, sep_deg, n_extra, rng):
    """One controlled scene. Returns (signals, azimuths, distances) with index
    0 = target, index 1 = the near interferer, 2.. = far speakers."""
    # dataset._energy_crop draws from the global RNG; seed it from `rng` so the
    # whole scene (including the crop positions) is reproducible per trial.
    random.seed(rng.randrange(2 ** 32))
    chosen_spks = rng.sample(spk_ids, k=2 + n_extra)
    signals = [load_clip(rng.choice(files_by_spk[s]), rng) for s in chosen_spks]

    az_t = rng.uniform(0.0, 360.0)
    side = rng.choice([-1.0, 1.0])
    az_i = (az_t + side * sep_deg) % 360.0
    azimuths = [az_t, az_i]

    for _ in range(n_extra):
        for _try in range(500):
            az = rng.uniform(0.0, 360.0)
            if all(_ang_diff(az, a) >= FAR_GUARD_DEG for a in azimuths):
                break
        azimuths.append(az)

    distances = [rng.uniform(*DIST_RANGE) for _ in azimuths]
    return signals, azimuths, distances


def _ang_diff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def projection_sir_db(est, target, interferer):
    """
    Energy of `est` explained by `interferer` vs. by `target`.

    Each reference is scaled to its least-squares fit of the estimate, and the
    two fitted energies are compared. High = clean target, low = the interferer
    is leaking through.
    """
    est = est.astype(np.float64)
    t = target.astype(np.float64)
    i = interferer.astype(np.float64)
    a_t = float(np.dot(est, t)) / (float(np.dot(t, t)) + 1e-12)
    a_i = float(np.dot(est, i)) / (float(np.dot(i, i)) + 1e-12)
    e_t = (a_t ** 2) * float(np.dot(t, t)) + 1e-12
    e_i = (a_i ** 2) * float(np.dot(i, i)) + 1e-12
    return 10.0 * math.log10(e_t / e_i)


def run_model(model, mixture, azimuth_deg, mic_positions, window):
    steering_np = compute_steering_vector(azimuth_deg, mic_positions, FS, tr.N_FFT)
    mix_t = torch.tensor(mixture, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    steering = torch.tensor(steering_np, dtype=torch.complex64, device=DEVICE).unsqueeze(0)
    x_stft = tr.compute_stft(mix_t, tr.N_FFT, tr.HOP_LENGTH, tr.WIN_LENGTH, window)
    with torch.no_grad():
        s_hat = model(x_stft, steering)
    wav = torch.istft(s_hat, n_fft=tr.N_FFT, hop_length=tr.HOP_LENGTH,
                      win_length=tr.WIN_LENGTH, window=window, center=True,
                      length=mixture.shape[-1])
    return wav.squeeze(0).cpu().numpy()


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------
def derive_thresholds(seps, med_sisdr, med_sir):
    """Smallest separation that is 'safe' by two independent criteria."""
    plateau_vals = [s for sep, s in zip(seps, med_sisdr) if sep >= PLATEAU_FROM_DEG]
    plateau = float(np.median(plateau_vals)) if plateau_vals else float("nan")

    def smallest_where(values, ok):
        """Smallest sep such that it and every larger sep satisfy `ok`."""
        best = None
        for sep, v in sorted(zip(seps, values), reverse=True):
            if ok(v):
                best = sep
            else:
                break
        return best

    return {
        "plateau_sisdr_db": plateau,
        "safe_deg_plateau": smallest_where(med_sisdr, lambda v: v >= plateau - PLATEAU_DROP_DB),
        "safe_deg_asr": smallest_where(med_sisdr, lambda v: v >= ASR_SISDR_DB),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    if ckpt_path is None:
        raise SystemExit("No checkpoint found.")
    model = tr.SteeringConditionedMVDRUNet(
        tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
        tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model: {ckpt_path} (epoch {ckpt['epoch']}, "
          f"val SI-SDR {ckpt.get('best_val_sisdr', float('nan')):.2f} dB) on {DEVICE}")

    # Validation-split speakers only (never seen as training targets)
    target_files = tr.collect_audio_files(tr.TARGET_AUDIO_DIR)
    shuffled = target_files.copy()
    random.Random(tr.SPLIT_SEED).shuffle(shuffled)
    val_files = shuffled[:max(1, int(len(shuffled) * tr.VAL_FRACTION))]
    files_by_spk = defaultdict(list)
    for f in val_files:
        files_by_spk[speaker_id(f)].append(f)
    spk_ids = sorted(files_by_spk)
    print(f"{len(spk_ids)} validation speakers, {len(val_files)} clips")

    mic_positions = get_mic_positions()
    window = torch.hann_window(tr.WIN_LENGTH, device=DEVICE)
    results = {}

    for cond_name, n_extra in CONDITIONS.items():
        print(f"\n=== {cond_name} ===")
        per_sep = {}
        for sep in SWEEP_DEG:
            sisdrs, sirs, mix_sisdrs = [], [], []
            for t in range(N_TRIALS):
                rng = random.Random(hash((cond_name, sep, t)) % (2 ** 32))
                signals, azimuths, distances = build_scene(
                    files_by_spk, spk_ids, sep, n_extra, rng)
                premix = simulate(signals, azimuths, distances, mic_positions)

                mixture = premix.sum(axis=0)
                scale = float(mixture.std()) or 1e-8
                mixture = (mixture / scale).astype(np.float32)
                target_ref = (premix[0, tr.REFERENCE_MIC] / scale).astype(np.float32)
                inter_ref = (premix[1, tr.REFERENCE_MIC] / scale).astype(np.float32)

                est = run_model(model, mixture, azimuths[0], mic_positions, window)

                sisdrs.append(tr.si_sdr_db(torch.tensor(est).unsqueeze(0),
                                           torch.tensor(target_ref).unsqueeze(0)).item())
                mix_sisdrs.append(tr.si_sdr_db(
                    torch.tensor(mixture[tr.REFERENCE_MIC]).unsqueeze(0),
                    torch.tensor(target_ref).unsqueeze(0)).item())
                sirs.append(projection_sir_db(est, target_ref, inter_ref))

            per_sep[sep] = {
                "sisdr_median": float(np.median(sisdrs)),
                "sisdr_mean": float(np.mean(sisdrs)),
                "sisdr_p25": float(np.percentile(sisdrs, 25)),
                "sisdr_p75": float(np.percentile(sisdrs, 75)),
                "sir_out_median": float(np.median(sirs)),
                "mixture_sisdr_median": float(np.median(mix_sisdrs)),
                "n": len(sisdrs),
            }
            r = per_sep[sep]
            print(f"  sep {sep:5.1f} deg | SI-SDR {r['sisdr_median']:6.2f} dB "
                  f"(p25 {r['sisdr_p25']:6.2f}, p75 {r['sisdr_p75']:6.2f}) | "
                  f"SIR_out {r['sir_out_median']:6.2f} dB | "
                  f"mixture {r['mixture_sisdr_median']:6.2f} dB")

        seps = SWEEP_DEG
        med_sisdr = [per_sep[s]["sisdr_median"] for s in seps]
        med_sir = [per_sep[s]["sir_out_median"] for s in seps]
        thr = derive_thresholds(seps, med_sisdr, med_sir)
        results[cond_name] = {"per_sep": {str(k): v for k, v in per_sep.items()},
                              "thresholds": thr}
        print(f"  -> plateau {thr['plateau_sisdr_db']:.2f} dB | "
              f"safe (within {PLATEAU_DROP_DB:g} dB of plateau): {thr['safe_deg_plateau']} deg | "
              f"safe (>= {ASR_SISDR_DB:g} dB for ASR): {thr['safe_deg_asr']} deg")

    # Recommended global threshold: the strictest ASR-safe angle over conditions
    asr_angles = [r["thresholds"]["safe_deg_asr"] for r in results.values()
                  if r["thresholds"]["safe_deg_asr"] is not None]
    plateau_angles = [r["thresholds"]["safe_deg_plateau"] for r in results.values()
                      if r["thresholds"]["safe_deg_plateau"] is not None]
    recommended = max(asr_angles) if asr_angles else (max(plateau_angles) if plateau_angles else None)
    results["recommended_safe_separation_deg"] = recommended
    print(f"\nRECOMMENDED SAFE_SEPARATION_DEG = {recommended}")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- report ----
    lines = [
        "Angular safety analysis -- direction-conditioned extraction",
        f"checkpoint: {ckpt_path} (epoch {ckpt['epoch']})",
        f"{N_TRIALS} trials per separation angle, anechoic, all sources at 0 dB SIR,",
        f"{tr.SEGMENT_SECONDS:.0f} s segments, 4x4 array.",
        "",
        "SI-SDR  = extraction quality vs. the target at the reference mic.",
        "SIR_out = target energy vs. NEAR-INTERFERER energy in the output",
        "          (how much of the neighbour leaks in).",
        "",
    ]
    for cond_name, r in results.items():
        if not isinstance(r, dict) or "per_sep" not in r:
            continue
        lines += [f"--- {cond_name} ---",
                  f"{'sep(deg)':>9} | {'SI-SDR':>8} | {'p25':>7} | {'p75':>7} | "
                  f"{'SIR_out':>8} | {'mixture':>8}",
                  "-" * 66]
        for s in SWEEP_DEG:
            v = r["per_sep"][str(s)]
            lines.append(f"{s:9.1f} | {v['sisdr_median']:8.2f} | {v['sisdr_p25']:7.2f} | "
                         f"{v['sisdr_p75']:7.2f} | {v['sir_out_median']:8.2f} | "
                         f"{v['mixture_sisdr_median']:8.2f}")
        t = r["thresholds"]
        lines += ["",
                  f"plateau SI-SDR (sep >= {PLATEAU_FROM_DEG} deg): {t['plateau_sisdr_db']:.2f} dB",
                  f"safe separation (within {PLATEAU_DROP_DB:g} dB of plateau): "
                  f"{t['safe_deg_plateau']} deg",
                  f"safe separation (SI-SDR >= {ASR_SISDR_DB:g} dB, ASR-reliable): "
                  f"{t['safe_deg_asr']} deg", ""]
    lines += [f"RECOMMENDED SAFE_SEPARATION_DEG = {recommended}",
              "",
              "Below this separation the single-model path degrades; "
              "fallback_separation.py takes over there."]
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = ["#2a78d6", "#d1662a"]
    for (cond_name, r), c in zip(
            [(k, v) for k, v in results.items() if isinstance(v, dict) and "per_sep" in v],
            colors):
        seps = SWEEP_DEG
        med = [r["per_sep"][str(s)]["sisdr_median"] for s in seps]
        p25 = [r["per_sep"][str(s)]["sisdr_p25"] for s in seps]
        p75 = [r["per_sep"][str(s)]["sisdr_p75"] for s in seps]
        sir = [r["per_sep"][str(s)]["sir_out_median"] for s in seps]
        axes[0].plot(seps, med, "o-", color=c, label=cond_name, lw=2)
        axes[0].fill_between(seps, p25, p75, color=c, alpha=0.15)
        axes[1].plot(seps, sir, "o-", color=c, label=cond_name, lw=2)

    axes[0].axhline(ASR_SISDR_DB, color="#9a9992", ls="--", lw=1)
    axes[0].annotate(f"{ASR_SISDR_DB:g} dB (ASR-reliable)", (SWEEP_DEG[-1], ASR_SISDR_DB),
                     ha="right", va="bottom", fontsize=8, color="#52514e")
    if recommended:
        for ax in axes:
            ax.axvline(recommended, color="#2f7d4f", ls=":", lw=1.6)
        axes[0].annotate(f"safe >= {recommended}°", (recommended, axes[0].get_ylim()[0]),
                         xytext=(4, 6), textcoords="offset points",
                         fontsize=8, color="#2f7d4f")
    axes[0].set_ylabel("SI-SDR (dB)")
    axes[1].set_ylabel("SIR_out: target vs. near interferer (dB)")
    for ax in axes:
        ax.set_xlabel("angular separation to nearest speaker (deg)")
        ax.set_xscale("log")
        ax.set_xticks(SWEEP_DEG)
        ax.set_xticklabels([str(s) for s in SWEEP_DEG], fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("How close can a neighbouring speaker get before extraction degrades?")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "curve.png"), dpi=140)
    print(f"\nSaved {OUTPUT_DIR}/report.txt, results.json, curve.png")


if __name__ == "__main__":
    main()
