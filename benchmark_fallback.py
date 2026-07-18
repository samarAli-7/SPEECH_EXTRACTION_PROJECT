"""
Does the close-speaker fallback actually help, and with which parts?

Builds controlled scenes with a neighbour at a known azimuth separation, and
compares, at every separation:

  direct              : the direction-conditioned model alone (baseline)
  <variant>/oracle    : best of the separated streams (upper bound on the
                        separation stage, ignoring identification)
  <variant>/enrol     : the stream ECAPA-TDNN picks using a DIFFERENT utterance
                        of the target speaker as enrolment (what the app does
                        when it has the speaker's clip)
  <variant>/bf-ref    : the stream ECAPA picks using the beamformed signal
                        itself as the reference (no enrolment available)

Variants differ in what is fed to the separator (beamformed output vs. the raw
reference-mic mixture), which separator is used, and how the output is
reconstructed ("direct" = the separated stream as-is; "mask" = the streams only
provide a T-F mask that is applied to the full-band beamformed signal).

The crossover -- the separation angle where the best fallback variant stops
beating `direct` -- is the angle at which the app should switch pipelines, and
is written to benchmark_fallback/results.json.

Runs the 2-speaker-cluster case (one near neighbour) and the 3-speaker-cluster
case (two near neighbours) separately, since they use different separators.
"""

import json
import math
import os
import random
import time
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train as tr
import fallback_separation as fbs
from analyze_angular_safety import (
    _ang_diff, load_clip, projection_sir_db, run_model, simulate, speaker_id,
)
from dataset import FS, get_mic_positions

# ---------------- settings ----------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "./benchmark_fallback"
CKPT_CANDIDATES = [
    os.path.join(tr.CHECKPOINT_DIR, "best.pt"), os.path.join(tr.CHECKPOINT_DIR, "last.pt"),
    "./checkpoints_v4/best.pt", "./checkpoints_v4/last.pt",
]
SWEEP_DEG = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30]
N_TRIALS = 12
N_FAR = 4                       # extra well-separated speakers in the room
FAR_GUARD_DEG = 45.0
DIST_RANGE = (3.0, 30.0)
SEG_LEN = int(round(tr.SEGMENT_SECONDS * FS))

# variant -> (separator dir, input signal, reconstruction)
VARIANTS_2SPK = {
    "wsj02mix/bf/mask":   ("sepformer-wsj02mix", "beamformed", "mask"),
    "wsj02mix/bf/direct": ("sepformer-wsj02mix", "beamformed", "direct"),
    "whamr16k/bf/mask":   ("sepformer-whamr16k", "beamformed", "mask"),
    "wsj02mix/mix/mask":  ("sepformer-wsj02mix", "mixture", "mask"),
}
VARIANTS_3SPK = {
    "wsj03mix/bf/mask":   ("sepformer-wsj03mix", "beamformed", "mask"),
    "wsj03mix/bf/direct": ("sepformer-wsj03mix", "beamformed", "direct"),
    "wsj03mix/mix/mask":  ("sepformer-wsj03mix", "mixture", "mask"),
}
# ------------------------------------------


def build_scene(files_by_spk, spk_ids, sep_deg, n_near, rng):
    """
    Scene with `n_near` interferers inside the cone (at +/- sep_deg) plus N_FAR
    far-away speakers. Returns signals, azimuths, distances, and a separate
    ENROLMENT clip for the target (a different utterance of the same speaker).
    """
    random.seed(rng.randrange(2 ** 32))
    usable = [s for s in spk_ids if len(files_by_spk[s]) >= 2]
    chosen = rng.sample(usable, k=1 + n_near + N_FAR)

    tgt_files = rng.sample(files_by_spk[chosen[0]], k=2)
    signals = [load_clip(tgt_files[0], rng)]
    enrolment = load_clip(tgt_files[1], rng)
    for s in chosen[1:]:
        signals.append(load_clip(rng.choice(files_by_spk[s]), rng))

    az_t = rng.uniform(0.0, 360.0)
    azimuths = [az_t]
    if n_near == 1:
        azimuths.append((az_t + rng.choice([-1.0, 1.0]) * sep_deg) % 360.0)
    else:
        azimuths.append((az_t + sep_deg) % 360.0)
        azimuths.append((az_t - sep_deg) % 360.0)
    for _ in range(N_FAR):
        az = rng.uniform(0.0, 360.0)
        for _try in range(500):
            if all(_ang_diff(az, a) >= FAR_GUARD_DEG for a in azimuths):
                break
            az = rng.uniform(0.0, 360.0)
        azimuths.append(az)

    distances = [rng.uniform(*DIST_RANGE) for _ in azimuths]
    return signals, azimuths, distances, enrolment


def sisdr(est, ref):
    return tr.si_sdr_db(torch.tensor(np.asarray(est, dtype=np.float32)).unsqueeze(0),
                        torch.tensor(np.asarray(ref, dtype=np.float32)).unsqueeze(0)).item()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    ckpt_path = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
    model = tr.SteeringConditionedMVDRUNet(
        tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
        tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Model: {ckpt_path} (epoch {ckpt['epoch']}) on {DEVICE}")

    target_files = tr.collect_audio_files(tr.TARGET_AUDIO_DIR)
    shuffled = target_files.copy()
    random.Random(tr.SPLIT_SEED).shuffle(shuffled)
    val_files = shuffled[:max(1, int(len(shuffled) * tr.VAL_FRACTION))]
    files_by_spk = defaultdict(list)
    for f in val_files:
        files_by_spk[speaker_id(f)].append(f)
    spk_ids = sorted(files_by_spk)

    mic_positions = get_mic_positions()
    window = torch.hann_window(tr.WIN_LENGTH, device=DEVICE)

    # one fallback object per (separator, reconstruction) combination
    fallbacks = {}

    def get_fb(sep_name, recon):
        key = (sep_name, recon)
        if key not in fallbacks:
            fb = fbs.SeparationFallback(device=DEVICE, reconstruction=recon)
            fb.separator_names = {2: sep_name, 3: sep_name}
            fallbacks[key] = fb
        return fallbacks[key]

    results = {}
    for case_name, n_near, variants in [
            ("2-speaker cluster (1 near interferer)", 1, VARIANTS_2SPK),
            ("3-speaker cluster (2 near interferers)", 2, VARIANTS_3SPK)]:
        print(f"\n########## {case_name} ##########")
        n_streams = n_near + 1
        case = {}
        for sep in SWEEP_DEG:
            acc = defaultdict(list)
            for t in range(N_TRIALS):
                rng = random.Random(hash((case_name, sep, t)) % (2 ** 32))
                signals, azimuths, distances, enrolment = build_scene(
                    files_by_spk, spk_ids, sep, n_near, rng)
                premix = simulate(signals, azimuths, distances, mic_positions)

                mixture = premix.sum(axis=0)
                scale = float(mixture.std()) or 1e-8
                mixture = (mixture / scale).astype(np.float32)
                target_ref = (premix[0, tr.REFERENCE_MIC] / scale).astype(np.float32)

                beamformed = run_model(model, mixture, azimuths[0], mic_positions, window)
                acc["direct"].append(sisdr(beamformed, target_ref))

                for vname, (sep_model, source, recon) in variants.items():
                    fb = get_fb(sep_model, recon)
                    inp = beamformed if source == "beamformed" else mixture[tr.REFERENCE_MIC]
                    streams, _ = fb.separate(inp, n_streams)
                    streams_np = streams.numpy()

                    stream_sisdrs = [sisdr(s, target_ref) for s in streams_np]
                    best = int(np.argmax(stream_sisdrs))
                    acc[f"{vname}/oracle"].append(stream_sisdrs[best])

                    stream_embs = fb.embed(streams)
                    for ref_kind, ref_wave in (("enrol", enrolment),
                                               ("bf-ref", beamformed)):
                        ref_emb = fb.embed(ref_wave)[0]
                        sims = [fbs.cosine_similarity(stream_embs[k], ref_emb)
                                for k in range(len(streams_np))]
                        pick = int(np.argmax(sims))
                        acc[f"{vname}/{ref_kind}"].append(stream_sisdrs[pick])
                        acc[f"{vname}/{ref_kind}/acc"].append(1.0 if pick == best else 0.0)

            case[sep] = {k: {"median": float(np.median(v)), "mean": float(np.mean(v))}
                         for k, v in acc.items()}
            summary = " | ".join(
                f"{k.split('/')[0] if k != 'direct' else 'direct'}"
                f"{'' if k == 'direct' else '/' + '/'.join(k.split('/')[1:])}"
                f" {case[sep][k]['median']:.1f}"
                for k in ["direct"] + [f"{v}/enrol" for v in variants])
            print(f"  sep {sep:5.1f} | {summary}")
        results[case_name] = {str(k): v for k, v in case.items()}

    # ---- crossover: where does the best fallback stop beating direct? ----
    crossovers = {}
    for case_name, case in results.items():
        variants = VARIANTS_2SPK if "2-speaker" in case_name else VARIANTS_3SPK
        best_variant, best_score = None, -1e9
        for vname in variants:
            score = np.mean([case[str(s)][f"{vname}/enrol"]["median"]
                             for s in SWEEP_DEG if s <= 12])
            if score > best_score:
                best_variant, best_score = vname, score
        # Largest angle up to which the fallback wins CONTIGUOUSLY from the
        # smallest separation -- a lone win at a large angle is noise, not a
        # reason to keep using the fallback there.
        cross, first_loss = None, None
        for s in SWEEP_DEG:
            d = case[str(s)]["direct"]["median"]
            f = case[str(s)][f"{best_variant}/enrol"]["median"]
            if f > d:
                cross = s
            else:
                first_loss = s
                break
        trigger = ((cross + first_loss) / 2.0 if cross is not None and first_loss is not None
                   else (cross if cross is not None else 0.0))
        crossovers[case_name] = {"best_variant": best_variant,
                                 "fallback_wins_up_to_deg": cross,
                                 "direct_wins_from_deg": first_loss,
                                 "trigger_deg": trigger,
                                 "mean_sisdr_below_12deg": float(best_score)}
        print(f"\n{case_name}: best variant '{best_variant}', fallback beats "
              f"direct up to {cross} deg, direct wins from {first_loss} deg "
              f"-> trigger below {trigger} deg")

    results["crossovers"] = crossovers
    # NOTE: the crossovers here are measured at ONE room density (N_FAR far
    # speakers). The trigger the app actually uses is density-aware and comes
    # from benchmark_density.py, which owns fallback_thresholds.json -- this
    # script deliberately does not write that file.
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- report ----
    lines = [f"Close-speaker fallback benchmark ({ckpt_path}, epoch {ckpt['epoch']})",
             f"{N_TRIALS} trials per point, {N_FAR} far speakers in the room, "
             f"all sources at 0 dB SIR, anechoic, {tr.SEGMENT_SECONDS:.0f} s.",
             "Values are median SI-SDR (dB) vs. the target at the reference mic.", ""]
    for case_name, case in results.items():
        if case_name == "crossovers":
            continue
        variants = VARIANTS_2SPK if "2-speaker" in case_name else VARIANTS_3SPK
        cols = ["direct"]
        for v in variants:
            cols += [f"{v}/oracle", f"{v}/enrol", f"{v}/bf-ref"]
        lines += [f"--- {case_name} ---", ""]
        header = f"{'sep':>5} | " + " | ".join(f"{c[-22:]:>22}" for c in cols)
        lines += [header, "-" * len(header)]
        for s in SWEEP_DEG:
            row = f"{s:5.1f} | " + " | ".join(
                f"{case[str(s)][c]['median']:22.2f}" for c in cols)
            lines.append(row)
        lines += ["", "identification accuracy (fraction of trials that picked the "
                  "best stream):"]
        for v in variants:
            for kind in ("enrol", "bf-ref"):
                accs = [case[str(s)][f"{v}/{kind}/acc"]["mean"] for s in SWEEP_DEG]
                lines.append(f"  {v}/{kind}: " +
                             " ".join(f"{s}deg={a:.2f}" for s, a in zip(SWEEP_DEG, accs)))
        c = crossovers[case_name]
        lines += ["", f"best variant: {c['best_variant']}",
                  f"fallback beats the direct path at separations up to "
                  f"{c['fallback_wins_up_to_deg']} deg", ""]
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    palette = ["#2a78d6", "#d1662a", "#2f7d4f", "#8a5cd1"]
    for ax, (case_name, case) in zip(axes, [(k, v) for k, v in results.items()
                                            if k != "crossovers"]):
        variants = VARIANTS_2SPK if "2-speaker" in case_name else VARIANTS_3SPK
        ax.plot(SWEEP_DEG, [case[str(s)]["direct"]["median"] for s in SWEEP_DEG],
                "o-", color="#0b0b0b", lw=2.4, label="direct (model only)")
        for (vname, _), c in zip(variants.items(), palette):
            ax.plot(SWEEP_DEG, [case[str(s)][f"{vname}/enrol"]["median"] for s in SWEEP_DEG],
                    "o-", color=c, lw=1.8, label=f"{vname} (enrol)")
            ax.plot(SWEEP_DEG, [case[str(s)][f"{vname}/oracle"]["median"] for s in SWEEP_DEG],
                    ":", color=c, lw=1.2, alpha=0.7)
        ax.set_title(case_name, fontsize=10)
        ax.set_xlabel("separation to near speaker (deg)")
        ax.set_ylabel("median SI-SDR (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("Fallback vs. direct extraction (dotted = oracle stream choice)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "curve.png"), dpi=140)
    print(f"\nSaved {OUTPUT_DIR}/ ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
