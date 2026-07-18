"""
Should the fallback be gated by angle alone, or by ECAPA similarity per scene?

A fixed angular trigger is a population-level rule: it fires whenever the
neighbour is close, even in the scenes where the beamformer happened to do
fine. This measures a per-scene alternative that needs no ground truth --
compare the ECAPA similarity of the BEAMFORMER output and of the chosen
SEPARATED stream against the target's enrolment embedding, and report whichever
is closer to the enrolment.

Strategies compared, per separation angle:
  always-direct   : beamformer only (no fallback at all)
  always-fallback : separation + identification every time
  angle-gate      : fallback iff the neighbour is inside the measured trigger
  arbitrate       : fallback iff its stream is closer to the enrolment than the
                    beamformer output is
  oracle          : per scene, the better of the two (upper bound)

Writes benchmark_arbitration/report.txt + results.json.
"""

import json
import os
import random
from collections import defaultdict

import numpy as np
import torch

import fallback_separation as fbs
import train as tr
from analyze_angular_safety import run_model, simulate, speaker_id
from benchmark_fallback import build_scene, sisdr
from dataset import get_mic_positions

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./benchmark_arbitration"
SWEEP_DEG = [4, 6, 8, 10, 12, 15, 20, 25]
N_TRIALS = 16
CKPT_CANDIDATES = [os.path.join(tr.CHECKPOINT_DIR, "best.pt"),
                   "./checkpoints_v4/best.pt"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = next(p for p in CKPT_CANDIDATES if os.path.exists(p))
    model = tr.SteeringConditionedMVDRUNet(
        tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
        tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = tr.collect_audio_files(tr.TARGET_AUDIO_DIR)
    shuffled = files.copy()
    random.Random(tr.SPLIT_SEED).shuffle(shuffled)
    val = shuffled[:max(1, int(len(shuffled) * tr.VAL_FRACTION))]
    by_spk = defaultdict(list)
    for f in val:
        by_spk[speaker_id(f)].append(f)
    spk_ids = sorted(by_spk)

    mic_positions = get_mic_positions()
    window = torch.hann_window(tr.WIN_LENGTH, device=DEVICE)
    fb = fbs.SeparationFallback(device=DEVICE)
    triggers = fbs.trigger_angles()

    results = {}
    for case_name, n_near in [("2-speaker cluster", 1), ("3-speaker cluster", 2)]:
        print(f"\n##### {case_name} #####")
        case = {}
        for sep in SWEEP_DEG:
            rows = []
            for t in range(N_TRIALS):
                rng = random.Random(hash((case_name, sep, t, "arb")) % (2 ** 32))
                signals, azimuths, distances, enrolment = build_scene(
                    by_spk, spk_ids, sep, n_near, rng)
                premix = simulate(signals, azimuths, distances, mic_positions)
                mixture = premix.sum(axis=0)
                scale = float(mixture.std()) or 1e-8
                mixture = (mixture / scale).astype(np.float32)
                target_ref = (premix[0, tr.REFERENCE_MIC] / scale).astype(np.float32)

                beamformed = run_model(model, mixture, azimuths[0], mic_positions, window)
                out = fb.run(beamformed, n_cluster=n_near + 1, reference=enrolment)

                ref_emb = fb.embed(enrolment)[0]
                sim_bf = fbs.cosine_similarity(fb.embed(beamformed)[0], ref_emb)
                sim_fb = out["similarities"][out["chosen"]]

                rows.append({
                    "sisdr_direct": sisdr(beamformed, target_ref),
                    "sisdr_fb": sisdr(out["waveform"], target_ref),
                    "sim_bf": sim_bf,
                    "sim_fb": sim_fb,
                    "gate_fires": sep < triggers[min(n_near + 1, 3)],
                })

            d = np.array([r["sisdr_direct"] for r in rows])
            f_ = np.array([r["sisdr_fb"] for r in rows])
            gate = np.array([r["gate_fires"] for r in rows])
            arb = np.array([r["sim_fb"] > r["sim_bf"] for r in rows])

            strategies = {
                "always-direct": d,
                "always-fallback": f_,
                "angle-gate": np.where(gate, f_, d),
                "arbitrate": np.where(arb, f_, d),
                "oracle": np.maximum(d, f_),
            }
            case[sep] = {k: {"median": float(np.median(v)), "mean": float(np.mean(v))}
                         for k, v in strategies.items()}
            case[sep]["arbitrate_fires"] = float(arb.mean())
            case[sep]["arbitrate_correct"] = float(
                np.mean(arb == (f_ > d)))
            print(f"  sep {sep:5.1f} | direct {np.median(d):6.2f} | fallback "
                  f"{np.median(f_):6.2f} | gate {np.median(strategies['angle-gate']):6.2f} | "
                  f"arbitrate {np.median(strategies['arbitrate']):6.2f} "
                  f"(fires {arb.mean():.2f}, correct {case[sep]['arbitrate_correct']:.2f}) | "
                  f"oracle {np.median(strategies['oracle']):6.2f}")
        results[case_name] = {str(k): v for k, v in case.items()}

    # overall means across the sweep
    lines = ["ECAPA arbitration vs. a fixed angular gate",
             f"checkpoint: {ckpt_path} (epoch {ckpt['epoch']}), {N_TRIALS} trials/point",
             "median SI-SDR (dB) vs. the target at the reference mic", ""]
    summary = {}
    for case_name, case in results.items():
        strat_names = ["always-direct", "always-fallback", "angle-gate", "arbitrate", "oracle"]
        lines += [f"--- {case_name} ---",
                  f"{'sep':>5} | " + " | ".join(f"{s:>15}" for s in strat_names) +
                  " | fires | correct",
                  "-" * 108]
        for s in SWEEP_DEG:
            lines.append(f"{s:5.1f} | " +
                         " | ".join(f"{case[str(s)][n]['median']:15.2f}" for n in strat_names) +
                         f" | {case[str(s)]['arbitrate_fires']:5.2f} | "
                         f"{case[str(s)]['arbitrate_correct']:7.2f}")
        means = {n: float(np.mean([case[str(s)][n]["mean"] for s in SWEEP_DEG]))
                 for n in strat_names}
        summary[case_name] = means
        lines += ["", "mean over the whole sweep: " +
                  ", ".join(f"{n} {v:.2f}" for n, v in means.items()), ""]
    results["summary"] = summary

    best = {c: max(("angle-gate", "arbitrate"), key=lambda n: m[n])
            for c, m in summary.items()}
    lines += ["Recommended gating: " + ", ".join(f"{c}: {b}" for c, b in best.items())]
    results["recommended_gating"] = best

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + lines[-1])
    print(f"Saved {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
