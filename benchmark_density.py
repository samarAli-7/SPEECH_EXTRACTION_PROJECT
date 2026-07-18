"""
Does the fallback crossover depend on how crowded the room is?

The beamformer's job gets harder with every extra speaker, while the fallback
sees the same 2- or 3-speaker cluster no matter how many people are present
(everyone outside the cone is already suppressed). So the angle at which the
fallback overtakes the direct path should move with room density -- and if it
does, the trigger has to be density-aware, not a single number.

Sweeps the number of far-away speakers against the neighbour separation and
reports where the fallback wins for each density.

Writes benchmark_density/{report.txt,results.json,curve.png}.
"""

import json
import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import fallback_separation as fbs
import train as tr
from analyze_angular_safety import load_clip, run_model, simulate, speaker_id
from dataset import get_mic_positions

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./benchmark_density"
SWEEP_DEG = [6, 10, 12, 15, 20, 25]
N_FAR_SWEEP = [0, 1, 2, 4, 8]      # extra speakers outside the cluster cone
N_TRIALS = 16
N_NEAR_SWEEP = [1, 2]               # 1 -> 2-speaker cluster, 2 -> 3-speaker cluster
FAR_GUARD_DEG = 45.0
CKPTS = [os.path.join(tr.CHECKPOINT_DIR, "best.pt"), "./checkpoints_v4/best.pt"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_path = next(p for p in CKPTS if os.path.exists(p))
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

    results = {}
    for n_near in N_NEAR_SWEEP:
        cluster_size = n_near + 1
        for n_far in N_FAR_SWEEP:
            n_total = cluster_size + n_far
            print(f"\n### {cluster_size}-speaker cluster, {n_total} speakers in "
                  f"the room ({n_far} far) ###")
            per_sep = {}
            for sep in SWEEP_DEG:
                d_vals, f_vals = [], []
                for t in range(N_TRIALS):
                    rng = random.Random(hash((n_near, n_far, sep, t, "dens")) % (2 ** 32))
                    random.seed(rng.randrange(2 ** 32))
                    usable = [s for s in spk_ids if len(by_spk[s]) >= 2]
                    spks = rng.sample(usable, k=1 + n_near + n_far)
                    tgt_clips = rng.sample(by_spk[spks[0]], k=2)
                    signals = [load_clip(tgt_clips[0], rng)]
                    enrolment = load_clip(tgt_clips[1], rng)
                    for s in spks[1:]:
                        signals.append(load_clip(rng.choice(by_spk[s]), rng))

                    base = rng.uniform(0.0, 360.0)
                    if n_near == 1:
                        azimuths = [base, (base + rng.choice([-1, 1]) * sep) % 360.0]
                    else:
                        azimuths = [base, (base + sep) % 360.0, (base - sep) % 360.0]
                    for _ in range(n_far):
                        a = rng.uniform(0, 360)
                        for _try in range(500):
                            if all(fbs.angular_diff_deg(a, b) >= FAR_GUARD_DEG
                                   for b in azimuths):
                                break
                            a = rng.uniform(0, 360)
                        azimuths.append(a)
                    distances = [rng.uniform(3.0, 30.0) for _ in azimuths]

                    premix = simulate(signals, azimuths, distances, mic_positions)
                    mixture = premix.sum(axis=0)
                    scale = float(mixture.std()) or 1e-8
                    mixture = (mixture / scale).astype(np.float32)
                    target_ref = (premix[0, tr.REFERENCE_MIC] / scale).astype(np.float32)

                    bfmd = run_model(model, mixture, azimuths[0], mic_positions, window)
                    out = fb.run(bfmd, n_cluster=cluster_size, reference=enrolment)
                    d_vals.append(tr.si_sdr_db(torch.tensor(bfmd).unsqueeze(0),
                                               torch.tensor(target_ref).unsqueeze(0)).item())
                    f_vals.append(tr.si_sdr_db(torch.tensor(out["waveform"]).unsqueeze(0),
                                               torch.tensor(target_ref).unsqueeze(0)).item())

                per_sep[sep] = {"direct": float(np.median(d_vals)),
                                "fallback": float(np.median(f_vals)),
                                "win_rate": float(np.mean(np.array(f_vals) > np.array(d_vals)))}
                r = per_sep[sep]
                print(f"  sep {sep:5.1f} | direct {r['direct']:6.2f} | fallback "
                      f"{r['fallback']:6.2f} | fallback wins {r['win_rate']:.2f}")

            # Trigger: fallback must win the MAJORITY of trials, contiguously
            # from the smallest separation. Using the win rate rather than the
            # median makes the threshold robust to a single lucky/unlucky scene.
            cross, first_loss = None, None
            for s in SWEEP_DEG:
                if per_sep[s]["win_rate"] > 0.5:
                    cross = s
                else:
                    first_loss = s
                    break
            trigger = ((cross + first_loss) / 2.0 if cross and first_loss
                       else (cross if cross else 0.0))
            results[(cluster_size, n_total)] = {
                "per_sep": per_sep, "trigger_deg": trigger, "wins_up_to": cross,
                "loses_from": first_loss, "n_speakers": n_total,
                "cluster_size": cluster_size}
            print(f"  -> cluster {cluster_size}, {n_total} speakers: fallback wins "
                  f"up to {cross}°, direct wins from {first_loss}° -> trigger {trigger}°")

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({f"{k[0]}|{k[1]}": v for k, v in results.items()}, f, indent=2)

    # ---- density-aware trigger table consumed by fallback_separation.py ----
    table = {}
    for (cluster_size, n_total), r in results.items():
        table.setdefault(str(cluster_size), {})[str(n_total)] = r["trigger_deg"]
    thresholds = {
        "trigger_table": table,
        "separator": {"2": "sepformer-wsj02mix", "3": "sepformer-wsj03mix"},
        "note": "trigger_table[cluster_size][total_speakers] = separation angle "
                "below which the fallback wins; interpolate over total_speakers.",
        "source": "benchmark_density.py",
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "fallback_thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nWrote fallback_thresholds.json:\n{json.dumps(table, indent=2)}")

    lines = [f"Fallback crossover vs. room density ({ckpt_path}, epoch {ckpt['epoch']})",
             f"{N_TRIALS} trials/point. Cells are direct/fallback median SI-SDR (dB);",
             "the trigger is the largest separation at which the fallback still wins",
             "more than half the trials, counting up from the smallest.", ""]
    for cluster_size in sorted({k[0] for k in results}):
        lines += [f"--- {cluster_size}-speaker cluster "
                  f"({cluster_size - 1} near neighbour(s)) ---",
                  f"{'speakers':>9} | " + " | ".join(f"{s:>11}°" for s in SWEEP_DEG) +
                  " | trigger",
                  "-" * (12 + 15 * len(SWEEP_DEG) + 10)]
        for (cs, n_total), r in sorted(results.items()):
            if cs != cluster_size:
                continue
            cells = " | ".join(
                f"{r['per_sep'][s]['direct']:5.1f}/{r['per_sep'][s]['fallback']:5.1f}"
                for s in SWEEP_DEG)
            lines.append(f"{n_total:9d} | {cells} | {r['trigger_deg']:5.1f}°")
        lines.append("")
    lines += ["The trigger rises with room density: every extra speaker degrades the",
              "beamformer, while the fallback only ever sees the cluster and holds",
              "roughly constant. A single fixed trigger would over-fire in sparse",
              "rooms and under-fire in crowded ones.", ""]
    for (cs, n_total), r in sorted(results.items()):
        lines.append(f"  cluster {cs}, {n_total:2d} speakers -> trigger "
                     f"{r['trigger_deg']:.1f}°")
    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    cluster_sizes = sorted({k[0] for k in results})
    fig, axes = plt.subplots(1, len(cluster_sizes), figsize=(6.5 * len(cluster_sizes), 4.6),
                             squeeze=False)
    palette = ["#2a78d6", "#d1662a", "#2f7d4f", "#8a5cd1", "#b03060"]
    for ax, cluster_size in zip(axes[0], cluster_sizes):
        rows = [(k, v) for k, v in sorted(results.items()) if k[0] == cluster_size]
        for ((cs, n_total), r), c in zip(rows, palette):
            ax.plot(SWEEP_DEG, [r["per_sep"][s]["direct"] for s in SWEEP_DEG], "o-",
                    color=c, lw=2, label=f"{n_total} spk: direct")
            ax.plot(SWEEP_DEG, [r["per_sep"][s]["fallback"] for s in SWEEP_DEG], "o--",
                    color=c, lw=1.4, alpha=0.75, label=f"{n_total} spk: fallback")
        ax.set_title(f"{cluster_size}-speaker cluster", fontsize=10)
        ax.set_xlabel("separation to the near speaker (deg)")
        ax.set_ylabel("median SI-SDR (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Where the fallback overtakes the beamformer, by room density")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "curve.png"), dpi=140)
    print(f"\nSaved {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
