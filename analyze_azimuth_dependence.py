"""
Does the safe angle depend on WHERE the target sits relative to the array grid?

The 4x4 grid is not rotationally symmetric: along a row/column the aperture is
0.30 m, along the diagonal it is 0.42 m, so angular resolution should vary with
azimuth. This sweeps the target azimuth over a quarter turn (the grid repeats
every 90 deg) at a few fixed neighbour separations and reports extraction
quality per azimuth.

Writes angular_safety/azimuth_dependence.{txt,png}.
"""

import os
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import train as tr
from analyze_angular_safety import load_clip, run_model, simulate, speaker_id
from dataset import get_mic_positions

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT_DIR = "./angular_safety"
AZIMUTHS = list(range(0, 91, 7))     # grid symmetry repeats every 90 deg
SEPARATIONS = [10.0, 15.0, 20.0]
N_TRIALS = 10
N_FAR = 4
FAR_GUARD_DEG = 45.0
CKPTS = [os.path.join(tr.CHECKPOINT_DIR, "best.pt"), "./checkpoints_v4/best.pt"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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

    table = {}
    for sep in SEPARATIONS:
        row = []
        for az0 in AZIMUTHS:
            vals = []
            for t in range(N_TRIALS):
                rng = random.Random(hash((sep, az0, t, "azdep")) % (2 ** 32))
                random.seed(rng.randrange(2 ** 32))
                spks = rng.sample(spk_ids, k=2 + N_FAR)
                signals = [load_clip(rng.choice(by_spk[s]), rng) for s in spks]

                azimuths = [float(az0), (az0 + sep) % 360.0]
                for _ in range(N_FAR):
                    a = rng.uniform(0, 360)
                    for _try in range(500):
                        if all(min(abs(a - b) % 360, 360 - abs(a - b) % 360) >= FAR_GUARD_DEG
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
                est = run_model(model, mixture, azimuths[0], mic_positions, window)
                vals.append(tr.si_sdr_db(torch.tensor(est).unsqueeze(0),
                                         torch.tensor(target_ref).unsqueeze(0)).item())
            row.append(float(np.median(vals)))
            print(f"  sep {sep:4.1f} | az {az0:3d} deg | SI-SDR {row[-1]:6.2f} dB")
        table[sep] = row

    lines = [f"Azimuth dependence of extraction quality ({ckpt_path})",
             f"{N_TRIALS} trials per point, median SI-SDR (dB).",
             "0 deg = along a grid row (aperture 0.30 m); "
             "45 deg = along the diagonal (aperture 0.42 m).", "",
             f"{'az':>5} | " + " | ".join(f"sep {s:4.1f}" for s in SEPARATIONS),
             "-" * 40]
    for i, az in enumerate(AZIMUTHS):
        lines.append(f"{az:5d} | " + " | ".join(f"{table[s][i]:8.2f}" for s in SEPARATIONS))
    lines.append("")
    for s in SEPARATIONS:
        v = table[s]
        best_i, worst_i = int(np.argmax(v)), int(np.argmin(v))
        lines.append(f"sep {s:4.1f}: best {v[best_i]:.2f} dB at {AZIMUTHS[best_i]} deg, "
                     f"worst {v[worst_i]:.2f} dB at {AZIMUTHS[worst_i]} deg "
                     f"(spread {v[best_i] - v[worst_i]:.2f} dB)")
    with open(os.path.join(OUT_DIR, "azimuth_dependence.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-len(SEPARATIONS):]))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for s, c in zip(SEPARATIONS, ["#2a78d6", "#d1662a", "#2f7d4f"]):
        ax.plot(AZIMUTHS, table[s], "o-", color=c, label=f"{s:.0f}° separation")
    ax.set_xlabel("target azimuth relative to the grid (deg)")
    ax.set_ylabel("median SI-SDR (dB)")
    ax.set_title("Extraction quality vs. target azimuth (4×4 grid, 90° symmetry)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "azimuth_dependence.png"), dpi=140)
    print(f"Saved {OUT_DIR}/azimuth_dependence.*")


if __name__ == "__main__":
    main()
