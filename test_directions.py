"""
Directional extraction listening test.

For each of a few deterministic validation examples this script saves, under
./test_outputs/example_XX/:

  - target.wav            : anechoic direct-path target speech at the reference
                            mic (the training ground truth)
  - mixture.wav           : the noisy multichannel mixture at the reference mic
  - extracted_az_XXX.wav  : the model's output when steered to azimuth XXX,
                            for a full sweep of directions (every AZ_STEP_DEG)
  - extracted_true_az_XXX.wav : the model's output steered at the TRUE target
                            azimuth
  - report.txt            : SI-SDR (dB) of every steered output vs. the target

If the model is direction-selective, only azimuths near the true one should
sound like (and score like) the target; all other directions should be
suppressed / garbled with low or negative SI-SDR.

Runs on CPU by default so it never competes with an active training run for
GPU memory. Uses the same deterministic per-index seeding as validation, the
same crop/normalization as training, and loads the checkpoint configured below
(from train.py's CHECKPOINT_DIR).
"""

import os
import random

import torch
import torch.nn.functional as F
import torchaudio

import train as tr
from dataset import generate_example, compute_steering_vector, FS

# ---------------- settings ----------------
DEVICE = "cpu"            # keep off the GPUs while training is running
# Newest available checkpoint: the current run's dir first, then older runs.
_CANDIDATES = [
    os.path.join(tr.CHECKPOINT_DIR, "best.pt"),
    os.path.join(tr.CHECKPOINT_DIR, "last.pt"),
    "./checkpoints_v4/best.pt", "./checkpoints_v4/last.pt",
    "./checkpoints_v3/best.pt", "./checkpoints_v3/last.pt",
]
CHECKPOINT = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
OUTPUT_DIR = "./test_outputs"
N_EXAMPLES = 5            # how many validation examples to render
AZ_STEP_DEG = 15          # sweep resolution: 360/15 = 24 directions
SEED_OFFSET = 10_000      # keeps these examples distinct from the val loader's
# ------------------------------------------


def build_model():
    model = tr.SteeringConditionedMVDRUNet(
        tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
        tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING,
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {CHECKPOINT} (epoch {ckpt['epoch']}, "
          f"best_val_sisdr={ckpt.get('best_val_sisdr', float('nan')):.2f} dB)")
    return model


def make_example(ds, idx):
    """
    Deterministically generate one example exactly the way training/validation
    does (same seeding formula, same energy crop, same level normalization),
    but also return the true target azimuth for the direction sweep.
    """
    random.seed((1000003 * (idx + 1) + 12345) % (2 ** 32))

    target_path = random.choice(ds.target_files)
    interferer_paths = ds._pick_interferers(target_path, k=10)
    example = generate_example(
        target_path, interferer_paths,
        max_signal_seconds=ds.segment_seconds + 0.5,
    )
    mixture = example["mixture"]              # (N_mics, T)
    target_reverb = example["target_reverb"]  # (N_mics, T)
    azimuth_deg = example["target_azimuth_deg"]

    # Fixed-length crop centred on the most energetic target window (as in training)
    total_len = mixture.shape[-1]
    if total_len > ds.segment_len:
        ref_energy = target_reverb[ds.reference_mic].pow(2)
        csum = torch.cumsum(ref_energy, dim=0)
        windows = csum[ds.segment_len - 1:] - F.pad(csum[:-ds.segment_len], (1, 0))
        start = int(torch.argmax(windows))
        mixture = mixture[:, start:start + ds.segment_len]
        target_reverb = target_reverb[:, start:start + ds.segment_len]
    elif total_len < ds.segment_len:
        pad = ds.segment_len - total_len
        mixture = F.pad(mixture, (0, pad))
        target_reverb = F.pad(target_reverb, (0, pad))

    target_ref = target_reverb[ds.reference_mic]

    # Same per-example level normalization as training
    scale = mixture.std().clamp_min(1e-8)
    mixture = mixture / scale
    target_ref = target_ref / scale

    return mixture, target_ref, azimuth_deg, target_path


def extract(model, mixture, azimuth_deg, mic_positions, window):
    """Run the model steered at `azimuth_deg`; returns the extracted waveform."""
    steering_np = compute_steering_vector(azimuth_deg, mic_positions, FS, tr.N_FFT)
    steering = torch.tensor(steering_np, dtype=torch.complex64, device=DEVICE).unsqueeze(0)

    x = mixture.unsqueeze(0).to(DEVICE)  # (1, N_mics, T)
    x_stft = tr.compute_stft(x, tr.N_FFT, tr.HOP_LENGTH, tr.WIN_LENGTH, window)
    with torch.no_grad():
        s_hat = model(x_stft, steering)  # (1, F, T')
    wav = torch.istft(
        s_hat, n_fft=tr.N_FFT, hop_length=tr.HOP_LENGTH, win_length=tr.WIN_LENGTH,
        window=window, center=True, length=x.shape[-1],
    )
    return wav.squeeze(0).cpu()


def save_wav(path, wav, scale):
    """Save mono 16-bit PCM at a common scale so relative levels are audible."""
    data = (wav * scale).clamp(-1.0, 1.0).unsqueeze(0)
    torchaudio.save(path, data, FS, encoding="PCM_S", bits_per_sample=16)


def main():
    model = build_model()
    window = torch.hann_window(tr.WIN_LENGTH, device=DEVICE)

    # Same val split as training (SPLIT_SEED-shuffled file list, first VAL_FRACTION)
    target_files = tr.collect_audio_files(tr.TARGET_AUDIO_DIR)
    noise_files = tr.collect_audio_files(tr.NOISE_AUDIO_DIR)
    shuffled = target_files.copy()
    random.Random(tr.SPLIT_SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * tr.VAL_FRACTION))
    val_targets = shuffled[:n_val]

    ds = tr.SimulatedSEDataset(
        val_targets, noise_files, tr.SEGMENT_SECONDS, N_EXAMPLES,
        tr.REFERENCE_MIC, tr.N_FFT, fixed_length=True, deterministic=True,
    )

    azimuths = list(range(0, 360, AZ_STEP_DEG))

    for i in range(N_EXAMPLES):
        out_dir = os.path.join(OUTPUT_DIR, f"example_{i:02d}")
        os.makedirs(out_dir, exist_ok=True)

        mixture, target_ref, true_az, target_path = make_example(ds, SEED_OFFSET + i)
        print(f"\n=== example_{i:02d}: true azimuth {true_az:.1f} deg | {os.path.basename(target_path)} ===")

        # Extract at every sweep direction plus the exact true azimuth
        results = []  # (azimuth, is_true, wav, sisdr_db)
        for az in azimuths:
            wav = extract(model, mixture, float(az), ds.mic_positions, window)
            sisdr = tr.si_sdr_db(wav.unsqueeze(0), target_ref.unsqueeze(0)).item()
            results.append((float(az), False, wav, sisdr))
        wav_true = extract(model, mixture, true_az, ds.mic_positions, window)
        sisdr_true = tr.si_sdr_db(wav_true.unsqueeze(0), target_ref.unsqueeze(0)).item()
        results.append((true_az, True, wav_true, sisdr_true))

        mix_ref = mixture[ds.reference_mic]
        mix_sisdr = tr.si_sdr_db(mix_ref.unsqueeze(0), target_ref.unsqueeze(0)).item()

        # One common scale across all files of this example -> comparable loudness
        peak = max(t.abs().max().item() for t in
                   [target_ref, mix_ref] + [r[2] for r in results])
        scale = 0.9 / max(peak, 1e-8)

        save_wav(os.path.join(out_dir, "target.wav"), target_ref, scale)
        save_wav(os.path.join(out_dir, "mixture.wav"), mix_ref, scale)

        lines = [
            f"target file      : {target_path}",
            f"true azimuth     : {true_az:.1f} deg",
            f"mixture SI-SDR   : {mix_sisdr:6.2f} dB (vs target, at reference mic)",
            "",
            f"{'azimuth':>10} | {'SI-SDR (dB)':>11} | note",
            "-" * 44,
        ]
        for az, is_true, wav, sisdr in results:
            if is_true:
                fname = f"extracted_true_az_{az:05.1f}.wav"
                note = "TRUE DIRECTION"
            else:
                fname = f"extracted_az_{int(az):03d}.wav"
                sep = min(abs(az - true_az), 360 - abs(az - true_az))
                note = f"{sep:5.1f} deg off"
            save_wav(os.path.join(out_dir, fname), wav, scale)
            lines.append(f"{az:10.1f} | {sisdr:11.2f} | {note}")
            print(f"  az {az:6.1f}  SI-SDR {sisdr:7.2f} dB  {note}")

        with open(os.path.join(out_dir, "report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"  mixture SI-SDR {mix_sisdr:.2f} dB | saved {len(results) + 2} wavs to {out_dir}")

    print(f"\nDone. Listen under {OUTPUT_DIR}/ -- only azimuths near the true "
          f"direction should contain the target speaker.")


if __name__ == "__main__":
    main()
