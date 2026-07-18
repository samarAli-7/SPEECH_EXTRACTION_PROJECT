"""
Generate listening examples from the CURRENT data pipeline (v3: anechoic,
5-second segments) and save them as wavs under ./listen_examples/example_XX/:

  - target.wav     : anechoic direct-path target speech at the reference mic
                     (the training ground truth)
  - mixture.wav    : the noisy mixture at the reference mic (what the array hears)
  - extracted.wav  : the model's extraction steered at the true target azimuth,
                     using the newest available checkpoint (checkpoints_v3 if it
                     exists, otherwise the v2 weights as a preview) -- only
                     written if a checkpoint is found
  - report.txt     : target file, azimuth, measured SIR, SI-SDR numbers

All files in one example share a common loudness scale so relative levels are
preserved. Runs on CPU. Examples are deterministic: the same index always
produces the same audio, so you can compare across checkpoints later.
"""

import os
import random

import torch
import torch.nn.functional as F
import torchaudio

import train as tr
from dataset import compute_steering_vector, FS

# ---------------- settings ----------------
DEVICE = "cpu"
OUTPUT_DIR = "./listen_examples"
N_EXAMPLES = 5
SEED_OFFSET = 20_000   # distinct from val loader and test_directions examples
# checkpoints to try, in order of preference
CHECKPOINT_CANDIDATES = [
    os.path.join(tr.CHECKPOINT_DIR, "best.pt"),      # current run
    os.path.join(tr.CHECKPOINT_DIR, "last.pt"),
    "./checkpoints_v4/best.pt", "./checkpoints_v4/last.pt",
    "./checkpoints_v3/best.pt", "./checkpoints_v3/last.pt",
]
# ------------------------------------------


def load_model_if_available():
    for path in CHECKPOINT_CANDIDATES:
        if os.path.exists(path):
            model = tr.SteeringConditionedMVDRUNet(
                tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
                tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING,
            ).to(DEVICE)
            ckpt = torch.load(path, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            model.eval()
            print(f"Extraction checkpoint: {path} (epoch {ckpt['epoch']})")
            if "checkpoints_v2" in path:
                print("  NOTE: these are v2 weights (trained on the old reverberant "
                      "2s task) -- extraction is a rough preview until v3 trains.")
            return model, path
    print("No checkpoint found -- saving target/mixture only.")
    return None, None


def save_wav(path, wav, scale):
    data = (wav * scale).clamp(-1.0, 1.0).unsqueeze(0)
    torchaudio.save(path, data, FS, encoding="PCM_S", bits_per_sample=16)


def main():
    model, ckpt_path = load_model_if_available()
    window = torch.hann_window(tr.WIN_LENGTH, device=DEVICE)

    # Same val split as training
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

    for i in range(N_EXAMPLES):
        out_dir = os.path.join(OUTPUT_DIR, f"example_{i:02d}")
        os.makedirs(out_dir, exist_ok=True)

        # Reproduce one deterministic example, keeping the azimuth
        # (same seeding formula as SimulatedSEDataset.__getitem__)
        idx = SEED_OFFSET + i
        random.seed((1000003 * (idx + 1) + 12345) % (2 ** 32))
        target_path = random.choice(ds.target_files)
        interferer_paths = ds._pick_interferers(target_path, k=10)
        from dataset import generate_example
        example = generate_example(
            target_path, interferer_paths,
            max_signal_seconds=ds.segment_seconds + 0.5,
        )
        mixture = example["mixture"]
        target_mics = example["target_reverb"]
        true_az = example["target_azimuth_deg"]

        total_len = mixture.shape[-1]
        if total_len > ds.segment_len:
            ref_energy = target_mics[ds.reference_mic].pow(2)
            csum = torch.cumsum(ref_energy, dim=0)
            windows = csum[ds.segment_len - 1:] - F.pad(csum[:-ds.segment_len], (1, 0))
            start = int(torch.argmax(windows))
            mixture = mixture[:, start:start + ds.segment_len]
            target_mics = target_mics[:, start:start + ds.segment_len]
        elif total_len < ds.segment_len:
            pad = ds.segment_len - total_len
            mixture = F.pad(mixture, (0, pad))
            target_mics = F.pad(target_mics, (0, pad))

        target_ref = target_mics[ds.reference_mic]
        scale_norm = mixture.std().clamp_min(1e-8)
        mixture = mixture / scale_norm
        target_ref = target_ref / scale_norm
        mix_ref = mixture[ds.reference_mic]

        mix_sisdr = tr.si_sdr_db(mix_ref.unsqueeze(0), target_ref.unsqueeze(0)).item()

        # Optional extraction at the true azimuth
        extracted, ext_sisdr = None, None
        if model is not None:
            steering_np = compute_steering_vector(true_az, ds.mic_positions, FS, tr.N_FFT)
            steering = torch.tensor(steering_np, dtype=torch.complex64,
                                    device=DEVICE).unsqueeze(0)
            x_stft = tr.compute_stft(mixture.unsqueeze(0).to(DEVICE),
                                     tr.N_FFT, tr.HOP_LENGTH, tr.WIN_LENGTH, window)
            with torch.no_grad():
                s_hat = model(x_stft, steering)
            extracted = torch.istft(
                s_hat, n_fft=tr.N_FFT, hop_length=tr.HOP_LENGTH,
                win_length=tr.WIN_LENGTH, window=window, center=True,
                length=mixture.shape[-1],
            ).squeeze(0).cpu()
            ext_sisdr = tr.si_sdr_db(extracted.unsqueeze(0), target_ref.unsqueeze(0)).item()

        # Common loudness scale within the example
        signals = [target_ref, mix_ref] + ([extracted] if extracted is not None else [])
        peak = max(s.abs().max().item() for s in signals)
        scale = 0.9 / max(peak, 1e-8)

        save_wav(os.path.join(out_dir, "target.wav"), target_ref, scale)
        save_wav(os.path.join(out_dir, "mixture.wav"), mix_ref, scale)
        if extracted is not None:
            save_wav(os.path.join(out_dir, "extracted.wav"), extracted, scale)

        lines = [
            f"target file    : {target_path}",
            f"true azimuth   : {true_az:.1f} deg",
            f"segment        : {tr.SEGMENT_SECONDS:.1f} s, anechoic (max_order=0)",
            f"mixture SI-SDR : {mix_sisdr:6.2f} dB (vs target, at reference mic)",
        ]
        if ext_sisdr is not None:
            lines.append(f"extract SI-SDR : {ext_sisdr:6.2f} dB (checkpoint: {ckpt_path})")
        with open(os.path.join(out_dir, "report.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

        msg = f"example_{i:02d}: az {true_az:6.1f} deg | mixture {mix_sisdr:6.2f} dB"
        if ext_sisdr is not None:
            msg += f" | extracted {ext_sisdr:6.2f} dB"
        print(msg + f" | saved to {out_dir}")

    print(f"\nDone. Listen under {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
