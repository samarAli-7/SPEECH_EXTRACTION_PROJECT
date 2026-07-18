"""
Directional Speaker Extraction -- interactive demo UI.

Run with:
    streamlit run app.py

What it does
------------
- Choose how many speakers are in the room (2-11, the model's training range).
- For each speaker: set a name, a voice (Male/Female -> a LibriSpeech speaker of
  that gender is picked using the official SPEAKERS.TXT metadata), and (x, y)
  coordinates in metres relative to the mic array at the origin.
- Optionally record your own voice for any speaker (browser mic, first 5 s are
  used; the recording widget has a built-in clear button to delete/re-record).
  Speakers without a recording use their LibriSpeech clip.
- A top-down map shows the mic array at the centre and every speaker at its
  coordinates; the selected target is highlighted with the steering direction.
- Pick the target speaker and press Run: the room is simulated (anechoic,
  max_order=0, same pipeline as training), the model is steered at the target's
  azimuth, and you can listen to the mixture, the extracted audio, and the
  clean reference, with SI-SDR metrics.

The mic-array spacing is matched to the loaded checkpoint's training geometry
(0.05 m for v2/v3 checkpoints, 0.10 m for v4+), so the demo stays consistent
with whatever model is available.
"""

import glob
import io
import math
import os
import random
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import streamlit as st
import torch
import torchaudio

import train as tr
from dataset import (
    CENTER, FS, MIN_ANGULAR_SEP_DEG, MIN_SOURCE_DIST, ROOM_SIZE,
    compute_steering_vector, load_and_pad_audio,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKERS_TXT = os.path.join(APP_DIR, "SPEAKERS.TXT")
LIBRI_DIR = tr.TARGET_AUDIO_DIR
SEG_LEN = int(round(tr.SEGMENT_SECONDS * FS))          # 5 s -> 80000 samples
DEVICE = "cpu"                                          # never competes with training GPUs
MIN_SPK, MAX_SPK = 2, 11                                # the model's training range

CKPT_CANDIDATES = [
    "./checkpoints_v5/best.pt", "./checkpoints_v5/last.pt",
    "./checkpoints_v4/best.pt", "./checkpoints_v4/last.pt",
    "./checkpoints_v3/best.pt", "./checkpoints_v3/last.pt",
    "./checkpoints_v2/best.pt", "./checkpoints_v2/last.pt",
]

# Validated reference palette (dataviz): target = categorical blue, others =
# neutral; identity is never color-alone (target also gets a star + label).
COL_TARGET = "#2a78d6"
COL_OTHER = "#9a9992"
COL_INK = "#0b0b0b"
COL_MUTED = "#52514e"
COL_GRID = "#e5e4df"


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
def find_checkpoint():
    for p in CKPT_CANDIDATES:
        if os.path.exists(os.path.join(APP_DIR, p)):
            return os.path.join(APP_DIR, p)
    return None


def mic_spacing_for(ckpt_path):
    """v2/v3 checkpoints were trained on the 0.05 m array; v4 and later on 0.10 m."""
    if ckpt_path and ("checkpoints_v2" in ckpt_path or "checkpoints_v3" in ckpt_path):
        return 0.05
    return 0.10


def build_mic_positions(spacing):
    """4x4 grid centred on the room centre, same formula as dataset.py."""
    pts = []
    off = (4 - 1) / 2.0
    for i in range(4):
        for j in range(4):
            pts.append([CENTER[0] + (i - off) * spacing,
                        CENTER[1] + (j - off) * spacing,
                        CENTER[2]])
    return np.array(pts).T  # (3, 16)


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_model(ckpt_path):
    model = tr.SteeringConditionedMVDRUNet(
        tr.N_MICS, tr.C1, tr.C2, tr.C3, tr.C4, tr.MVDR_LEVELS,
        tr.LEAKY_SLOPE, tr.MVDR_EPS, tr.MVDR_DIAG_LOADING,
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, int(ckpt.get("epoch", 0))


@st.cache_data(show_spinner="Indexing LibriSpeech voices...")
def speaker_pools():
    """{'Male': [(spk_id, [files...]), ...], 'Female': [...]} from SPEAKERS.TXT."""
    gender_of = {}
    with open(SPEAKERS_TXT) as f:
        for line in f:
            if line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2 and parts[0].isdigit():
                gender_of[parts[0]] = parts[1]

    files_by_spk = {}
    for path in glob.glob(os.path.join(LIBRI_DIR, "**", "*.flac"), recursive=True):
        spk = os.path.basename(os.path.dirname(os.path.dirname(path)))
        files_by_spk.setdefault(spk, []).append(path)

    pools = {"Male": [], "Female": []}
    for spk, files in sorted(files_by_spk.items()):
        g = gender_of.get(spk)
        if g == "M":
            pools["Male"].append((spk, sorted(files)))
        elif g == "F":
            pools["Female"].append((spk, sorted(files)))
    return pools


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def energy_crop_or_pad(sig, length=SEG_LEN):
    """Deterministic: keep the most energetic `length` window; wrap-pad if short."""
    sig = np.asarray(sig, dtype=np.float32)
    if len(sig) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(sig) >= length:
        csum = np.cumsum(sig.astype(np.float64) ** 2)
        windows = csum[length - 1:] - np.concatenate(([0.0], csum[:-length]))
        start = int(np.argmax(windows))
        return sig[start:start + length]
    return np.pad(sig, (0, length - len(sig)), mode="wrap")


def load_librispeech_clip(path):
    return energy_crop_or_pad(load_and_pad_audio(path))


def load_recording(uploaded_file):
    """Browser recording (wav bytes) -> mono 16 kHz, exactly 5 s (trim/pad)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(uploaded_file.getvalue())
        tmp = f.name
    try:
        wav, sr = torchaudio.load(tmp)
    finally:
        os.unlink(tmp)
    if sr != FS:
        wav = torchaudio.transforms.Resample(sr, FS)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    sig = wav.squeeze(0).numpy().astype(np.float32)
    if len(sig) >= SEG_LEN:
        sig = sig[:SEG_LEN]
    else:
        sig = np.pad(sig, (0, SEG_LEN - len(sig)))
    return sig


def pick_voice(gender_label, exclude_spks):
    pools = speaker_pools()
    if not pools[gender_label]:
        return None
    candidates = [e for e in pools[gender_label] if e[0] not in exclude_spks]
    if not candidates:
        candidates = pools[gender_label]
    spk, files = random.choice(candidates)
    return spk, random.choice(files)


# ---------------------------------------------------------------------------
# Simulation + inference
# ---------------------------------------------------------------------------
def azimuth_of(x, y):
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def simulate_room(signals, rel_positions, mic_positions):
    """Anechoic sim at user positions; every source equalized to unit mic power.
    Returns premix scaled, shape (n_sources, 16, T)."""
    room = pra.ShoeBox(ROOM_SIZE, fs=FS, max_order=0)
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))
    for sig, (x, y) in zip(signals, rel_positions):
        room.add_source([CENTER[0] + x, CENTER[1] + y, CENTER[2]], signal=sig)
    premix = room.simulate(return_premix=True)
    scaled = np.empty_like(premix)
    for k in range(premix.shape[0]):
        p = float(np.mean(premix[k] ** 2)) + 1e-12
        scaled[k] = premix[k] / math.sqrt(p)
    return scaled


def run_extraction(model, mic_positions, signals, rel_positions, target_idx):
    """Full pipeline: sim -> crop -> normalize -> steer -> model -> metrics."""
    premix = simulate_room(signals, rel_positions, mic_positions)
    mix_all = premix.sum(axis=0)                      # (16, T)
    target_pre = premix[target_idx]                   # (16, T)

    # 5 s window with the most target energy at the reference mic (training-style)
    ref = target_pre[tr.REFERENCE_MIC].astype(np.float64)
    if mix_all.shape[-1] > SEG_LEN:
        csum = np.cumsum(ref ** 2)
        windows = csum[SEG_LEN - 1:] - np.concatenate(([0.0], csum[:-SEG_LEN]))
        s = int(np.argmax(windows))
    else:
        s = 0
    mixture = mix_all[:, s:s + SEG_LEN]
    target_ref = target_pre[tr.REFERENCE_MIC, s:s + SEG_LEN]
    if mixture.shape[-1] < SEG_LEN:                   # safety pad
        pad = SEG_LEN - mixture.shape[-1]
        mixture = np.pad(mixture, ((0, 0), (0, pad)))
        target_ref = np.pad(target_ref, (0, pad))

    # Same per-example level normalization as training
    scale = float(mixture.std()) or 1e-8
    mixture = (mixture / scale).astype(np.float32)
    target_ref = (target_ref / scale).astype(np.float32)

    # Steering vector for the target azimuth
    x, y = rel_positions[target_idx]
    az = azimuth_of(x, y)
    steering_np = compute_steering_vector(az, mic_positions, FS, tr.N_FFT)

    window = torch.hann_window(tr.WIN_LENGTH)
    mix_t = torch.tensor(mixture).unsqueeze(0)
    steering = torch.tensor(steering_np, dtype=torch.complex64).unsqueeze(0)
    x_stft = tr.compute_stft(mix_t, tr.N_FFT, tr.HOP_LENGTH, tr.WIN_LENGTH, window)
    with torch.no_grad():
        s_hat = model(x_stft, steering)
    extracted = torch.istft(
        s_hat, n_fft=tr.N_FFT, hop_length=tr.HOP_LENGTH, win_length=tr.WIN_LENGTH,
        window=window, center=True, length=SEG_LEN,
    ).squeeze(0).numpy()

    tgt_t = torch.tensor(target_ref).unsqueeze(0)
    sisdr_mix = tr.si_sdr_db(torch.tensor(mixture[tr.REFERENCE_MIC]).unsqueeze(0), tgt_t).item()
    sisdr_ext = tr.si_sdr_db(torch.tensor(extracted).unsqueeze(0), tgt_t).item()

    return {
        "mixture_ref": mixture[tr.REFERENCE_MIC],
        "extracted": extracted.astype(np.float32),
        "target_ref": target_ref,
        "azimuth": az,
        "sisdr_mix": sisdr_mix,
        "sisdr_ext": sisdr_ext,
    }


def playback_group(*signals):
    """Common scale across signals so relative loudness is preserved."""
    peak = max(float(np.abs(s).max()) for s in signals)
    g = 0.9 / max(peak, 1e-8)
    return [(s * g).astype(np.float32) for s in signals]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def room_plot(speakers, target_idx):
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    fig.patch.set_facecolor("white")

    # Recessive range rings + crosshair
    for r in (10, 20, 30, 40):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=COL_GRID, lw=1))
        ax.annotate(f"{r} m", (r * 0.7071, r * 0.7071), color=COL_MUTED,
                    fontsize=7, ha="left", va="bottom")
    ax.axhline(0, color=COL_GRID, lw=1, zorder=0)
    ax.axvline(0, color=COL_GRID, lw=1, zorder=0)

    # Mic array at the origin
    ax.scatter([0], [0], marker="s", s=90, color=COL_INK, zorder=5)
    ax.annotate("mic array", (0, 0), xytext=(0, -3.0), ha="center",
                color=COL_INK, fontsize=9, fontweight="bold")

    # Steering line to the target
    tx, ty = speakers[target_idx]["x"], speakers[target_idx]["y"]
    ax.plot([0, tx], [0, ty], color=COL_TARGET, lw=1.5, ls="--", zorder=2)
    az = azimuth_of(tx, ty)

    for i, spk in enumerate(speakers):
        is_t = i == target_idx
        color = COL_TARGET if is_t else COL_OTHER
        marker = "*" if is_t else "o"
        size = 260 if is_t else 90
        ax.scatter([spk["x"]], [spk["y"]], marker=marker, s=size, color=color,
                   edgecolors="white", linewidths=1.2, zorder=6)
        sym = "♀" if spk["gender"] == "Female" else "♂"
        tag = f"{spk['name']} {sym}" + ("  (TARGET)" if is_t else "")
        ax.annotate(tag, (spk["x"], spk["y"]), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    color=COL_INK if is_t else COL_MUTED,
                    fontsize=9, fontweight="bold" if is_t else "normal", zorder=7)

    ax.annotate(f"steering {az:.1f}°", (tx * 0.5, ty * 0.5), color=COL_TARGET,
                fontsize=8, ha="center", va="bottom")

    lim = 46
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=COL_MUTED)
    ax.set_ylabel("y (m)", color=COL_MUTED)
    ax.tick_params(colors=COL_MUTED, labelsize=8)
    for side in ax.spines.values():
        side.set_color(COL_GRID)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Directional Speaker Extraction",
                       page_icon="\U0001f3af", layout="wide")
    st.title("\U0001f3af Directional Speaker Extraction")

    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        st.error("No trained checkpoint found (looked in checkpoints_v4/v3/v2). "
                 "Train the model first, then reload this app.")
        st.stop()
    model, ckpt_epoch = load_model(ckpt_path)
    spacing = mic_spacing_for(ckpt_path)
    mic_positions = build_mic_positions(spacing)
    st.caption(
        f"Model: `{os.path.relpath(ckpt_path, APP_DIR)}` (epoch {ckpt_epoch}) · "
        f"mic array: 4×4 grid, {spacing*100:.0f} cm spacing (matched to this "
        f"checkpoint's training geometry) · inference on CPU"
    )

    pools = speaker_pools()
    if not pools["Male"] or not pools["Female"]:
        st.error("LibriSpeech voice pools are empty -- check LIBRI_DIR / SPEAKERS.TXT.")
        st.stop()

    # ---------------- sidebar: room setup ----------------
    with st.sidebar:
        st.header("Room setup")
        n_spk = st.number_input("Number of speakers", MIN_SPK, MAX_SPK, 3,
                                help=f"The model was trained with {MIN_SPK}-{MAX_SPK} "
                                     "speakers in the room.")
        speakers = []
        for i in range(int(n_spk)):
            with st.expander(f"Speaker {i + 1}", expanded=i == 0):
                name = st.text_input("Name", value=f"Speaker {i + 1}",
                                     key=f"name_{i}", max_chars=20)
                gender = st.selectbox("Voice type", ["Male", "Female"],
                                      key=f"gender_{i}")

                # Default positions: evenly spread on a 10 m ring
                ang = 2 * math.pi * i / int(n_spk)
                col_x, col_y = st.columns(2)
                x = col_x.number_input("x (m)", -45.0, 45.0,
                                       value=round(10 * math.cos(ang), 1),
                                       step=0.5, key=f"x_{i}")
                y = col_y.number_input("y (m)", -45.0, 45.0,
                                       value=round(10 * math.sin(ang), 1),
                                       step=0.5, key=f"y_{i}")

                rec = st.audio_input(
                    "\U0001f3a4 Record this speaker (optional, first 5 s used)",
                    key=f"rec_{i}",
                    help="Click the mic to record; the X on the widget deletes "
                         "the take so you can record again.",
                )

                # LibriSpeech voice assignment (kept stable in session state,
                # re-picked when the gender changes or on 'New voice')
                vkey, gkey = f"voice_{i}", f"voiceg_{i}"
                used = {st.session_state.get(f"voice_{j}", (None, None))[0]
                        for j in range(MAX_SPK) if j != i}
                if vkey not in st.session_state or st.session_state.get(gkey) != gender:
                    st.session_state[vkey] = pick_voice(gender, used)
                    st.session_state[gkey] = gender
                if rec is not None:
                    st.caption("Using **your recording** for this speaker.")
                else:
                    spk_id, clip_path = st.session_state[vkey]
                    vc1, vc2 = st.columns([3, 1])
                    vc1.caption(f"LibriSpeech voice: speaker **{spk_id}** ({gender})")
                    if vc2.button("\U0001f500", key=f"newvoice_{i}",
                                  help="Pick a different LibriSpeech voice"):
                        st.session_state[vkey] = pick_voice(gender, used)
                        st.rerun()

                speakers.append({"name": name.strip() or f"Speaker {i + 1}",
                                 "gender": gender, "x": float(x), "y": float(y),
                                 "recording": rec,
                                 "voice": st.session_state[vkey]})

    # ---------------- validity warnings ----------------
    warnings = []
    for i, s in enumerate(speakers):
        if math.hypot(s["x"], s["y"]) < MIN_SOURCE_DIST:
            warnings.append(f"**{s['name']}** is {math.hypot(s['x'], s['y']):.1f} m "
                            f"from the array (training minimum: {MIN_SOURCE_DIST:.0f} m).")
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            a1 = azimuth_of(speakers[i]["x"], speakers[i]["y"])
            a2 = azimuth_of(speakers[j]["x"], speakers[j]["y"])
            d = abs(a1 - a2) % 360
            sep = min(d, 360 - d)
            if sep < MIN_ANGULAR_SEP_DEG:
                warnings.append(f"**{speakers[i]['name']}** and **{speakers[j]['name']}** "
                                f"are only {sep:.1f}° apart (training minimum: "
                                f"{MIN_ANGULAR_SEP_DEG:.0f}°) -- extraction will degrade.")

    # ---------------- main: map + target + run ----------------
    col_map, col_run = st.columns([1.05, 1])

    with col_run:
        st.subheader("Target")
        target_idx = st.radio(
            "Which speaker should the model extract?",
            options=list(range(len(speakers))),
            format_func=lambda i: f"{speakers[i]['name']} "
                                  f"({azimuth_of(speakers[i]['x'], speakers[i]['y']):.0f}°)",
            key="target_idx_radio",
        )
        for w in warnings:
            st.warning(w)

        run = st.button("▶ Run extraction", type="primary", use_container_width=True)

    with col_map:
        st.subheader("Room map")
        st.pyplot(room_plot(speakers, target_idx), clear_figure=True)

    if run:
        with st.spinner("Preparing voices, simulating the room, running the model..."):
            signals = []
            for s in speakers:
                if s["recording"] is not None:
                    signals.append(load_recording(s["recording"]))
                else:
                    signals.append(load_librispeech_clip(s["voice"][1]))
            rel_positions = [(s["x"], s["y"]) for s in speakers]
            result = run_extraction(model, mic_positions, signals,
                                    rel_positions, int(target_idx))
            result["names"] = [s["name"] for s in speakers]
            result["target_name"] = speakers[int(target_idx)]["name"]
            result["signals"] = signals
            st.session_state["results"] = result

    # ---------------- results ----------------
    if "results" in st.session_state:
        res = st.session_state["results"]
        st.divider()
        st.subheader(f"Results -- target: {res['target_name']} "
                     f"(steered at {res['azimuth']:.1f}°)")

        m1, m2, m3 = st.columns(3)
        m1.metric("SI-SDR in mixture", f"{res['sisdr_mix']:.2f} dB")
        m2.metric("SI-SDR extracted", f"{res['sisdr_ext']:.2f} dB",
                  delta=f"{res['sisdr_ext'] - res['sisdr_mix']:+.2f} dB")
        m3.metric("Speakers in room", f"{len(res['names'])}")

        mix_p, ext_p, tgt_p = playback_group(
            res["mixture_ref"], res["extracted"], res["target_ref"])
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("**Mixture** (reference mic)")
            st.audio(mix_p, sample_rate=FS)
        with a2:
            st.markdown("**Extracted target**")
            st.audio(ext_p, sample_rate=FS)
        with a3:
            st.markdown("**Clean target** (ground truth)")
            st.audio(tgt_p, sample_rate=FS)

        with st.expander("Input voice clips (before room simulation)"):
            for name, sig in zip(res["names"], res["signals"]):
                st.markdown(f"**{name}**")
                peak = float(np.abs(sig).max()) or 1e-8
                st.audio((sig * (0.9 / peak)).astype(np.float32), sample_rate=FS)


if __name__ == "__main__":
    main()
