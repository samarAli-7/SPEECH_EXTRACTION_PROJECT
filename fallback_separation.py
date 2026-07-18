"""
Close-speaker fallback for direction-conditioned extraction.

Why
---
The direction-conditioned MVDR-UNet separates purely by azimuth, so its
resolution is bounded by the array beamwidth. `analyze_angular_safety.py`
measures exactly where that breaks down; below the measured safe angle the
neighbouring speaker starts leaking into the output and quality collapses.

What this does
--------------
When (and ONLY when) the target has a neighbour inside the unsafe cone:

  1. Beamform: run the direction-conditioned model steered at the target. Every
     speaker OUTSIDE the cone is already strongly suppressed by this stage, so
     what remains is essentially a 2- or 3-speaker mixture no matter how many
     people are in the room.
  2. Separate: run a blind speech separator on that beamformed signal --
     the 2-speaker pipeline if one neighbour is inside the cone, the 3-speaker
     pipeline if two or more are.
  3. Identify: embed every separated stream with ECAPA-TDNN and report the
     stream whose embedding is closest (cosine) to the reference embedding.
     The reference is the target's enrolment audio when the caller has it
     (the app does), otherwise the beamformed signal itself -- which still
     leans towards the target because of step 1.

Separator choice and output reconstruction are configured by the constants
below; `benchmark_fallback.py` measures the options and the defaults here are
whatever it found best.

The handover is decided by angle alone. `benchmark_arbitration.py` tested the
obvious alternative -- run both paths and keep whichever output ECAPA finds
closer to the enrolment -- and it was consistently WORSE than the angular gate
(near the boundary that similarity comparison picks the better output only
about half the time), so it is deliberately not used.

Nothing in this module is imported at module load time -- the separation and
embedding models are loaded lazily on first use, so the normal (safe-geometry)
path pays nothing for it.
"""

import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

FS = 16000
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(APP_DIR, "pretrained")
THRESHOLDS_JSON = os.path.join(APP_DIR, "fallback_thresholds.json")

# --- geometry thresholds ---------------------------------------------------
# Hand over to the fallback when the nearest neighbour is closer than the
# trigger angle. The trigger depends on BOTH how many speakers are in the
# cluster and how many are in the room: every extra speaker degrades the
# beamformer, while the fallback only ever sees the cluster and stays flat, so
# the crossover moves outwards in crowded rooms (benchmark_density.py).
# trigger_table[cluster size][speakers in the room] = degrees; values in
# between are interpolated. 25.0 is the sweep ceiling -- at that density the
# fallback still won at the widest separation tested, which is also the widest
# separation that counts as a cluster (CLUSTER_CONE_DEG).
# Overridden by fallback_thresholds.json when that file is present.
DEFAULT_TRIGGER_TABLE = {
    2: {2: 8.0, 3: 8.0, 4: 11.0, 6: 17.5, 10: 25.0},
    3: {3: 13.5, 4: 17.5, 5: 25.0, 7: 25.0, 11: 25.0},
}
# Cone used to decide WHO is in the cluster (and therefore whether the 2- or
# 3-speaker separator runs). Matches the widest trigger in the table.
CLUSTER_CONE_DEG = 25.0

# --- model choice (see benchmark_fallback.py) ---
# Both chosen by measurement: at the separations where the fallback runs,
# wsj02mix beats whamr16k despite being an 8 kHz model, because "mask"
# reconstruction restores the full band from the beamformed signal anyway.
SEPARATOR_2SPK = "sepformer-wsj02mix"     # dir under pretrained/
SEPARATOR_3SPK = "sepformer-wsj03mix"
SEPARATOR_SR = {"sepformer-wsj02mix": 8000,
                "sepformer-wsj03mix": 8000,
                "sepformer-whamr16k": 16000}
SEPARATOR_SOURCE = {"sepformer-wsj02mix": "speechbrain/sepformer-wsj02mix",
                    "sepformer-wsj03mix": "speechbrain/sepformer-wsj03mix",
                    "sepformer-whamr16k": "speechbrain/sepformer-whamr16k"}
ECAPA_DIR = "spkrec-ecapa-voxceleb"
ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Reconstruction: "direct" returns the separated stream as-is (band-limited when
# the separator runs at 8 kHz); "mask" uses the separated streams only to build
# a time-frequency mask and takes the audio from the full-band beamformed
# signal, which restores the 4-8 kHz band an 8 kHz separator throws away.
RECONSTRUCTION = "mask"
MASK_N_FFT = 1024
MASK_HOP = 256
MASK_FLOOR = 0.0
MASK_POWER = 2.0        # ratio mask exponent: |S_k|^p / sum_j |S_j|^p


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def angular_diff_deg(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def trigger_table():
    """Measured trigger angles, from fallback_thresholds.json when available."""
    try:
        with open(THRESHOLDS_JSON) as f:
            raw = json.load(f)["trigger_table"]
        return {int(cs): {int(n): float(v) for n, v in row.items()}
                for cs, row in raw.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {k: dict(v) for k, v in DEFAULT_TRIGGER_TABLE.items()}


def trigger_for(cluster_size, n_speakers, table=None):
    """
    Trigger angle for this geometry: linear interpolation over the number of
    speakers in the room, clamped to the measured range at both ends.
    """
    table = trigger_table() if table is None else table
    cluster_size = min(max(int(cluster_size), 2), max(table))
    row = table.get(cluster_size) or table[max(table)]

    xs = sorted(row)
    n = int(n_speakers)
    if n <= xs[0]:
        return row[xs[0]]
    if n >= xs[-1]:
        return row[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= n <= hi:
            w = (n - lo) / (hi - lo)
            return row[lo] * (1 - w) + row[hi] * w
    return row[xs[-1]]


def analyze_geometry(azimuths_deg, target_idx, cone_deg=None, table=None):
    """
    Decide whether this scene needs the fallback, and which pipeline it needs.

    Returns a dict with:
      nearest_sep_deg  : azimuth distance to the closest other speaker
      nearest_idx      : index of that speaker
      cluster          : indices (target first) inside the cluster cone
      n_cluster        : cluster size, capped at 3 -- picks the 2- or 3-speaker
                         separator ("no matter how many are there")
      n_speakers       : speakers in the room (the trigger depends on it)
      needs_fallback   : True when the nearest neighbour is inside the trigger
      trigger_deg      : the trigger that applied for this cluster and density
    """
    cone_deg = CLUSTER_CONE_DEG if cone_deg is None else float(cone_deg)

    az_t = azimuths_deg[target_idx]
    seps = sorted((angular_diff_deg(az_t, az), i)
                  for i, az in enumerate(azimuths_deg) if i != target_idx)
    nearest_sep, nearest_idx = seps[0] if seps else (float("inf"), None)

    cluster = [target_idx] + [i for sep, i in seps if sep < cone_deg]
    n_cluster = min(len(cluster), 3)
    n_speakers = len(azimuths_deg)
    trigger = (trigger_for(n_cluster, n_speakers, table) if n_cluster >= 2 else 0.0)

    return {
        "nearest_sep_deg": nearest_sep,
        "nearest_idx": nearest_idx,
        "cluster": cluster,
        "n_cluster": n_cluster,
        "n_speakers": n_speakers,
        "needs_fallback": nearest_sep < trigger,
        "trigger_deg": trigger,
        "cone_deg": cone_deg,
    }


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------
def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.tensor(x, dtype=torch.float32)
    return x.detach().to(torch.float32)


def _resample(wav, sr_from, sr_to):
    if sr_from == sr_to:
        return wav
    import torchaudio
    return torchaudio.functional.resample(wav, sr_from, sr_to)


def cosine_similarity(a, b):
    a = a / (a.norm() + 1e-9)
    b = b / (b.norm() + 1e-9)
    return float((a * b).sum())


def _mask_reconstruct(streams, full_band, device):
    """
    Rebuild each stream from the FULL-BAND beamformed signal using a ratio mask
    derived from the (possibly band-limited) separated streams.

    streams   : (K, T) separated waveforms, already resampled to 16 kHz
    full_band : (T,) the beamformed signal the separator was run on
    """
    window = torch.hann_window(MASK_N_FFT, device=device)
    stft = lambda w: torch.stft(w, n_fft=MASK_N_FFT, hop_length=MASK_HOP,
                                win_length=MASK_N_FFT, window=window,
                                center=True, return_complex=True)
    S = torch.stack([stft(s) for s in streams])          # (K, F, T')
    B = stft(full_band)                                  # (F, T')

    mag = S.abs().clamp_min(1e-8) ** MASK_POWER
    # An 8 kHz separator leaves the top half of the band empty; there the mask
    # is undefined, so reuse the mask from the highest bin that carries energy.
    energy_per_bin = mag.sum(dim=(0, 2))                 # (F,)
    valid = energy_per_bin > energy_per_bin.max() * 1e-6
    if valid.any():
        last_valid = int(torch.nonzero(valid).max())
        if last_valid < mag.shape[1] - 1:
            mag[:, last_valid + 1:, :] = mag[:, last_valid:last_valid + 1, :]

    mask = mag / mag.sum(dim=0, keepdim=True).clamp_min(1e-8)
    if MASK_FLOOR > 0:
        mask = mask.clamp_min(MASK_FLOOR)

    out = []
    for k in range(mask.shape[0]):
        y = torch.istft(mask[k] * B, n_fft=MASK_N_FFT, hop_length=MASK_HOP,
                        win_length=MASK_N_FFT, window=window, center=True,
                        length=full_band.shape[-1])
        out.append(y)
    return torch.stack(out)


# ---------------------------------------------------------------------------
# the fallback
# ---------------------------------------------------------------------------
class SeparationFallback:
    """
    Lazy holder for the separation + speaker-embedding models.

    Usage:
        fb = SeparationFallback(device="cpu")
        out = fb.run(beamformed_wave, n_cluster=2, reference=enrolment_wave)
        out["waveform"]      -> the chosen stream (np.float32, 16 kHz)
        out["similarities"]  -> cosine similarity of every stream to the reference
        out["chosen"]        -> index of the reported stream
    """

    def __init__(self, device="cpu", separator_2spk=SEPARATOR_2SPK,
                 separator_3spk=SEPARATOR_3SPK, reconstruction=RECONSTRUCTION):
        self.device = device
        self.separator_names = {2: separator_2spk, 3: separator_3spk}
        self.reconstruction = reconstruction
        self._separators = {}
        self._ecapa = None

    # -- model loading -------------------------------------------------------
    def _get_separator(self, n_spk):
        n_spk = 2 if n_spk <= 2 else 3
        name = self.separator_names[n_spk]
        if name not in self._separators:
            from speechbrain.inference.separation import SepformerSeparation
            self._separators[name] = SepformerSeparation.from_hparams(
                source=SEPARATOR_SOURCE[name],
                savedir=os.path.join(PRETRAINED_DIR, name),
                run_opts={"device": self.device},
            )
        return self._separators[name], SEPARATOR_SR[name], name

    def _get_ecapa(self):
        if self._ecapa is None:
            from speechbrain.inference.speaker import EncoderClassifier
            self._ecapa = EncoderClassifier.from_hparams(
                source=ECAPA_SOURCE,
                savedir=os.path.join(PRETRAINED_DIR, ECAPA_DIR),
                run_opts={"device": self.device},
            )
        return self._ecapa

    def warmup(self, n_spk=(2, 3)):
        """Preload models (so the UI can pay the cost up front)."""
        for n in n_spk:
            self._get_separator(n)
        self._get_ecapa()

    # -- pieces --------------------------------------------------------------
    def separate(self, wave, n_spk):
        """
        Blind-separate a mono 16 kHz waveform into n_spk streams (16 kHz).

        The separator may run at 8 kHz internally; with RECONSTRUCTION="mask"
        the returned streams are rebuilt full-band from `wave`.
        """
        model, sr, name = self._get_separator(n_spk)
        w = _to_tensor(wave).to(self.device)
        peak = w.abs().max().clamp_min(1e-8)
        w_norm = w / peak

        w_sep = _resample(w_norm, FS, sr)
        with torch.no_grad():
            est = model.separate_batch(w_sep.unsqueeze(0))   # (1, T, K)
        streams = est.squeeze(0).permute(1, 0).contiguous()   # (K, T)
        streams = _resample(streams, sr, FS)
        if streams.shape[-1] < w.shape[-1]:
            streams = F.pad(streams, (0, w.shape[-1] - streams.shape[-1]))
        streams = streams[..., :w.shape[-1]]

        if self.reconstruction == "mask":
            streams = _mask_reconstruct(streams, w_norm, self.device)
        return (streams * peak).cpu(), name

    def embed(self, wave):
        """ECAPA-TDNN embedding of a mono 16 kHz waveform (L2-normalized)."""
        ecapa = self._get_ecapa()
        w = _to_tensor(wave).to(self.device)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        w = w / w.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
        with torch.no_grad():
            emb = ecapa.encode_batch(w).squeeze(1)           # (B, 192)
        return (emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-9)).cpu()

    # -- full pipeline -------------------------------------------------------
    def run(self, beamformed, n_cluster=2, reference=None):
        """
        beamformed : (T,) mono 16 kHz output of the direction-conditioned model,
                     steered at the target.
        n_cluster  : speakers inside the unsafe cone (2 -> 2-spk pipeline,
                     >=3 -> 3-spk pipeline).
        reference  : (T',) enrolment audio of the target speaker. When None the
                     beamformed signal itself is the reference -- it still leans
                     towards the target, so the closest stream is usually right,
                     but enrolment audio is markedly more reliable at very small
                     separations.
        """
        beamformed = _to_tensor(beamformed).squeeze()
        streams, sep_name = self.separate(beamformed, n_cluster)

        ref_wave = beamformed if reference is None else _to_tensor(reference).squeeze()
        ref_emb = self.embed(ref_wave)[0]
        stream_embs = self.embed(streams)
        sims = [cosine_similarity(stream_embs[k], ref_emb) for k in range(streams.shape[0])]
        chosen = int(np.argmax(sims))

        order = np.argsort(sims)[::-1]
        margin = float(sims[order[0]] - sims[order[1]]) if len(sims) > 1 else float("nan")

        return {
            "waveform": streams[chosen].numpy().astype(np.float32),
            "streams": streams.numpy().astype(np.float32),
            "similarities": sims,
            "chosen": chosen,
            "margin": margin,
            "reference_kind": "enrolment" if reference is not None else "beamformed",
            "separator": sep_name,
            "reconstruction": self.reconstruction,
        }


# module-level singleton so the Streamlit app reuses one set of loaded models
_SHARED = {}


def shared_fallback(device="cpu", **kwargs):
    key = (device, tuple(sorted(kwargs.items())))
    if key not in _SHARED:
        _SHARED[key] = SeparationFallback(device=device, **kwargs)
    return _SHARED[key]
