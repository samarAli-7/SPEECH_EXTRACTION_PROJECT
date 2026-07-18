import numpy as np
import pyroomacoustics as pra
import torch
import torchaudio
import librosa
import math
import random
import os

# -----------------------
# Global Parameters
# -----------------------
ROOM_SIZE = [100, 100, 10]
FS = 16000
# Mic spacing: 0.10 m (4x4 grid -> 30 cm aperture). Doubled from 0.05 m: angular
# resolution scales with aperture (beamwidth ~ wavelength/aperture), and the old
# 15 cm aperture was the main limit on how sharply the model could separate
# directions. Wider spacing does alias above ~1.7 kHz, but a learned wideband
# system resolves that ambiguity across frequencies (only the true direction is
# consistent at every frequency). NOTE: if you have real hardware, this MUST
# match its physical spacing -- a model trained at 0.10 m will not work on a
# 0.05 m array.
SPACING = 0.10
CENTER = np.array([ROOM_SIZE[0] / 2, ROOM_SIZE[1] / 2, ROOM_SIZE[2] / 2])

# Direction-conditioned training constraints:
# ALL source pairs must be at least this many degrees of azimuth apart --
# any speaker can become the extraction target, so if two sources shared a
# direction, "extract the speaker at direction theta" would be ill-posed and
# the model would get contradictory supervision at overlapping T-F bins.
MIN_ANGULAR_SEP_DEG = 15.0
# Sources must be at least this far from the array centre so the far-field
# steering-vector model (plane wave) stays valid and the azimuth is well-defined.
# Raised alongside the wider aperture: far-field distance grows with aperture^2.
MIN_SOURCE_DIST = 3.0
# Every source's received level at the mics (after simulation, so independent of
# its distance) is drawn uniformly from this dB range around a common reference.
# All sources are treated symmetrically: no source is statistically louder or
# quieter by construction, so level cannot be used as a shortcut to identify
# the target -- only direction can.
SOURCE_LEVEL_DB_RANGE = (-6.0, 6.0)

# -----------------------
# Helper Functions
# -----------------------
def get_mic_positions():
    """Generates the 3x16 array of microphone positions for the 4x4 grid."""
    mic_positions = []
    offset = (4 - 1) / 2.0
    for i in range(4):
        for j in range(4):
            x = CENTER[0] + (i - offset) * SPACING
            y = CENTER[1] + (j - offset) * SPACING
            z = CENTER[2]
            mic_positions.append([x, y, z])
            
    # Convert to shape (3, 16)
    return np.array(mic_positions).T

def angular_difference_deg(a, b):
    """Smallest absolute difference between two azimuths, in [0, 180]."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

def _dist_from_center(pos):
    return math.hypot(pos[0] - CENTER[0], pos[1] - CENTER[1])

def pick_source_position(existing_azimuths_deg):
    """
    Sample a source position anywhere in the room that is (a) at least
    MIN_SOURCE_DIST from the array centre and (b) at least MIN_ANGULAR_SEP_DEG
    of azimuth away from EVERY already-placed source. The separation must hold
    pairwise because any speaker can become the extraction target. Falls back
    to the last draw if 500 tries fail (with 11 sources x 15 deg exclusion
    cones there is always >= 195 deg of free azimuth, so this is theoretical).
    """
    pos = None
    for _ in range(500):
        pos = [random.uniform(1, 99), random.uniform(1, 99), 5]
        if _dist_from_center(pos) < MIN_SOURCE_DIST:
            continue
        az = get_target_direction(pos, CENTER)
        if all(angular_difference_deg(az, a) >= MIN_ANGULAR_SEP_DEG
               for a in existing_azimuths_deg):
            return pos
    return pos

def pick_no_of_interferers():
    return random.randint(1, 10)

def get_target_direction(target_pos, center_pos):
    """
    Computes the target direction in degrees from the center mic.
    Calculates the azimuth angle (0-360 degrees) on the X-Y plane.
    """
    dx = target_pos[0] - center_pos[0]
    dy = target_pos[1] - center_pos[1]
    
    # Calculate angle in radians using atan2 (handles all quadrants correctly)
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Normalize to [0, 360)
    return (angle_deg + 360) % 360


def compute_steering_vector(azimuth_deg, mic_positions, fs, n_fft):
    """
    Computes the narrowband steering vector for a far-field source at the
    given azimuth angle.

    The steering vector encodes the phase delay at each microphone relative
    to the array center, for every frequency bin:

        a_m(f) = exp(-j * 2*pi*f * tau_m)

    where tau_m = (d_m · u) / c is the time delay at mic m,
    d_m is the mic position relative to the array center,
    u = [cos(az), sin(az), 0] is the unit direction vector (horizontal plane),
    and c = 343 m/s is the speed of sound.

    Args:
        azimuth_deg   : float, azimuth angle in degrees [0, 360)
        mic_positions : np.ndarray, shape (3, N_mics), mic positions in metres
        fs            : int, sample rate in Hz
        n_fft         : int, FFT size

    Returns:
        steering : np.ndarray, shape (N_mics, n_fft//2 + 1), complex64
                   The steering vector for each mic and each frequency bin.
    """
    c = 343.0  # speed of sound in m/s
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0, fs / 2, n_freqs)  # (n_freqs,)

    az_rad = math.radians(azimuth_deg)
    # Unit direction vector in the horizontal plane
    u = np.array([math.cos(az_rad), math.sin(az_rad), 0.0])  # (3,)

    # Mic positions relative to the array center
    center = mic_positions.mean(axis=1, keepdims=True)  # (3, 1)
    d = mic_positions - center  # (3, N_mics)

    # Time delay at each mic: tau_m = (d_m · u) / c
    # d.T @ u -> (N_mics,)
    tau = (d.T @ u) / c  # (N_mics,)

    # Phase delay per mic per frequency: exp(-j * 2*pi*f * tau_m)
    # tau: (N_mics, 1), freqs: (1, n_freqs) -> (N_mics, n_freqs)
    phase = -2.0 * math.pi * tau[:, np.newaxis] * freqs[np.newaxis, :]
    steering = np.exp(1j * phase).astype(np.complex64)  # (N_mics, n_freqs)

    return steering

def load_and_pad_audio(file_path, target_length=None):
    """Loads audio, enforces 16kHz mono, and matches a target length if provided."""
    wav, sr = torchaudio.load(file_path)
    
    # Resample if needed
    if sr != FS:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=FS)(wav)
        
    # Convert to mono
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
        
    wav = wav.squeeze().numpy()

    # Match length to the target audio so matrix additions don't crash
    if target_length is not None:
        if len(wav) < target_length:
            # Pad or wrap the noise to match target length
            wav = np.pad(wav, (0, target_length - len(wav)), mode='wrap')
        elif len(wav) > target_length:
            # Truncate
            wav = wav[:target_length]
            
    return wav

# -----------------------
# Main Simulation Function
# -----------------------
def _energy_crop(signal, crop_len, n_tries=8):
    """
    Random-crop `signal` to `crop_len` samples, preferring crops that actually
    contain speech energy (LibriSpeech has long leading/trailing silences, and
    training on all-silent targets teaches the model to output silence).
    Accepts the first crop whose mean power reaches half of the full-utterance
    mean power; otherwise keeps the most energetic of `n_tries` draws.
    """
    if len(signal) <= crop_len:
        return signal
    full_power = float(np.mean(signal ** 2)) + 1e-12
    best_start, best_power = 0, -1.0
    for _ in range(n_tries):
        start = random.randint(0, len(signal) - crop_len)
        p = float(np.mean(signal[start:start + crop_len] ** 2))
        if p >= 0.5 * full_power:
            return signal[start:start + crop_len]
        if p > best_power:
            best_start, best_power = start, p
    return signal[best_start:best_start + crop_len]


def generate_example(target_audio_path, interferer_audio_paths, max_signal_seconds=None):
    """
    Simulates a room acoustic environment and returns training data.
    Import this function into your main PyTorch training script.

    ANY-SPEAKER-TARGET version: all sources in the room are treated
    symmetrically -- same position sampling (pairwise >= MIN_ANGULAR_SEP_DEG
    apart), same energy-cropped clips, same per-source received-level
    randomization -- and ONE of them is chosen uniformly at random as the
    supervision target after simulation. The model therefore cannot identify
    the target by level, prominence, or any statistical role; only the given
    direction identifies it.

    Args:
        target_audio_path (str): First source clip; also sets the segment length.
        interferer_audio_paths (list): Clips for the other sources (all of them
            candidates for being the chosen target).
        max_signal_seconds (float or None): if given, every clip is energy-
            cropped to this length *before* the room simulation. Simulating only
            the training segment instead of the full utterance makes on-the-fly
            data generation many times faster.

    Returns:
        dict: Contains mixture, the chosen speaker's direct-path signal at the
              mics ("target_reverb", name kept for compatibility), and its
              azimuth ("target_azimuth_deg").
    """

    # 1. Load the first source clip (optionally pre-cropped for speed); it
    #    defines the common segment length for every source.
    first_signal = load_and_pad_audio(target_audio_path)
    if max_signal_seconds is not None:
        first_signal = _energy_crop(first_signal, int(round(max_signal_seconds * FS)))
    signal_length = len(first_signal)

    # 2. Initialize the Room -- ANECHOIC: max_order=0 keeps only the direct
    #    path from each source to each mic (no wall reflections at all).
    room = pra.ShoeBox(
        ROOM_SIZE,
        fs=FS,
        max_order=0,
    )

    # 3. Add the 4x4 Microphone Array
    mic_positions = get_mic_positions()
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

    # 4. Load all source clips. Every clip (not just the first) gets an
    #    energy-aware crop so each speaker is actually talking in the segment;
    #    clips shorter than the segment are wrap-padded.
    num_interferers = min(pick_no_of_interferers(), len(interferer_audio_paths))
    signals = [first_signal]
    for path in interferer_audio_paths[:num_interferers]:
        s = load_and_pad_audio(path)
        s = _energy_crop(s, signal_length)
        if len(s) < signal_length:
            s = np.pad(s, (0, signal_length - len(s)), mode="wrap")
        signals.append(s)
    n_sources = len(signals)

    # 5. Place all sources with pairwise angular separation and record azimuths.
    azimuths = []
    for sig in signals:
        pos = pick_source_position(azimuths)
        azimuths.append(get_target_direction(pos, CENTER))
        room.add_source(pos, signal=sig)

    # 6. Simulate the room
    # return_premix=True makes simulate() return an array of shape
    # (n_sources, n_mics, n_samples) with each source separated at the mics.
    premix = room.simulate(return_premix=True)

    # 7. Per-source received-level randomization: scale each source so its
    #    power at the mics sits at a random level in SOURCE_LEVEL_DB_RANGE
    #    around a common reference. This (a) undoes the huge 1/r level spread
    #    from random distances, so no speaker is inaudible, and (b) keeps all
    #    sources statistically identical, so level carries no information about
    #    which one is the target.
    scaled = np.empty_like(premix)
    for k in range(n_sources):
        p_k = float(np.mean(premix[k] ** 2)) + 1e-12
        level_db = random.uniform(*SOURCE_LEVEL_DB_RANGE)
        scaled[k] = premix[k] * math.sqrt(10.0 ** (level_db / 10.0) / p_k)

    mixture_mics = scaled.sum(axis=0)

    # 8. Choose the supervision target uniformly among ALL sources.
    chosen = random.randrange(n_sources)
    target_reverb_mics = scaled[chosen]
    target_azimuth_deg = azimuths[chosen]
    target_signal = signals[chosen]

    # Return a dictionary containing everything your training script will need.
    # "target_*" keys describe the one randomly chosen speaker (kept for
    # single-target consumers); "sources_reverb"/"azimuths" expose EVERY source
    # so the trainer can supervise all of them (v5 all-speaker training).
    return {
        "mixture": torch.tensor(mixture_mics, dtype=torch.float32),               # Shape: (16, N)
        "target_reverb": torch.tensor(target_reverb_mics, dtype=torch.float32),   # Shape: (16, N), anechoic direct path
        "clean_target": torch.tensor(target_signal, dtype=torch.float32),         # Shape: (N)
        "target_azimuth_deg": target_azimuth_deg,                                 # Float, [0, 360)
        "sources_reverb": torch.tensor(scaled, dtype=torch.float32),              # Shape: (K, 16, N), every source at the mics
        "azimuths": [float(a) for a in azimuths],                                 # K floats, [0, 360)
        "mic_positions": mic_positions,                                           # Shape: (3, 16) np.ndarray
        "fs": FS
    }