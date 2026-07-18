
"""
=======================================================================================
 Training script for: "An MVDR-Embedded U-Net Beamformer for Effective and Robust
 Multichannel Speech Enhancement" (Lee, Patel, Yang, Shen, Jin -- ICASSP 2024)
=======================================================================================

This implements the full proposed system of the paper:
  - A complex-valued U-Net direct beamformer (5 encoder levels, 4 decoder levels).
  - An "intra-MVDR" module embedded between the encoder and decoder at levels 1-4
    (Fig. 2 / Fig. 3, Sec. 3.1-3.3): each module estimates T-F masks from the
    encoder's own feature maps, forms speech/noise PSD matrices from those masks
    (Eqs. 1-2), computes an MVDR filter with respect to *every* microphone as
    reference (Eqs. 3-4), and feeds the resulting N MVDR-filtered spectrograms into
    the decoder as extra spatial features.
  - Final reconstruction as a weighted sum of filter-and-sum on the noisy STFTs and
    on the level-1 MVDR-filtered STFTs (Eq. 5).
  - The power-law compressed reconstruction loss used in the paper (Sec. 4,
    "Model settings", after Braun & Tashev, ref. [29]).
  - Adam optimization, lr 1e-3 -> 1e-4 at epoch 50, 80 epochs, batch size 4, 4-second
    training crops -- all exactly as stated in Sec. 4.

Data pipeline: this script imports `generate_example` from `dataset_generation.py`
(the pyroomacoustics-based simulator you supplied) and wraps it in a torch Dataset
that performs the room simulation on the fly and applies the paper's 4-second random
cropping during training.

-----------------------------------------------------------------------------------
WHAT IS TAKEN DIRECTLY FROM THE PAPER (should not be changed if you want a faithful
reproduction):
  - U-Net topology, per-level channel-concatenation pattern (N+C_k in the encoder,
    N+2*C_k in the decoder), and the multi-level intra-MVDR placement (Fig. 2).
  - MVDR PSD estimation (Eqs. 1-2) and per-reference-channel filter (Eqs. 3-4).
  - Final combination rule, Eq. 5.
  - Encoder/decoder block structure: two stacks of (complex 3x3 conv -> complex BN
    -> complex leaky ReLU) (Sec. 3, "Network architecture details").
  - Mask estimation network structure: complex 3x3 conv -> BN -> leaky ReLU -> 1x1
    conv -> Sigmoid, with C_out equal to the encoder's own output channels at that
    level.
  - STFT: Hann window, 1024-point FFT, 256-sample hop.
  - Training regime: Adam, lr 1e-3 decayed to 1e-4 at epoch 50, 80 epochs, batch
    size 4, 4-second training crops, full utterances at test time.
  - Base model channel widths C1..C4 = 32, 64, 64, 64.
  - The combined power-law compressed loss, alpha=0.3 on the complex term / 0.7 on
    the magnitude term, power=0.3.

WHAT THE PAPER DOES *NOT* PIN DOWN NUMERICALLY, AND WHICH I HAD TO CHOOSE (clearly
marked below with `# NOTE:` comments so you can change them if you have another
source for these values):
  - The leaky-ReLU negative slope (paper never gives a number). I use 0.1, a common
    choice in the complex speech-enhancement literature the paper itself cites
    (Deep Complex U-Net / DCCRN).
  - How the complex encoder feature map is turned into a *real* [0,1] mask inside
    the mask-estimation network (the paper writes "1x1 convolution -> Sigmoid" but
    Sigmoid needs a real input). I take the magnitude of the complex features right
    before the final 1x1 conv, and that final 1x1 conv is a real-valued conv
    (Cout -> 2 channels: M_s, M_v).
  - Diagonal loading added to Phi_v before inversion, needed for numerical
    stability of the MVDR solve (any real implementation of Eq. 4 needs this; the
    paper's equations are the noise-free idealization).
  - Complex batch normalization is implemented as the full covariance-whitening
    formulation of Trabelsi et al. (2018) / Deep Complex U-Net (paper's ref. [23]),
    since the paper explicitly says it follows [23, 24].
  - Complex max-pooling picks, per pooling window, the complex value whose
    *magnitude* is largest (a standard extension of max-pooling to complex tensors;
    the paper does not spell out how "max pool 2x2" is defined for complex tensors).
  - Reference microphone index into the 16-channel array (REFERENCE_MIC below) --
    the paper uses channel 5 of CHiME-3's 6-mic array; there is no direct analogue
    for the 4x4 array in your simulator, so I picked one of the centre-most
    elements. Change it freely.
  - Gradient clipping (GRAD_CLIP_NORM) is an optional safeguard, not mentioned in
    the paper, added because the differentiable matrix inverse inside intra-MVDR
    can occasionally produce large gradients early in training. Set it to None to
    disable it and be strictly faithful to the paper's stated optimizer settings.

-----------------------------------------------------------------------------------
V2 CHANGES FOR DIRECTION-CONDITIONED TARGET-SPEAKER EXTRACTION (this revision):
  - Steering conditioning is now done with *phase-aligned mixture channels*
    (x_m * conj(a_m), see SteeringConditionedMVDRUNet) instead of concatenating
    the raw steering vector; the intra-MVDR modules and the final filter-and-sum
    operate on the 16 physical mic channels only (see MVDRUNetSE docstring).
  - Data pipeline: interferers are kept >= 15 deg of azimuth away from the target
    (otherwise direction-conditioned extraction is ill-posed), come from different
    speakers than the target, the mixture SIR is sampled uniformly in [-5, 10] dB,
    sources are kept >= 2 m from the array (far-field validity), and each example
    is level-normalized. Target utterances are energy-cropped *before* the room
    simulation (~7x faster data generation), avoiding silent training targets.
  - Loss: power-law compressed loss (now bin-averaged) + weighted negative SI-SDR
    on the iSTFT waveform.
  - Validation is deterministic (fixed seed per index), tracks SI-SDR, and the
    best checkpoint is selected by highest validation SI-SDR.
  - MVDR diagonal loading raised to 1e-4 (relative) for a stable solve early on.
  - Checkpoints go to ./checkpoints_v2 (v1 checkpoints are incompatible).

V3 CHANGES:
  - ANECHOIC simulation: max_order=0 in dataset.py (direct path only, no wall
    reflections). The training target is the anechoic direct-path signal at the
    reference mic.
  - 5-second training segments (SEGMENT_SECONDS = 5.0), batch size 2.
  - Checkpoints go to ./checkpoints_v3 (same weight shapes as v2, but different
    task/metrics -- do not mix runs).

V4 CHANGES (any-speaker-target training, for the best direction generalization):
  - Every source in the room is now statistically identical: pairwise >= 15 deg
    azimuth separation, per-source energy-cropped clips, per-source received
    level drawn from SOURCE_LEVEL_DB_RANGE (replaces target-relative SIR), all
    speakers pairwise distinct. ONE source is chosen uniformly at random as the
    supervision target after simulation (dataset.py, generate_example). The
    model therefore cannot use level/prominence/role shortcuts -- the steering
    direction is the only cue identifying the target, which is exactly the
    ability we want maximized.
  - Mic spacing doubled to 0.10 m (30 cm aperture) for sharper angular
    resolution; MIN_SOURCE_DIST raised to 3 m to stay far-field.
  - Checkpoints go to ./checkpoints_v4. NOTE: v2/v3 checkpoints load (same
    shapes) but were trained on the 0.05 m array -- their spatial filters do
    not transfer; train v4 from scratch.

V5 CHANGES (all-speaker supervision):
  - Every simulated room now trains extraction of EVERY speaker in it, not one
    randomly chosen speaker: the dataset returns per-speaker targets/steerings
    (padded to MAX_SOURCES, masked by n_src), and each optimization step runs
    the model once per speaker on the shared mixture STFT, accumulating
    per-speaker gradients weighted to the mean over all (mixture, speaker)
    pairs. Validation SI-SDR is likewise the mean over all speakers of all
    rooms -- the model is graded on extracting *everyone* correctly.
  - Epoch sizes reduced (NUM_TRAIN_EXAMPLES 3000, NUM_VAL_EXAMPLES 300) since
    each room now contributes ~6.5x the supervision and compute.
  - Warm start from checkpoints_v4/best.pt (same geometry/task family) via
    INIT_FROM when no v5 checkpoint exists yet.
  - Checkpoints go to ./checkpoints_v5.
  - Multi-GPU: one process per GPU in GPUS (torch.multiprocessing.spawn + NCCL),
    per-GPU batch BATCH_SIZE=4 (measured: 11.2 GiB peak at 5 s segments; B=6
    OOMs on 16 GB), gradients averaged manually each step via
    all_reduce_gradients() -- see its docstring for why DDP is not used here.
    Validation is sharded by index across ranks and all-reduced, so the
    reported val SI-SDR is identical to a single-GPU evaluation.
-----------------------------------------------------------------------------------
"""

import os
import glob
import math
import random
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from dataset import generate_example, compute_steering_vector, get_mic_positions, FS as DATASET_FS

# =======================================================================================
# 1. HYPERPARAMETERS  (everything lives here -- no argparse, edit directly)
# =======================================================================================

# ---- Data locations (EDIT THESE to point at your actual audio) ----
TARGET_AUDIO_DIR = "/media/uasdtu/DataSets2/VOCAL_ALERTNESS/DSENet/DATASET/LibriSpeech"   # recursively scanned for .wav/.flac
NOISE_AUDIO_DIR = "/media/uasdtu/DataSets2/VOCAL_ALERTNESS/DSENet/DATASET/LibriSpeech"       # recursively scanned for .wav/.flac
VAL_FRACTION = 0.12          # fraction of target *files* held out for validation
SPLIT_SEED = 0               # RNG seed used only for the train/val file split

# ---- Array / model geometry (must match dataset_generation.py) ----
N_MICS = 16                   # 4x4 planar array from dataset_generation.py
REFERENCE_MIC = 5             # NOTE: implementation choice, see module docstring above

# ---- U-Net channel widths (paper's "base" model, Table 1) ----
C1, C2, C3, C4 = 32, 64, 64, 64

# ---- Which intra-MVDR levels are active. (1,2,3,4) = paper's best full model
#      (last row of Table 1, PESQ 2.64). Level 1 must always be included because
#      its MVDR output feeds the final reconstruction, Eq. 5. ----
MVDR_LEVELS = (1, 2, 3, 4)

# ---- STFT settings (Sec. 4, "Model settings") ----
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024

# ---- Training regime (Sec. 4, "Model settings") ----
SEGMENT_SECONDS = 5.0          # 5-second training segments
# Per-GPU batch size. Measured on the 16 GB RTX 5060 Ti at 5 s segments:
# B=2 -> 5.7 GiB / 0.74 s per fwd+bwd, B=4 -> 11.2 GiB / 0.83 s (~1.8x the
# per-sample throughput), B=6 -> OOM. So 4 is the sweet spot per GPU.
BATCH_SIZE = 4
# GPUs to train on (one process per GPU, gradient-averaged each step).
# Global batch = BATCH_SIZE * len(GPUS) = 8. Set GPUS = [1] for single-GPU.
GPUS = [0, 1]
NUM_EPOCHS = 80
LR_INITIAL = 1e-3
# The v1 run plateaued around epoch 8-11, but that was largely the ill-posed
# task (interferers at the target azimuth, same-speaker interference). With the
# fixed data pipeline, stay at high LR longer before decaying.
LR_DECAY_EPOCH = 25
LR_DECAY_FACTOR = 0.1          # 1e-3 -> 1e-4 at epoch 25, -> 1e-5 at epoch 55

# ---- Loss (Braun & Tashev power-law compressed loss, as cited in the paper) ----
COMPRESSION_POWER = 0.3
LOSS_ALPHA = 0.3               # weight on the complex (phase-aware) term; (1-alpha) on magnitude term
# The compressed-spectrum terms are averaged (not summed) over T-F bins so the
# loss scale is independent of segment length, then combined with a negative
# SI-SDR term on the iSTFT waveform. SI-SDR directly optimizes the quantity we
# care about for extraction quality; the weight puts ~10 dB of SI-SDR on the
# same footing as the spectral term.
SISDR_LOSS_WEIGHT = 0.05

# ---- Virtual epoch sizes for the on-the-fly simulator. ----
# v5: every simulated room now supervises ALL of its speakers (2-11, avg ~6.5
# extraction problems per simulation), so one "example" is ~6.5x the training
# signal and ~6.5x the compute of before. Epoch sizes are scaled down to keep
# epochs a reasonable length while still seeing ~19k extraction problems/epoch.
NUM_TRAIN_EXAMPLES = 3000
NUM_VAL_EXAMPLES = 300

# Maximum sources per room (1 first source + up to 10 others in dataset.py);
# dataset tensors are padded to this count and masked by the true source count.
MAX_SOURCES = 11

# ---- Implementation-only knobs (NOT specified by the paper -- see docstring) ----
LEAKY_SLOPE = 0.1
MVDR_EPS = 1e-6
MVDR_DIAG_LOADING = 1e-4       # relative diagonal loading of Phi_v before the solve
GRAD_CLIP_NORM = 5.0           # set to None to disable

# ---- Misc / bookkeeping ----
NUM_WORKERS = 4
LOG_EVERY = 10                 # v5 steps are heavy (~all speakers x 4 rooms x 2 GPUs);
                               # step 1 is also always logged so a fresh run shows
                               # signs of life within the first minute.
# v5: all-speaker supervision -- every simulated room trains extraction of
# EVERY speaker in it (previously one random speaker per room). Same geometry
# and weight shapes as v4.
CHECKPOINT_DIR = "./checkpoints_v5"
# Warm start: initialize model weights (only) from this checkpoint when v5 has
# no last.pt yet. v4 used the same 0.10 m array and data distribution, so its
# weights are a valid and much faster starting point. Set to None to disable.
INIT_FROM = "./checkpoints_v4/best.pt"
DEVICE = "cuda:1"


# =======================================================================================
# 2. Complex-valued building blocks
# =======================================================================================

class ComplexBatchNorm2d(nn.Module):
    """
    Complex batch normalization via 2x2 covariance whitening, following
    Trabelsi et al. 2018 ("Deep Complex Networks"), which is also the formulation
    used by the Deep Complex U-Net / DCCRN papers cited by this paper as refs
    [23, 24] for "complex-valued network operations".
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # learnable complex affine transform, parameterized as a 2x2 matrix + bias
        self.gamma_rr = nn.Parameter(torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.gamma_ii = nn.Parameter(torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean_r", torch.zeros(num_features))
        self.register_buffer("running_mean_i", torch.zeros(num_features))
        self.register_buffer("running_vrr", torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.register_buffer("running_vii", torch.full((num_features,), 1.0 / math.sqrt(2)))
        self.register_buffer("running_vri", torch.zeros(num_features))

    def forward(self, x):
        assert torch.is_complex(x)
        B, C, H, W = x.shape
        xr, xi = x.real, x.imag

        if self.training:
            mean_r = xr.mean(dim=(0, 2, 3))
            mean_i = xi.mean(dim=(0, 2, 3))
            xr_c = xr - mean_r.view(1, C, 1, 1)
            xi_c = xi - mean_i.view(1, C, 1, 1)
            vrr = (xr_c ** 2).mean(dim=(0, 2, 3)) + self.eps
            vii = (xi_c ** 2).mean(dim=(0, 2, 3)) + self.eps
            vri = (xr_c * xi_c).mean(dim=(0, 2, 3))

            with torch.no_grad():
                m = self.momentum
                self.running_mean_r.mul_(1 - m).add_(m * mean_r)
                self.running_mean_i.mul_(1 - m).add_(m * mean_i)
                self.running_vrr.mul_(1 - m).add_(m * vrr)
                self.running_vii.mul_(1 - m).add_(m * vii)
                self.running_vri.mul_(1 - m).add_(m * vri)
        else:
            mean_r, mean_i = self.running_mean_r, self.running_mean_i
            vrr, vii, vri = self.running_vrr, self.running_vii, self.running_vri
            xr_c = xr - mean_r.view(1, C, 1, 1)
            xi_c = xi - mean_i.view(1, C, 1, 1)

        # inverse square root of [[vrr, vri], [vri, vii]] (Trabelsi et al., Eq. 9)
        det = (vrr * vii - vri ** 2).clamp_min(self.eps)
        s = torch.sqrt(det)
        t = torch.sqrt((vrr + vii + 2 * s).clamp_min(self.eps))
        inv_st = 1.0 / (s * t)
        wrr = (vii + s) * inv_st
        wii = (vrr + s) * inv_st
        wri = -vri * inv_st

        wrr, wii, wri = (v.view(1, C, 1, 1) for v in (wrr, wii, wri))
        xr_hat = wrr * xr_c + wri * xi_c
        xi_hat = wri * xr_c + wii * xi_c

        grr, gii, gri = (v.view(1, C, 1, 1) for v in (self.gamma_rr, self.gamma_ii, self.gamma_ri))
        br, bi = self.beta_r.view(1, C, 1, 1), self.beta_i.view(1, C, 1, 1)
        out_r = grr * xr_hat + gri * xi_hat + br
        out_i = gri * xr_hat + gii * xi_hat + bi
        return torch.complex(out_r, out_i)


class ComplexLeakyReLU(nn.Module):
    """Leaky ReLU applied independently to the real and imaginary parts."""

    def __init__(self, negative_slope=0.1):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        return torch.complex(
            F.leaky_relu(x.real, self.negative_slope),
            F.leaky_relu(x.imag, self.negative_slope),
        )


def complex_max_pool2d(x, kernel_size=2, stride=2):
    """
    2x2 max pooling for complex tensors: within each window, keep the complex
    value whose *magnitude* is largest (selection is magnitude-based; both the
    real and imaginary part of the winning element are kept together).
    """
    mag = x.abs()
    _, idx = F.max_pool2d(mag, kernel_size=kernel_size, stride=stride, return_indices=True)
    B, C, H, W = x.shape
    Hp, Wp = idx.shape[-2], idx.shape[-1]
    xr_flat = x.real.reshape(B, C, H * W)
    xi_flat = x.imag.reshape(B, C, H * W)
    idx_flat = idx.reshape(B, C, Hp * Wp)
    xr_pooled = torch.gather(xr_flat, 2, idx_flat).reshape(B, C, Hp, Wp)
    xi_pooled = torch.gather(xi_flat, 2, idx_flat).reshape(B, C, Hp, Wp)
    return torch.complex(xr_pooled, xi_pooled)


class ComplexConvBlock(nn.Module):
    """
    Two stacks of (complex 3x3 conv -> complex batch norm -> complex leaky ReLU),
    exactly as described in Sec. 3 "Network architecture details":
    stack 1 maps Cin -> Cout, stack 2 maps Cout -> Cout. Used for every
    encoder and decoder layer of the U-Net.
    """

    def __init__(self, c_in, c_out, negative_slope=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, dtype=torch.complex64)
        self.bn1 = ComplexBatchNorm2d(c_out)
        self.act1 = ComplexLeakyReLU(negative_slope)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, dtype=torch.complex64)
        self.bn2 = ComplexBatchNorm2d(c_out)
        self.act2 = ComplexLeakyReLU(negative_slope)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x


class ComplexUpConv(nn.Module):
    """2x2 transposed convolution, stride 2 (the "up-conv 2x2" of Fig. 2's legend)."""

    def __init__(self, channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2, dtype=torch.complex64)

    def forward(self, x):
        return self.up(x)


def pad_to_size(x, target_h, target_w):
    """
    Zero-pad (or centre-crop, if larger) a complex tensor's last two dims to
    (target_h, target_w). Needed because F=513 (odd, from a 1024-point FFT)
    means floor-division max-pooling and exact-doubling up-convs do not always
    land back on the same resolution as their skip connection.
    """
    h, w = x.shape[-2], x.shape[-1]
    if h == target_h and w == target_w:
        return x
    if h > target_h:
        x = x[..., :target_h, :]
        h = target_h
    if w > target_w:
        x = x[..., :target_w]
        w = target_w
    pad_h, pad_w = target_h - h, target_w - w
    if pad_h > 0 or pad_w > 0:
        real = F.pad(x.real, (0, pad_w, 0, pad_h))
        imag = F.pad(x.imag, (0, pad_w, 0, pad_h))
        x = torch.complex(real, imag)
    return x


def align_and_concat(*tensors, dim=1):
    """Pad every tensor to the largest (H, W) among them, then concat along `dim`."""
    target_h = max(t.shape[-2] for t in tensors)
    target_w = max(t.shape[-1] for t in tensors)
    aligned = [pad_to_size(t, target_h, target_w) for t in tensors]
    return torch.cat(aligned, dim=dim)


# =======================================================================================
# 3. The intra-MVDR module (Fig. 3, Eqs. 1-4)
# =======================================================================================

class IntraMVDR(nn.Module):
    """
    T-F mask estimation network + mask-based MVDR filtering with respect to
    *all* N microphones (Sec. 3.1-3.2).

    Inputs:
        x_noisy : (B, N, F, T) complex -- noisy STFT at this level's resolution
        enc_feat: (B, C, F, T) complex -- this level's encoder output feature maps
    Returns:
        z       : (B, N, F, T) complex -- MVDR-filtered STFTs, one per reference mic
        m_s, m_v: (B, F, T) real       -- estimated speech / noise masks
    """

    def __init__(self, enc_channels, n_mics, eps=1e-6, negative_slope=0.1,
                 diag_loading=1e-4):
        super().__init__()
        self.n_mics = n_mics
        self.eps = eps
        self.diag_loading = diag_loading

        # Mask estimation network: (complex) 3x3 conv -> BN -> leaky ReLU -> 1x1 conv -> Sigmoid
        self.mask_conv = nn.Conv2d(enc_channels, enc_channels, kernel_size=3, padding=1, dtype=torch.complex64)
        self.mask_bn = ComplexBatchNorm2d(enc_channels)
        self.mask_act = ComplexLeakyReLU(negative_slope)
        # NOTE: Sigmoid needs a real input; we take the magnitude of the complex
        # features and apply a real-valued 1x1 conv -> 2 channels (M_s, M_v).
        self.mask_out = nn.Conv2d(enc_channels, 2, kernel_size=1)

    def forward(self, x_noisy, enc_feat):
        # x_noisy and enc_feat are pooled from equal-sized tensors at every level so
        # their spatial dims already match in practice; this is just a safety net.
        if x_noisy.shape[-2:] != enc_feat.shape[-2:]:
            target_h = max(x_noisy.shape[-2], enc_feat.shape[-2])
            target_w = max(x_noisy.shape[-1], enc_feat.shape[-1])
            x_noisy = pad_to_size(x_noisy, target_h, target_w)
            enc_feat = pad_to_size(enc_feat, target_h, target_w)

        h = self.mask_act(self.mask_bn(self.mask_conv(enc_feat)))
        masks = torch.sigmoid(self.mask_out(h.abs()))  # (B, 2, F, T)
        m_s, m_v = masks[:, 0], masks[:, 1]            # each (B, F, T)

        B, N, Fbins, T = x_noisy.shape
        x_ft = x_noisy.permute(0, 2, 1, 3)              # (B, F, N, T)

        m_s_c = m_s.unsqueeze(2).to(x_ft.dtype)          # (B, F, 1, T)
        m_v_c = m_v.unsqueeze(2).to(x_ft.dtype)
        x_ft_H = x_ft.conj().transpose(-2, -1)           # (B, F, T, N)

        z_s = m_s.sum(dim=2).clamp_min(self.eps).to(x_ft.dtype).view(B, Fbins, 1, 1)
        z_v = m_v.sum(dim=2).clamp_min(self.eps).to(x_ft.dtype).view(B, Fbins, 1, 1)

        # Eqs. (1)-(2): mask-weighted PSD matrices, computed as batched matmuls
        # instead of an explicit (B,F,T,N,N) outer-product tensor, for memory
        # efficiency -- mathematically identical to the paper's formulation.
        phi_s = ((x_ft * m_s_c) @ x_ft_H) / z_s           # (B, F, N, N)
        phi_v = ((x_ft * m_v_c) @ x_ft_H) / z_v           # (B, F, N, N)

        # Diagonal loading for numerical stability of the inverse (not in the
        # paper's idealized equations, but required for a working implementation).
        # Loading is relative to the per-frequency trace; 1e-4 keeps the solve
        # well-conditioned even with the poor masks of early training, at a
        # negligible cost in beam sharpness.
        eye = torch.eye(N, dtype=phi_v.dtype, device=phi_v.device).view(1, 1, N, N)
        trace_v = torch.diagonal(phi_v, dim1=-2, dim2=-1).sum(-1).real.clamp_min(self.eps)
        loading = (self.diag_loading * trace_v / N).view(B, Fbins, 1, 1).to(phi_v.dtype)
        phi_v = phi_v + loading * eye

        # Eq. (4): h_i(f) = [Phi_v^-1(f) Phi_s(f)]_{:,i} / Trace(Phi_v^-1(f) Phi_s(f))
        M = torch.linalg.solve(phi_v, phi_s)              # Phi_v^{-1} @ Phi_s, (B, F, N, N)
        trace_m = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1)          # (B, F) complex
        trace_m = trace_m + self.eps

        # Eq. (3), computed for all N reference channels i at once:
        #   Z_i(f,t) = h_i(f)^H x(f,t) = [M(f)^H x(f,t)]_i / conj(Trace(M(f)))
        M_H = M.conj().transpose(-2, -1)                  # (B, F, N, N)
        z = (M_H @ x_ft) / trace_m.conj().view(B, Fbins, 1, 1)   # (B, F, N, T)
        z = z.permute(0, 2, 1, 3)                          # (B, N, F, T)
        return z, m_s, m_v


# =======================================================================================
# 4. The full MVDR-embedded U-Net beamformer (Fig. 2)
# =======================================================================================

class MVDRUNetSE(nn.Module):
    """
    The paper's MVDR-embedded U-Net, extended with a separate conditioning
    input: the encoder consumes `n_cond` channels (the noisy mics plus any
    direction-conditioning features), while the intra-MVDR modules and the
    final filter-and-sum operate on the `n_mics` *physical* microphone STFTs
    only. Keeping deterministic conditioning channels out of the PSD matrices
    (Eqs. 1-2) matters: a time-constant steering channel has a rank-deficient
    "noise" covariance that destabilizes the MVDR solve and wastes its output
    channels on non-acoustic signals.
    """

    def __init__(self, n_mics, c1, c2, c3, c4, mvdr_levels=(1, 2, 3, 4),
                 leaky_slope=0.1, mvdr_eps=1e-6, diag_loading=1e-4, n_cond=None):
        super().__init__()
        assert 1 in mvdr_levels, (
            "Level-1 intra-MVDR must be enabled: its output Z is required for the "
            "final reconstruction, Eq. (5)."
        )
        self.n_mics = n_mics
        self.mvdr_levels = set(mvdr_levels)
        N = n_mics
        n_cond = n_cond if n_cond is not None else n_mics
        self.n_cond = n_cond

        # ---- Encoders: (Nc, C1), (Nc+C1, C2), (Nc+C2, C3), (Nc+C3, C4), bottleneck (C4, C4) ----
        self.enc1 = ComplexConvBlock(n_cond, c1, leaky_slope)
        self.enc2 = ComplexConvBlock(n_cond + c1, c2, leaky_slope)
        self.enc3 = ComplexConvBlock(n_cond + c2, c3, leaky_slope)
        self.enc4 = ComplexConvBlock(n_cond + c3, c4, leaky_slope)
        self.enc5 = ComplexConvBlock(c4, c4, leaky_slope)  # bottleneck, no intra-MVDR

        # ---- intra-MVDR modules (operate on the N physical mic channels) ----
        if 1 in self.mvdr_levels:
            self.mvdr1 = IntraMVDR(c1, N, mvdr_eps, leaky_slope, diag_loading)
        if 2 in self.mvdr_levels:
            self.mvdr2 = IntraMVDR(c2, N, mvdr_eps, leaky_slope, diag_loading)
        if 3 in self.mvdr_levels:
            self.mvdr3 = IntraMVDR(c3, N, mvdr_eps, leaky_slope, diag_loading)
        if 4 in self.mvdr_levels:
            self.mvdr4 = IntraMVDR(c4, N, mvdr_eps, leaky_slope, diag_loading)

        # ---- up-convolutions ----
        self.up4 = ComplexUpConv(c4)
        self.up3 = ComplexUpConv(c3)
        self.up2 = ComplexUpConv(c2)
        self.up1 = ComplexUpConv(c1)

        # ---- Decoders: (N+2C_k, C_{k-1}), matching Fig. 2 ----
        d4_in = c4 + c4 + (N if 4 in self.mvdr_levels else 0)
        self.dec4 = ComplexConvBlock(d4_in, c3, leaky_slope)
        d3_in = c3 + c3 + (N if 3 in self.mvdr_levels else 0)
        self.dec3 = ComplexConvBlock(d3_in, c2, leaky_slope)
        d2_in = c2 + c2 + (N if 2 in self.mvdr_levels else 0)
        self.dec2 = ComplexConvBlock(d2_in, c1, leaky_slope)
        d1_in = c1 + c1 + N   # level 1 is always active
        self.dec1 = ComplexConvBlock(d1_in, c1, leaky_slope)

        # ---- final 1x1 conv -> 2N filter weights (Eq. 5) ----
        self.final_conv = nn.Conv2d(c1, 2 * N, kernel_size=1, dtype=torch.complex64)

    def forward(self, x, x_cond=None):
        """
        x      : (B, N, F, T) complex -- noisy multichannel STFT (physical mics)
        x_cond : (B, n_cond, F, T) complex or None -- encoder input including any
                 conditioning channels; defaults to x when no conditioning is used.
        Returns (B, F, T) complex enhanced STFT.
        """
        N = self.n_mics
        if x_cond is None:
            x_cond = x

        # ---- encoder path (conditioned input), with per-level pooled noisy mics
        #      kept alongside for the intra-MVDR modules ----
        x1, c1_in = x, x_cond
        e1 = self.enc1(c1_in)

        x2, c2_in = complex_max_pool2d(x1), complex_max_pool2d(c1_in)
        e2 = self.enc2(align_and_concat(c2_in, complex_max_pool2d(e1)))

        x3, c3_in = complex_max_pool2d(x2), complex_max_pool2d(c2_in)
        e3 = self.enc3(align_and_concat(c3_in, complex_max_pool2d(e2)))

        x4, c4_in = complex_max_pool2d(x3), complex_max_pool2d(c3_in)
        e4 = self.enc4(align_and_concat(c4_in, complex_max_pool2d(e3)))

        e5 = self.enc5(complex_max_pool2d(e4))  # bottleneck

        # ---- intra-MVDR modules, computed from each level's encoder output ----
        z1 = self.mvdr1(x1, e1)[0] if 1 in self.mvdr_levels else None
        z2 = self.mvdr2(x2, e2)[0] if 2 in self.mvdr_levels else None
        z3 = self.mvdr3(x3, e3)[0] if 3 in self.mvdr_levels else None
        z4 = self.mvdr4(x4, e4)[0] if 4 in self.mvdr_levels else None

        # ---- decoder path ----
        pieces4 = [self.up4(e5), e4] + ([z4] if z4 is not None else [])
        d4 = self.dec4(align_and_concat(*pieces4))

        pieces3 = [self.up3(d4), e3] + ([z3] if z3 is not None else [])
        d3 = self.dec3(align_and_concat(*pieces3))

        pieces2 = [self.up2(d3), e2] + ([z2] if z2 is not None else [])
        d2 = self.dec2(align_and_concat(*pieces2))

        pieces1 = [self.up1(d2), e1, z1]
        d1 = self.dec1(align_and_concat(*pieces1))

        w = self.final_conv(d1)                       # (B, 2N, F, T)
        target_h, target_w = x1.shape[-2], x1.shape[-1]
        w = pad_to_size(w, target_h, target_w)
        z1 = pad_to_size(z1, target_h, target_w)

        w_direct, w_mvdr = w[:, :N], w[:, N:2 * N]     # Eq. (5)
        s_hat = (w_direct * x1).sum(dim=1) + (w_mvdr * z1).sum(dim=1)  # (B, F, T)
        return s_hat


# =======================================================================================
# 4b. Steering-conditioned wrapper
# =======================================================================================

class SteeringConditionedMVDRUNet(nn.Module):
    """
    Wraps MVDRUNetSE and conditions it on the target direction via
    *phase-aligned mixture channels* rather than raw steering channels.

    For each mic m and frequency f we form

        x_aligned_m(f, t) = x_m(f, t) * conj(a_m(f))

    i.e. the mixture with the target direction's inter-channel phase delays
    compensated. In this representation, any energy arriving from the target
    direction is phase-identical across all channels (their sum is the
    delay-and-sum beam pointed at the target), while energy from other
    directions stays incoherent -- exactly the spatial contrast the network
    needs. This is a far stronger cue than concatenating the (signal-
    independent, time-constant) steering vector itself, which forces the
    network to *learn* phase comparison from scratch.

    Encoder input: cat([x, x_aligned]) -> 2*N_mics channels.
    The intra-MVDR modules and the final filter-and-sum see only the true
    N_mics mixture channels (see MVDRUNetSE docstring).

    At inference:
        model(x_stft, steering)
        where x_stft   : (B, N_mics, F, T) complex
              steering  : (B, N_mics, F)   complex   (no time dim -- broadcast over T)
    """

    def __init__(self, n_mics, c1, c2, c3, c4, mvdr_levels=(1, 2, 3, 4),
                 leaky_slope=0.1, mvdr_eps=1e-6, diag_loading=1e-4):
        super().__init__()
        self.n_mics = n_mics
        self.unet = MVDRUNetSE(
            n_mics=n_mics,
            c1=c1, c2=c2, c3=c3, c4=c4,
            mvdr_levels=mvdr_levels,
            leaky_slope=leaky_slope,
            mvdr_eps=mvdr_eps,
            diag_loading=diag_loading,
            n_cond=2 * n_mics,   # mixture + phase-aligned mixture
        )

    def forward(self, x, steering):
        """
        x        : (B, N_mics, F, T) complex -- multichannel noisy STFT
        steering : (B, N_mics, F)   complex  -- steering vector (no time dim)

        Returns  : (B, F, T) complex          -- enhanced single-channel STFT
        """
        # Compensate the target direction's phase delays: (B, N, F, 1) conj
        # broadcasts over time. |a|=1 so this is a pure phase rotation.
        x_aligned = x * steering.conj().unsqueeze(-1)
        x_cond = torch.cat([x, x_aligned], dim=1)  # (B, 2N, F, T)
        return self.unet(x, x_cond)




def power_law_compressed_loss(s_hat, s_true, power=0.3, alpha=0.3, eps=1e-8):
    """
    L(S_hat, S) = alpha * || S_hat^p - S^p ||^2 + (1-alpha) * || |S_hat|^p - |S|^p ||^2
    where X^p := |X|^p * exp(j*phase(X)) is the phase-preserving power-law
    compressed complex spectrogram (Braun & Tashev 2021, paper's ref. [29]).
    Squared error is averaged (not summed) over T-F bins so the loss scale is
    independent of segment length and comparable to the SI-SDR term.
    """
    mag_hat = torch.abs(s_hat).clamp_min(eps)
    mag_true = torch.abs(s_true).clamp_min(eps)

    # |X|^p * exp(j phase(X)) == |X|^(p-1) * X
    s_hat_c = (mag_hat ** (power - 1)).to(s_hat.dtype) * s_hat
    s_true_c = (mag_true ** (power - 1)).to(s_true.dtype) * s_true

    complex_term = torch.mean(torch.abs(s_hat_c - s_true_c) ** 2, dim=(-2, -1))
    mag_term = torch.mean((mag_hat ** power - mag_true ** power) ** 2, dim=(-2, -1))

    loss_per_sample = alpha * complex_term + (1 - alpha) * mag_term
    return loss_per_sample.mean()


def si_sdr_db(est, ref, eps=1e-8):
    """
    Scale-invariant SDR in dB, per batch element. est/ref: (B, T) waveforms.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    ref_energy = ref.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    proj = (est * ref).sum(dim=-1, keepdim=True) / ref_energy * ref
    noise = est - proj
    ratio = proj.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(ratio.clamp_min(eps))


# =======================================================================================
# 6. Dataset: wraps the supplied pyroomacoustics simulator
# =======================================================================================

def collect_audio_files(directory, extensions=(".wav", ".flac")):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
    if len(files) == 0:
        raise RuntimeError(
            f"No audio files found under {directory} with extensions {extensions}. "
            f"Point TARGET_AUDIO_DIR / NOISE_AUDIO_DIR at real data before training."
        )
    return sorted(files)


def librispeech_speaker_id(path):
    """
    LibriSpeech layout is .../<speaker>/<chapter>/<spk>-<chap>-<utt>.flac, so
    the speaker id is the grandparent directory name. Falls back to the parent
    directory for other layouts -- worst case the exclusion is just stricter.
    """
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


class SimulatedSEDataset(Dataset):
    """
    Wraps `generate_example` (room simulation) and yields fixed-length training
    segments. The target utterance is energy-cropped to the segment length
    *before* the simulation (inside generate_example), so the room is only ever
    simulated for the audio that is actually trained on -- a large speedup over
    simulating full utterances and cropping afterwards.

    The clips passed to generate_example are all pairwise distinct speakers,
    placed symmetrically (pairwise angular separation, random levels).

    Two modes:
      all_speakers=False (default -- single-target, v4 behaviour):
        one randomly chosen source is the supervision target.
        Returns:
          mixture   : (N_mics, T) float32
          target_ref: (T,) float32           -- chosen speaker at reference mic
          steering  : (N_mics, F) complex64  -- steering for the chosen speaker

      all_speakers=True (v5 -- supervise EVERY speaker in the room):
        Returns:
          mixture   : (N_mics, T) float32
          targets   : (MAX_SOURCES, T) float32          -- each source at ref mic,
                                                           zero-padded past n_src
          steerings : (MAX_SOURCES, N_mics, F) complex64 -- per-source steering,
                                                           zero-padded past n_src
          n_src     : int64 scalar -- how many of the MAX_SOURCES slots are real

    With `deterministic=True` (validation), every index maps to a fixed RNG
    seed, so the validation set is identical across epochs and runs -- val
    metrics become comparable and best-checkpoint selection meaningful.
    """

    def __init__(self, target_files, noise_files, segment_seconds, num_examples,
                 reference_mic, n_fft, fixed_length=True, deterministic=False,
                 all_speakers=False):
        self.target_files = target_files
        self.noise_files = noise_files
        self.noise_speakers = [librispeech_speaker_id(p) for p in noise_files]
        self.segment_seconds = segment_seconds
        self.segment_len = int(round(segment_seconds * DATASET_FS))
        self.num_examples = num_examples
        self.reference_mic = reference_mic
        self.n_fft = n_fft
        self.fixed_length = fixed_length
        self.deterministic = deterministic
        self.all_speakers = all_speakers
        # Mic positions are fixed for the array -- precompute once
        self.mic_positions = get_mic_positions()  # (3, 16)

    def __len__(self):
        return self.num_examples

    def _pick_interferers(self, target_path, k=10):
        """
        Sample k clips for the other sources in the room. ALL speakers must be
        pairwise distinct (not just distinct from the first source): any of
        them can be chosen as the extraction target, and two sources with the
        same voice would make the supervision ambiguous beyond direction.
        """
        target_spk = librispeech_speaker_id(target_path)
        picked, used_speakers, tries = [], {target_spk}, 0
        while len(picked) < k and tries < 200 * k:
            i = random.randrange(len(self.noise_files))
            tries += 1
            if self.noise_files[i] == target_path:
                continue
            spk = self.noise_speakers[i]
            if spk in used_speakers:
                continue
            picked.append(self.noise_files[i])
            used_speakers.add(spk)
        if not picked:  # degenerate pool (e.g. single-speaker dir): give up on exclusion
            picked = random.choices(self.noise_files, k=k)
        return picked

    def __getitem__(self, idx):
        if self.deterministic:
            # Fixed seed per index -> reproducible validation examples.
            random.seed((1000003 * (idx + 1) + 12345) % (2 ** 32))

        target_path = random.choice(self.target_files)
        interferer_paths = self._pick_interferers(target_path, k=10)

        # Simulate only a bit more than one training segment (margin covers the
        # propagation delay from source to array, up to ~0.2s for far sources).
        example = generate_example(
            target_path, interferer_paths,
            max_signal_seconds=self.segment_seconds + 0.5 if self.fixed_length else None,
        )
        mixture = example["mixture"]              # (N_mics, T) float32

        if self.all_speakers:
            return self._item_all_speakers(example, mixture)

        target_reverb = example["target_reverb"]  # (N_mics, T) float32
        azimuth_deg = example["target_azimuth_deg"]  # float

        if self.fixed_length:
            total_len = mixture.shape[-1]
            if total_len > self.segment_len:
                # The pre-simulation energy crop already chose the content;
                # keep the segment where the (delayed) direct sound lands.
                ref_energy = target_reverb[self.reference_mic].pow(2)
                csum = torch.cumsum(ref_energy, dim=0)
                windows = csum[self.segment_len - 1:] - F.pad(csum[:-self.segment_len], (1, 0))
                start = int(torch.argmax(windows))
                mixture = mixture[:, start:start + self.segment_len]
                target_reverb = target_reverb[:, start:start + self.segment_len]
            elif total_len < self.segment_len:
                pad = self.segment_len - total_len
                mixture = F.pad(mixture, (0, pad))
                target_reverb = F.pad(target_reverb, (0, pad))

        target_ref_wav = target_reverb[self.reference_mic]  # (T,)

        # Per-example level normalization: source distances vary by an order of
        # magnitude, so raw mixture levels do too. Scaling mixture and target by
        # the same factor keeps the task unchanged but the loss well-conditioned.
        scale = mixture.std().clamp_min(1e-8)
        mixture = mixture / scale
        target_ref_wav = target_ref_wav / scale

        # Compute the steering vector for the target direction.
        # Shape: (N_mics, n_fft//2+1) complex64
        steering_np = compute_steering_vector(
            azimuth_deg, self.mic_positions, DATASET_FS, self.n_fft
        )
        steering = torch.tensor(steering_np, dtype=torch.complex64)  # (N_mics, F)

        return mixture, target_ref_wav, steering

    def _item_all_speakers(self, example, mixture):
        """v5: one crop/normalization of the mixture, plus per-speaker targets
        and steering vectors for EVERY source, padded to MAX_SOURCES slots."""
        sources = example["sources_reverb"]       # (K, N_mics, T) float32
        azimuths = example["azimuths"]            # K floats
        n_src = sources.shape[0]

        if self.fixed_length:
            total_len = mixture.shape[-1]
            if total_len > self.segment_len:
                # Window with the most TOTAL speech energy (every clip was
                # energy-cropped pre-sim, so all speakers are active in it).
                ref_energy = mixture[self.reference_mic].pow(2)
                csum = torch.cumsum(ref_energy, dim=0)
                windows = csum[self.segment_len - 1:] - F.pad(csum[:-self.segment_len], (1, 0))
                start = int(torch.argmax(windows))
                mixture = mixture[:, start:start + self.segment_len]
                sources = sources[:, :, start:start + self.segment_len]
            elif total_len < self.segment_len:
                pad = self.segment_len - total_len
                mixture = F.pad(mixture, (0, pad))
                sources = F.pad(sources, (0, pad))

        # One common normalization for the mixture and every target
        scale = mixture.std().clamp_min(1e-8)
        mixture = mixture / scale
        targets = sources[:, self.reference_mic, :] / scale     # (K, T)

        steerings = torch.stack([
            torch.tensor(
                compute_steering_vector(az, self.mic_positions, DATASET_FS, self.n_fft),
                dtype=torch.complex64,
            )
            for az in azimuths
        ])                                                       # (K, N_mics, F)

        # Pad the speaker axis to MAX_SOURCES so the default collate works;
        # padded slots are never used (masked by n_src in the training loop).
        if n_src < MAX_SOURCES:
            pad_k = MAX_SOURCES - n_src
            targets = F.pad(targets, (0, 0, 0, pad_k))
            steerings = torch.cat([
                steerings,
                torch.zeros(pad_k, *steerings.shape[1:], dtype=torch.complex64),
            ])

        return mixture, targets, steerings, torch.tensor(n_src, dtype=torch.int64)


# =======================================================================================
# 7. STFT helper
# =======================================================================================

def compute_stft(waveform, n_fft, hop_length, win_length, window):
    """waveform: (..., T) real -> (..., F, T') complex."""
    orig_shape = waveform.shape[:-1]
    wav_flat = waveform.reshape(-1, waveform.shape[-1])
    spec = torch.stft(
        wav_flat, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True,
    )
    return spec.reshape(*orig_shape, spec.shape[-2], spec.shape[-1])



# =======================================================================================
# 9. Live loss plotting
# =======================================================================================

import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- saves to file instead of opening a window
import matplotlib.pyplot as plt

PLOT_PATH = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
LOSS_LOG_PATH = os.path.join(CHECKPOINT_DIR, "loss_log.json")


def save_loss_plot(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr=()):
    """
    Saves the live loss curve to PLOT_PATH.
    - Top panel: per-step training loss (updated every LOG_EVERY steps)
    - Middle panel: per-epoch train vs val loss
    - Bottom panel: per-epoch validation SI-SDR (the metric that matters)
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    # Top: step-level train loss
    ax = axes[0]
    if step_losses:
        steps, losses = zip(*step_losses)
        ax.plot(steps, losses, color="steelblue", linewidth=0.8)
    ax.set_title("Training Loss (per step)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # Bottom: epoch-level train vs val
    ax = axes[1]
    if epoch_train_losses:
        epochs_t, tlosses = zip(*epoch_train_losses)
        ax.plot(epochs_t, tlosses, marker="o", label="Train", color="steelblue")
    if epoch_val_losses:
        epochs_v, vlosses = zip(*epoch_val_losses)
        ax.plot(epochs_v, vlosses, marker="s", label="Val", color="tomato")
    ax.set_title("Epoch Train vs Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom: epoch-level validation SI-SDR
    ax = axes[2]
    if epoch_val_sisdr:
        epochs_s, sisdrs = zip(*epoch_val_sisdr)
        ax.plot(epochs_s, sisdrs, marker="^", color="seagreen")
    ax.set_title("Validation SI-SDR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SI-SDR (dB)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    plt.close(fig)


def load_loss_log():
    """Load existing loss history from disk (used when resuming)."""
    if os.path.exists(LOSS_LOG_PATH):
        import json
        with open(LOSS_LOG_PATH) as f:
            data = json.load(f)
        return (
            [tuple(x) for x in data.get("step_losses", [])],
            [tuple(x) for x in data.get("epoch_train_losses", [])],
            [tuple(x) for x in data.get("epoch_val_losses", [])],
            [tuple(x) for x in data.get("epoch_val_sisdr", [])],
        )
    return [], [], [], []


def save_loss_log(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr):
    import json
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(LOSS_LOG_PATH, "w") as f:
        json.dump({
            "step_losses": step_losses,
            "epoch_train_losses": epoch_train_losses,
            "epoch_val_losses": epoch_val_losses,
            "epoch_val_sisdr": epoch_val_sisdr,
        }, f)


# =======================================================================================
# 10. Training / validation loop (with resume + live plot)
# =======================================================================================

def all_reduce_gradients(model, world_size):
    """
    Average gradients across ranks (hand-rolled data parallelism). Used instead
    of DistributedDataParallel because the all-speaker pass runs a *variable*
    number of backward calls per rank (max n_src in the local batch differs),
    which would desynchronize DDP's per-backward collectives; here exactly one
    collective per parameter runs per optimization step on every rank. Complex
    grads are all-reduced through a real view (elementwise complex sum == sum
    of the real/imag parts).
    """
    if world_size <= 1:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        dist.all_reduce(torch.view_as_real(g) if g.is_complex() else g)
        g /= world_size


def train_worker(rank, world_size):
    # Line-buffer stdout so progress lines appear immediately even when the
    # output is redirected to a file (nohup / background runs).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    # Allocator config must be set before this process's first CUDA call.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = f"cuda:{GPUS[rank]}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", rank=rank, world_size=world_size,
                                device_id=torch.device(device))

    target_files = collect_audio_files(TARGET_AUDIO_DIR)
    noise_files = collect_audio_files(NOISE_AUDIO_DIR)

    shuffled = target_files.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    val_targets, train_targets = shuffled[:n_val], shuffled[n_val:]

    # Each rank simulates its own share of the virtual epoch (fresh random
    # rooms per rank -- per-rank RNG seeds are decorrelated below).
    train_ds = SimulatedSEDataset(train_targets, noise_files, SEGMENT_SECONDS,
                                   NUM_TRAIN_EXAMPLES // world_size, REFERENCE_MIC,
                                   N_FFT, fixed_length=True, all_speakers=True)
    # Validation: the SAME deterministic examples every epoch, sharded across
    # ranks by index (Subset keeps the original index -> same seed -> same
    # example regardless of world size); metrics are all-reduced afterwards.
    val_ds_full = SimulatedSEDataset(val_targets, noise_files, SEGMENT_SECONDS,
                                      NUM_VAL_EXAMPLES, REFERENCE_MIC, N_FFT,
                                      fixed_length=True, deterministic=True,
                                      all_speakers=True)
    val_ds = Subset(val_ds_full, list(range(rank, NUM_VAL_EXAMPLES, world_size)))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        drop_last=True, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        drop_last=False, persistent_workers=NUM_WORKERS > 0, pin_memory=True,
    )

    # Identical weight init on every rank (same seed before construction)...
    torch.manual_seed(SPLIT_SEED)
    model = SteeringConditionedMVDRUNet(N_MICS, C1, C2, C3, C4, MVDR_LEVELS,
                                        LEAKY_SLOPE, MVDR_EPS, MVDR_DIAG_LOADING).to(device)
    # ...then decorrelate the ranks' data pipelines (DataLoader worker seeds
    # derive from this process's torch RNG).
    torch.manual_seed(100_003 + rank)
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model parameters: {n_params:,} | world_size={world_size} "
              f"(per-GPU batch {BATCH_SIZE}, global batch {BATCH_SIZE * world_size})")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INITIAL)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[LR_DECAY_EPOCH, LR_DECAY_EPOCH + 30], gamma=LR_DECAY_FACTOR
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Resume logic (every rank loads the same file -> identical state) ----
    start_epoch = 1
    best_val_sisdr = float("-inf")
    resume_path = os.path.join(CHECKPOINT_DIR, "last.pt")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_sisdr = ckpt.get("best_val_sisdr", float("-inf"))
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            for _ in range(ckpt["epoch"]):
                scheduler.step()
        if is_main:
            print(f"Resumed from {resume_path} at epoch {start_epoch}, "
                  f"best_val_sisdr={best_val_sisdr:.2f} dB, "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")
    elif INIT_FROM and os.path.exists(INIT_FROM):
        # Warm start: model weights only (fresh optimizer/scheduler/metrics).
        src = torch.load(INIT_FROM, map_location=device)
        model.load_state_dict(src["model"])
        if is_main:
            print(f"No {CHECKPOINT_DIR} checkpoint; warm-started model weights from "
                  f"{INIT_FROM} (epoch {src.get('epoch', '?')}).")
    else:
        if is_main:
            print("No checkpoint found, starting from scratch.")

    # ---- Load existing loss history (for plot continuity on resume) ----
    step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr = load_loss_log()
    global_step = step_losses[-1][0] if step_losses else 0

    window = torch.hann_window(WIN_LENGTH, device=device)

    def all_speaker_pass(mixture, targets, steerings, n_src, backward=False):
        """
        One pass supervising EVERY speaker of every mixture in the batch.

        The mixture STFT is computed once; the model then runs once per speaker
        slot k, on the batch rows that actually have a k-th speaker. Slot losses
        are weighted by their pair counts so the total equals the mean over all
        (mixture, speaker) pairs. With backward=True each slot's scaled loss is
        backpropagated immediately (gradient accumulation), so peak memory
        stays at single-forward level regardless of the speaker count.

        mixture  : (B, N_mics, T) float
        targets  : (B, MAX_SOURCES, T) float, zero-padded past n_src
        steerings: (B, MAX_SOURCES, N_mics, F) complex, zero-padded past n_src
        n_src    : (B,) int64 -- real speaker count per mixture
        Returns (mean_loss, mean_sisdr_db, n_pairs).
        """
        x_stft = compute_stft(mixture, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
        total = int(n_src.sum().item())
        loss_sum, sisdr_sum = 0.0, 0.0
        for k in range(targets.shape[1]):
            mask = n_src > k
            if not bool(mask.any()):
                break  # n_src > k is monotone in k: no later slot is populated
            tgt = targets[mask, k]                       # (b_k, T)
            s_hat = model(x_stft[mask], steerings[mask, k])
            s_true = compute_stft(tgt, N_FFT, HOP_LENGTH, WIN_LENGTH, window)
            loss_spec = power_law_compressed_loss(s_hat, s_true, COMPRESSION_POWER, LOSS_ALPHA)
            wav_hat = torch.istft(
                s_hat, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                window=window, center=True, length=tgt.shape[-1],
            )
            sisdr = si_sdr_db(wav_hat, tgt)              # (b_k,)
            loss_k = loss_spec + SISDR_LOSS_WEIGHT * (-sisdr.mean())
            cnt = int(mask.sum().item())
            if backward:
                (loss_k * (cnt / total)).backward()
            loss_sum += loss_k.item() * cnt
            sisdr_sum += sisdr.sum().item()
        return loss_sum / total, sisdr_sum / total, total

    if is_main:
        print(f"Epoch plan: {len(train_loader)} steps/GPU/epoch; each step trains "
              f"every speaker of {BATCH_SIZE} rooms on each of {world_size} GPU(s). "
              f"Logging step 1 and then every {LOG_EVERY} steps.")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t_epoch = time.time()
        for step, (mixture, targets, steerings, n_src) in enumerate(train_loader):
            mixture = mixture.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            steerings = steerings.to(device, non_blocking=True)
            n_src = n_src.to(device)

            optimizer.zero_grad()
            loss, _, _ = all_speaker_pass(mixture, targets, steerings, n_src, backward=True)
            # Average gradients across GPUs (exactly one collective per param
            # per step on every rank), then clip identically everywhere.
            all_reduce_gradients(model, world_size)
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            running_loss += loss
            global_step += 1

            if is_main and (step == 0 or (step + 1) % LOG_EVERY == 0):
                avg_loss = running_loss / (step + 1)
                s_per_step = (time.time() - t_epoch) / (step + 1)
                print(f"Epoch {epoch} Step {step + 1}/{len(train_loader)} "
                      f"Loss {avg_loss:.4f} ({s_per_step:.1f} s/step)")
                if (step + 1) % LOG_EVERY == 0:
                    # Record and plot (rank 0's local loss -- representative)
                    step_losses.append((global_step, avg_loss))
                    save_loss_log(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr)
                    save_loss_plot(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr)

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss_sum, val_sisdr_sum, val_pairs = 0.0, 0.0, 0
        with torch.no_grad():
            for mixture, targets, steerings, n_src in val_loader:
                mixture = mixture.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                steerings = steerings.to(device, non_blocking=True)
                n_src = n_src.to(device)
                loss, sisdr, pairs = all_speaker_pass(mixture, targets, steerings, n_src)
                val_loss_sum += loss * pairs
                val_sisdr_sum += sisdr * pairs
                val_pairs += pairs

        # Combine the ranks' validation shards into global means.
        stats = torch.tensor([val_loss_sum, val_sisdr_sum, float(val_pairs)],
                             dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(stats)
        total_pairs = max(1.0, stats[2].item())
        val_loss = stats[0].item() / total_pairs
        val_sisdr = stats[1].item() / total_pairs

        if is_main:
            print(f"==> Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_sisdr={val_sisdr:.2f} dB (over {int(total_pairs)} speaker extractions) "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

            # Record epoch losses and update plot
            epoch_train_losses.append((epoch, train_loss))
            epoch_val_losses.append((epoch, val_loss))
            epoch_val_sisdr.append((epoch, val_sisdr))
            save_loss_log(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr)
            save_loss_plot(step_losses, epoch_train_losses, epoch_val_losses, epoch_val_sisdr)
            print(f"    Loss curve saved to {PLOT_PATH}")

        # "best" = highest validation SI-SDR; all ranks track it (identical
        # value post-allreduce), rank 0 writes the files.
        is_best = val_sisdr > best_val_sisdr
        if is_best:
            best_val_sisdr = val_sisdr
        if is_main:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_sisdr": best_val_sisdr,
            }
            torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "last.pt"))
            if is_best:
                torch.save(ckpt, os.path.join(CHECKPOINT_DIR, "best.pt"))
        if world_size > 1:
            dist.barrier()

    if is_main:
        print("Training complete.")
        print(f"Best val SI-SDR: {best_val_sisdr:.2f} dB")
        print(f"Final loss curve: {PLOT_PATH}")
    if world_size > 1:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Refuse to double-launch: two concurrent runs fight over the GPUs, the
    # checkpoint dir, and (formerly) a fixed NCCL port -- which is exactly what
    # made a run appear "frozen at startup" once. Stale locks from crashed
    # runs are detected via /proc and ignored.
    lock_path = os.path.join(CHECKPOINT_DIR, "train.pid")
    if os.path.exists(lock_path):
        old_pid = open(lock_path).read().strip()
        if old_pid.isdigit() and os.path.exists(f"/proc/{old_pid}"):
            raise SystemExit(
                f"Another training run (pid {old_pid}) is already active according "
                f"to {lock_path}. Kill it first (kill {old_pid}) or, if that pid is "
                f"not actually train.py, delete the lock file and relaunch."
            )
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    # Fresh rendezvous port per launch -- never collides with an older run.
    os.environ["MASTER_PORT"] = str(_free_port())

    try:
        world_size = len(GPUS)
        if world_size > 1:
            torch.multiprocessing.spawn(train_worker, args=(world_size,),
                                        nprocs=world_size, join=True)
        else:
            train_worker(0, 1)
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


if __name__ == "__main__":
    train()