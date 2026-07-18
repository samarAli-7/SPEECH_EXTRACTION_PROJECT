# Direction-Conditioned Target Speaker Extraction

Extract one speaker out of a crowded room by pointing at them: you give the
system an azimuth, it gives you back that person's speech. A 16-microphone
planar array feeds an MVDR-embedded complex U-Net conditioned on a steering
vector, and when two people stand too close together in angle for direction
alone to tell them apart, a measured fallback takes over — beamform, blindly
separate the cluster, then identify the right stream by speaker embedding.

Everything below is measured on this repository's code, not quoted from the
literature. Every table has the command that reproduces it.

---

## Table of contents

1. [The task](#1-the-task)
2. [Data simulation](#2-data-simulation)
3. [Model](#3-model)
4. [Training](#4-training)
5. [Where direction stops working](#5-where-direction-stops-working)
6. [The fallback pipeline](#6-the-fallback-pipeline)
7. [Design decisions, and what they cost](#7-design-decisions-and-what-they-cost)
8. [When the fallback engages](#8-when-the-fallback-engages)
9. [End-to-end results](#9-end-to-end-results)
10. [Reproducing everything](#10-reproducing-everything)
11. [Repository map](#11-repository-map)
12. [Limitations](#12-limitations)

---

## 1. The task

Given a 16-channel mixture of up to 11 simultaneous speakers and one azimuth
θ, output the speech of whoever is standing at θ.

Direction is the *only* cue the model is allowed to use. Speakers are not
distinguishable by loudness (every source is level-randomised independently),
by role (any speaker can be the target), or by position in the input (sources
are placed symmetrically). If two speakers shared a direction the task would
be ill-posed — which is exactly the failure mode Sections 5–8 measure and fix.

Base architecture and training regime follow *"An MVDR-Embedded U-Net
Beamformer for Effective and Robust Multichannel Speech Enhancement"* (Lee,
Patel, Yang, Shen, Jin — ICASSP 2024), adapted from noise suppression to
direction-conditioned extraction.

---

## 2. Data simulation

Rooms are simulated on the fly with `pyroomacoustics` (`dataset.py`).

| Property | Value | Why |
|---|---|---|
| Array | 4×4 planar grid, 16 mics | as in the source simulator |
| Spacing | **0.10 m** (0.30 m aperture) | doubled from 0.05 m; beamwidth ∝ λ/aperture, and the old 15 cm aperture was the binding constraint on angular resolution |
| Sample rate | 16 kHz | — |
| Reverberation | **none** (`max_order=0`) | direct path only; the target is the anechoic direct-path signal at the reference mic |
| Segment length | 5.0 s | |
| Sources per room | 2–11 | uniformly drawn |
| Min. source distance | 3.0 m | keeps the far-field plane-wave steering model valid (far-field distance grows with aperture²) |
| Pairwise angular separation | ≥ 15° | below this, "extract the speaker at θ" has no unique answer |
| Per-source level | U(−6, +6) dB at the mics | removes level as a shortcut for identifying the target |
| Speakers | LibriSpeech, pairwise distinct, energy-cropped | wrong-speaker and silent-target supervision both removed |

Two details matter more than they look:

**Energy cropping before simulation.** LibriSpeech utterances have long silent
heads and tails. Cropping to a speech-dense window *before* the room simulation
both avoids training on silent targets and makes data generation ~7× faster.

**Level randomisation after simulation.** Each source is rescaled to a random
received level *after* propagation, which undoes the 1/r spread from random
distances. Without this, distant speakers are inaudible and the model can cheat
by learning "the target is the loud one."

The steering vector for azimuth θ is the far-field phase delay per mic and
frequency, a_m(f) = exp(−j2πf·τ_m), τ_m = (d_m·u)/c.

---

## 3. Model

A complex-valued U-Net (5 encoder levels, 4 decoder levels) with **intra-MVDR
modules embedded at levels 1–4**. Each module estimates speech/noise masks from
the encoder's own feature maps, builds PSD matrices, solves for an MVDR filter
with respect to every microphone as reference, and hands the resulting
filtered spectrograms to the decoder as extra spatial features. The output is a
weighted filter-and-sum over the noisy STFTs and the level-1 MVDR-filtered
STFTs.

| Component | Setting |
|---|---|
| Channel widths C1–C4 | 32, 64, 64, 64 |
| MVDR levels | 1, 2, 3, 4 |
| STFT | 1024-point FFT, 256 hop, Hann |
| Blocks | 2 × (complex 3×3 conv → complex BN → complex leaky ReLU, slope 0.1) |
| Complex BN | full covariance whitening (Trabelsi et al.) |
| MVDR diagonal loading | 1e-4 (relative), for a stable solve |
| Reference mic | index 5 (a centre-most element) |
| Parameters | ~5.2 M (20.6 MB checkpoint) |

**Steering conditioning.** Rather than concatenating the raw steering vector,
the mixture is *phase-aligned* against it and both are fed in:

```python
x_aligned = x * steering.conj().unsqueeze(-1)
x_cond    = torch.cat([x, x_aligned], dim=1)
return self.unet(x, x_cond)
```

After alignment, energy arriving from θ is phase-coherent across channels while
everything else is not, so the network sees the direction cue directly instead
of having to infer it. The intra-MVDR modules and the final filter-and-sum
still operate on the 16 *physical* channels only.

Deviations from the paper (it does not pin these down numerically) are marked
with `# NOTE:` comments in `train.py`: leaky-ReLU slope, how a complex feature
map becomes a real [0,1] mask, diagonal loading, complex max-pooling, reference
mic index, and gradient clipping.

---

## 4. Training

### 4.1 What each revision changed

| Rev | Change | Reason |
|---|---|---|
| v2 | Phase-aligned steering conditioning; bin-averaged loss + SI-SDR term; deterministic validation; best-checkpoint by val SI-SDR | direction conditioning that actually works |
| v3 | Anechoic (`max_order=0`); 5 s segments | removes reverberation from the problem |
| v4 | **Any-speaker target**: all sources statistically identical, one chosen at random for supervision; spacing 0.05 → 0.10 m | kills every non-directional shortcut |
| v5 | **All-speaker supervision**: every room trains extraction of *every* speaker in it; dual-GPU | grades the model on extracting everyone, not a lucky one |

### 4.2 Loss

```
L = powerlaw_compressed(Ŝ, S)  +  0.05 · (−SI-SDR(ŝ, s))
```

Power-law compression uses exponent 0.3 with α = 0.3 on the complex
(phase-aware) term and 0.7 on the magnitude term. The SI-SDR term is computed
on the iSTFT waveform and keeps the objective aligned with how the output is
actually judged.

### 4.3 Regime

| Setting | Value |
|---|---|
| Optimiser | Adam, lr 1e-3 |
| LR schedule | MultiStepLR, ×0.1 at epochs 25 and 55 |
| Epochs | 80 |
| Batch size | 4 per GPU (measured: 11.2 GiB peak; 6 OOMs on 16 GB) |
| Rooms per epoch | 3000 train / 300 val |
| Gradient clipping | 5.0 |
| Hardware | 2 × RTX 5060 Ti 16 GB |

### 4.4 All-speaker supervision (v5)

A v4 step supervised one speaker per room. A v5 step runs the model once per
speaker slot, up to the batch's largest speaker count, accumulating per-speaker
gradients weighted by `count/total` — verified equivalent to the joint mean
loss to 9.2e-8 relative error.

This costs ~6.3 s/step against ~1 s for v4, but the accounting favours it:

| | rooms/epoch | speakers/room | extraction problems/epoch |
|---|---:|---:|---:|
| v4 | 8000 | 1 | 8,000 |
| v5 | 3000 | ~6.5 | **~19,500** |

So a v5 epoch carries 2.4× the supervision for roughly the same wall clock
(~45 min including validation).

**Multi-GPU.** One process per GPU (`torch.multiprocessing.spawn` + NCCL) with
hand-rolled gradient averaging. `DistributedDataParallel` is deliberately *not*
used: per-rank backward counts vary with each batch's maximum speaker count,
which desynchronises DDP's collectives and deadlocks. The hand-rolled
`all_reduce_gradients` (complex grads via `torch.view_as_real`) was verified
bit-identical across ranks with deliberately unequal backward counts (2 vs 4;
max parameter difference exactly 0.0). Validation is sharded by index and
all-reduced, so the reported number matches a single-GPU evaluation.

### 4.5 Progression

Validation SI-SDR, mean over held-out rooms (v5 = mean over *all* speakers of
all rooms, a strictly harder metric than v4's):

| Epoch | v4 (one target/room) | v5 (all speakers, warm-started from v4) |
|---:|---:|---:|
| 1 | 4.98 | 9.69 |
| 2 | 7.51 | 10.29 |
| 3 | 8.42 | 10.26 |
| 4 | 9.75 | 10.18 |
| 5 | 9.99 | 10.37 |
| 6 | 10.99 | **10.94** |
| 7 | 11.04 | — |
| 8 | 10.97 | — |
| 9 | **11.32** | — |

All results in this document use `checkpoints_v5/best.pt` (epoch 6, val SI-SDR
10.94 dB). Training was stopped early; the numbers below should improve with
more epochs, and the thresholds in Section 8 should be regenerated when it does.

---

## 5. Where direction stops working

The model separates purely by azimuth, so its resolution is bounded by the
array beamwidth. `analyze_angular_safety.py` walks an interferer towards the
target and scores the result. All sources at 0 dB SIR, 24 trials per point.

`SIR_out` is the target-to-near-interferer energy ratio in the output — a
direct measure of how much of the neighbour leaks in.

| Separation | SI-SDR (2 spk) | SIR_out (2 spk) | SI-SDR (6 spk) | SIR_out (6 spk) |
|---:|---:|---:|---:|---:|
| 2°   |  0.26 dB |  1.9 dB |  0.18 dB |  0.6 dB |
| 4°   |  3.18 dB |  7.2 dB | −0.00 dB |  0.7 dB |
| 6°   |  7.30 dB | 12.4 dB |  1.16 dB |  2.7 dB |
| 8°   | 11.31 dB | 19.9 dB |  3.57 dB |  6.2 dB |
| 10°  | 14.08 dB | 27.3 dB |  5.31 dB |  9.1 dB |
| 12°  | 16.92 dB | 37.4 dB |  7.67 dB | 12.8 dB |
| 15°  | 18.30 dB | 34.6 dB | 10.76 dB | 19.9 dB |
| 20°  | 20.13 dB | 42.9 dB | 13.64 dB | 27.0 dB |
| 25°  | 21.21 dB | 41.8 dB | 14.86 dB | 37.1 dB |
| 30°  | 21.40 dB | 39.3 dB | 15.69 dB | 35.8 dB |
| 60°  | 22.99 dB | 47.7 dB | 16.74 dB | 42.8 dB |
| 90°  | 23.57 dB | 45.1 dB | 17.42 dB | 43.1 dB |
| 120° | 22.94 dB | 43.0 dB | 16.99 dB | 44.2 dB |

Two readings of "safe":

| Criterion | 2 speakers | 6 speakers |
|---|---:|---:|
| Uncrowded plateau | 22.99 dB | 16.99 dB |
| Full quality (within 3 dB of plateau) | **20°** | **25°** |
| ASR-reliable (SI-SDR ≥ 10 dB) | **8°** | **15°** |

**Room density costs as much as angle.** Six speakers instead of two costs ~6 dB
at every separation and pushes the ASR-reliable threshold from 8° to 15°. This
is why the fallback trigger in Section 8 is a function of both.

**Array orientation does not matter.** The 4×4 grid has a 0.30 m aperture along
a row but 0.42 m along the diagonal, so resolution might plausibly vary with
azimuth. `analyze_azimuth_dependence.py` finds a ~5 dB spread across azimuths,
but the best and worst angles disagree between separations (best at 70° for a
10° separation, worst at 56° for a 20° one), so it is scene variance rather
than a systematic effect. No correction is applied.

---

## 6. The fallback pipeline

When a neighbour sits inside the trigger cone (`fallback_separation.py`):

```
16-ch mixture
      │
      ├─► [1] direction-conditioned MVDR-UNet, steered at θ
      │        → everyone outside the cluster cone is suppressed;
      │          what remains is a 2- or 3-speaker mixture,
      │          no matter how many people are in the room
      │
      ├─► [2] SepFormer   (2-speaker model if one neighbour is in the cone,
      │                    3-speaker model if two or more)
      │        → K streams, rebuilt full-band by masking the beamformed signal
      │
      └─► [3] ECAPA-TDNN embeddings, cosine-compared against the target's
               enrolment clip → report the closest stream
```

| Stage | Model | Runs at |
|---|---|---|
| Beamform | this repo's `SteeringConditionedMVDRUNet` | 16 kHz |
| Separate (2-spk) | `speechbrain/sepformer-wsj02mix` | 8 kHz internally |
| Separate (3-spk) | `speechbrain/sepformer-wsj03mix` | 8 kHz internally |
| Identify | `speechbrain/spkrec-ecapa-voxceleb` | 16 kHz |

Cost: **~1.7 s on CPU** for a 5-second segment, paid only when the geometry
demands it. Models load lazily, so the normal path pays nothing.

---

## 7. Design decisions, and what they cost

Every choice below was benchmarked rather than assumed
(`benchmark_fallback.py`, 12 trials/point, 2-speaker cluster, median SI-SDR).

### 7.1 Beamform first — not optional

Feeding SepFormer the raw reference-mic mixture instead of the beamformed
signal:

| Separation | beamformed → SepFormer | raw mixture → SepFormer |
|---:|---:|---:|
| 2°  |  9.33 dB | −6.18 dB |
| 6°  | 10.25 dB | −5.78 dB |
| 12° | 12.93 dB | −5.53 dB |
| 20° | 13.02 dB | −5.39 dB |

SepFormer is trained on 2- and 3-speaker mixtures; a room with six people is far
outside that. The beamformer is what reduces any room to a problem the
separator was actually trained for. Identification accuracy shows the same
story — 100% on beamformed input, 42–83% on the raw mixture.

### 7.2 The 8 kHz separator beats the 16 kHz one

| Separation | `wsj02mix` (8 kHz) + mask | `whamr16k` (16 kHz) + mask |
|---:|---:|---:|
| 2°  |  9.33 dB |  9.21 dB |
| 6°  | 10.25 dB |  9.94 dB |
| 12° | **12.93 dB** | 10.68 dB |
| 15° | **13.33 dB** | 11.84 dB |
| 20° | 13.02 dB | **14.87 dB** |
| 30° | 10.63 dB | **16.42 dB** |

`whamr16k` wins at wide separations — but the fallback never runs there. Across
the range where it *does* run, `wsj02mix` wins, so it is the default.

### 7.3 Mask reconstruction restores the lost band

The separated streams are used only to build a time-frequency ratio mask, which
is applied to the **full-band beamformed signal**. Bins above 4 kHz (empty for
an 8 kHz separator) inherit the mask of the highest bin carrying energy.

| Separation | mask reconstruction | separated stream as-is |
|---:|---:|---:|
| 2°  |  **9.33 dB** |  8.64 dB |
| 4°  |  **9.39 dB** |  7.59 dB |
| 12° | **12.93 dB** | 11.42 dB |
| 15° | **13.33 dB** | 10.70 dB |

This is what lets an 8 kHz separator run without throwing away the 4–8 kHz band.

### 7.4 Identification: enrolment matters at small angles

Fraction of scenes where ECAPA picks the best available stream:

| Separation | vs. enrolment clip | vs. beamformed signal |
|---:|---:|---:|
| 2°  | **1.00** | 0.42 |
| 4°  | **1.00** | 0.67 |
| 6°  | 1.00 | 0.92 |
| 8°–30° | 0.92–1.00 | 0.92–1.00 |

With a separate utterance of the target speaker as reference, identification is
essentially solved. Without one — using the beamformed signal itself as the
reference — it holds above 6° but collapses at 2–4°, precisely where the
beamformed signal no longer favours the target enough to identify it. The app
uses each speaker's own clip whenever it has one and falls back gracefully.

### 7.5 A cleverer gate was tried, and rejected

The tempting alternative to an angular threshold: run both paths, embed both
outputs, keep whichever ECAPA finds closer to the enrolment. No ground truth
needed. `benchmark_arbitration.py`, 16 trials/point, median SI-SDR:

| Separation | always-direct | always-fallback | **angle gate** | ECAPA arbitration | oracle |
|---:|---:|---:|---:|---:|---:|
| 4°  |  0.39 |  10.54 | **10.54** |  9.99 | 10.54 |
| 8°  |  4.76 |  12.51 | **12.51** | 12.51 | 12.51 |
| 12° |  9.52 |  13.25 | **13.25** | 11.84 | 13.46 |
| 15° | 10.64 |  13.96 | **13.96** | 11.21 | 14.47 |
| 20° | 13.70 |  10.67 | **13.70** | 13.67 | 13.94 |

Arbitration loses to the plain angular gate everywhere. Near the boundary it
identifies the better output only 38–50% of the time — no better than chance —
because ECAPA similarity measures speaker identity, not signal quality, and
both candidates contain the right speaker. The gate stays.

---

## 8. When the fallback engages

The crossover where the fallback overtakes the direct path **moves with room
density**: every extra speaker degrades the beamformer, while the fallback only
ever sees the cluster and stays roughly flat at 8–14 dB.

`benchmark_density.py`, 16 trials/point, cells are `direct / fallback` median
SI-SDR (dB). The trigger is the widest separation at which the fallback still
wins a majority of trials, counting up from the smallest.

**One near neighbour → 2-speaker pipeline**

| Speakers in room | 6° | 10° | 12° | 15° | 20° | 25° | **Trigger** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  2 | 8.6 / 12.5 | 15.0 / 14.1 | 14.7 / 13.0 | 18.3 / 11.3 | 19.8 / 11.0 | 21.8 / 8.9 | **8.0°** |
|  3 | 2.9 / 9.4 | 14.1 / 12.0 | 12.4 / 13.9 | 17.2 / 15.4 | 19.6 / 14.6 | 20.9 / 10.3 | **8.0°** |
|  4 | 3.3 / 10.8 | 8.8 / 12.4 | 13.3 / 11.6 | 15.9 / 11.9 | 16.6 / 12.1 | 17.3 / 11.2 | **11.0°** |
|  6 | 2.0 / 10.4 | 5.8 / 12.0 | 7.3 / 11.9 | 12.0 / 13.9 | 13.1 / 10.3 | 14.0 / 12.0 | **17.5°** |
| 10 | 0.0 / 7.5 | 1.0 / 7.7 | 2.3 / 8.7 | 5.3 / 9.1 | 3.7 / 9.5 | 9.2 / 8.7 | **25.0°** |

**Two or more near neighbours → 3-speaker pipeline**

| Speakers in room | 6° | 10° | 12° | 15° | 20° | 25° | **Trigger** |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  3 | 1.0 / 7.9 | 7.0 / 9.8 | 11.4 / 12.5 | 14.4 / 12.9 | 17.1 / 9.8 | 17.5 / 17.6 | **13.5°** |
|  4 | −1.7 / 6.3 | 4.0 / 10.3 | 5.3 / 10.4 | 11.5 / 12.7 | 14.9 / 14.4 | 16.1 / 16.6 | **17.5°** |
|  5 | −1.2 / 7.6 | 1.7 / 7.8 | 4.5 / 8.6 | 7.5 / 10.0 | 11.6 / 10.1 | 14.6 / 15.1 | **25.0°** |
|  7 | −2.3 / 6.8 | −0.0 / 7.6 | 1.7 / 9.1 | 5.3 / 10.7 | 8.3 / 11.0 | 10.0 / 10.5 | **25.0°** |
| 11 | −3.3 / 6.0 | −1.9 / 6.4 | −1.0 / 5.2 | 1.3 / 6.5 | 2.0 / 6.6 | 5.3 / 7.8 | **25.0°** |

Triggers are stored in `fallback_thresholds.json` and interpolated over the
speaker count at runtime. Concretely: **12° of separation needs no help in a
3-speaker room but does need it in a 7-speaker room** (interpolated trigger
19.4°). A single fixed threshold over-fires in sparse rooms and under-fires in
crowded ones — this was a real bug, caught by `test_close_speakers.py` firing
the fallback at 15° in a sparse room where the beamformer was already
delivering 14.6 dB.

25.0° is the sweep ceiling and also the cluster cone: a speaker further away
than that is not a cluster member, so the fallback does not apply.

---

## 9. End-to-end results

`test_close_speakers.py` — five random scenes per separation (voices,
orientation and far-speaker placement redrawn each time), 5 speakers in the
room, full app inference path:

| Separation | Mixture | Beamformer | **Final** | Path taken |
|---:|---:|---:|---:|---|
|  3° | −5.9 dB |  0.6 dB | **11.2 dB** | fallback |
|  6° | −5.8 dB |  3.9 dB | **12.5 dB** | fallback |
| 10° | −6.0 dB |  7.0 dB | **11.5 dB** | fallback |
| 15° | −5.8 dB | 12.9 dB | 12.9 dB | direct |
| 30° | −8.0 dB | 16.8 dB | 16.8 dB | direct |

Below the trigger the fallback adds **4–11 dB** and lifts scenes that were
unusable (0–4 dB) past the ~10 dB mark where ASR becomes reliable. Above it,
nothing changes — the direct path is untouched, which is the point of gating.

The same script verifies 9 geometry cases (cluster sizing, density-dependent
triggering, wrap-around across 0°, capping at 3) and runs a headless smoke test
of the Streamlit app.

---

## 10. Reproducing everything

### 10.1 Setup

```bash
# Python 3.10, PyTorch 2.11 (CUDA), SpeechBrain 1.1
conda activate ai_env

pip install torch torchaudio speechbrain pyroomacoustics librosa \
            streamlit==1.45.0 matplotlib numpy scipy

# SepFormer (2-spk, 3-spk) + ECAPA-TDNN -> ./pretrained/  (~330 MB, needs network)
python fetch_pretrained.py
```

Point `TARGET_AUDIO_DIR` / `NOISE_AUDIO_DIR` in `train.py` at a LibriSpeech
tree, and keep `SPEAKERS.TXT` (the official gender metadata) beside `app.py`.

### 10.2 Train

```bash
python train.py          # dual-GPU, ~45 min/epoch, checkpoints -> ./checkpoints_v5/
```

Resumes from `checkpoints_v5/last.pt`, otherwise warm-starts from
`INIT_FROM` (`checkpoints_v4/best.pt`), otherwise trains from scratch. A pid
lock prevents accidental double launches from colliding on the GPUs.

### 10.3 Analysis and benchmarks

| Command | Produces | Runtime |
|---|---|---|
| `python analyze_angular_safety.py` | `angular_safety/{report.txt,results.json,curve.png}` — Section 5 | ~7 min |
| `python analyze_azimuth_dependence.py` | `angular_safety/azimuth_dependence.{txt,png}` | ~5 min |
| `python benchmark_fallback.py` | `benchmark_fallback/{report.txt,results.json,curve.png}` — Section 7.1–7.4 | ~3 min |
| `python benchmark_density.py` | `benchmark_density/*` **and `fallback_thresholds.json`** — Section 8 | ~10 min |
| `python benchmark_arbitration.py` | `benchmark_arbitration/*` — Section 7.5 | ~4 min |

Times are for 2 × RTX 5060 Ti; the benchmarks use one GPU and fall back to CPU
automatically.

> `benchmark_density.py` **owns** `fallback_thresholds.json`. Rerun it after
> further training — as the beamformer improves, the triggers should shrink, and
> the app picks the new table up with no code change.

### 10.4 Tests and listening

```bash
python test_close_speakers.py   # geometry + end-to-end audio + app smoke test
python test_directions.py       # 24-azimuth sweep per example -> ./test_outputs/
python listen_examples.py       # target / mixture / extracted wavs -> ./listen_examples/
```

`test_close_speakers.py` exits non-zero if any geometry case is wrong or if the
fallback regresses where it fires.

### 10.5 Interactive demo

```bash
streamlit run app.py
```

Choose 2–11 speakers, name them, pick male/female voices (drawn from
LibriSpeech by gender) or record your own 5 s clip in the browser, place them by
coordinates on a live room map, select the target, and run. The sidebar reports
whether the geometry is safe — naming the nearest neighbour and the trigger for
that room size — and when the fallback runs you get the beamformer output and
the final output side by side, plus every separated stream with its similarity
score and a mark on the one reported as the target.

---

## 11. Repository map

| File | Role |
|---|---|
| `train.py` | model, dataset wrapper, training loop, multi-GPU, loss, checkpointing |
| `dataset.py` | room simulation, array geometry, steering vectors, scene constraints |
| `fallback_separation.py` | close-speaker pipeline + density-aware geometry gate |
| `app.py` | Streamlit demo |
| `fetch_pretrained.py` | downloads SepFormer 2/3-spk and ECAPA-TDNN |
| `analyze_angular_safety.py` | safe-angle sweep (Section 5) |
| `analyze_azimuth_dependence.py` | array-orientation check (Section 5) |
| `benchmark_fallback.py` | separator / input / reconstruction comparison (Section 7) |
| `benchmark_density.py` | density-aware triggers → `fallback_thresholds.json` (Section 8) |
| `benchmark_arbitration.py` | angular gate vs. per-scene arbitration (Section 7.5) |
| `test_close_speakers.py` | geometry, end-to-end audio, app smoke test (Section 9) |
| `test_directions.py` | direction-selectivity sweep |
| `listen_examples.py` | deterministic listening examples |
| `fallback_thresholds.json` | measured trigger table (generated) |
| `SPEAKERS.TXT` | LibriSpeech gender metadata for the demo's voice picker |

---

## 12. Limitations

**Anechoic training.** `max_order=0` means no reflections at all. Real rooms add
reverberation, which smears the direction cue; expect degradation on real
recordings, and retrain with `max_order > 0` for a reverberant deployment.

**Array spacing is baked in.** The model learns spatial filters for a specific
geometry. A model trained at 0.10 m spacing will **not** work on a 0.05 m array.
Match `SPACING` in `dataset.py` to your hardware before training.

**Simulated evaluation only.** Every number here comes from `pyroomacoustics`
simulations with LibriSpeech sources. No real-array recordings have been tested.

**Enrolment assumed for identification.** The fallback is most reliable when a
clip of the target speaker is available. Without one it degrades gracefully
above 6° but is close to a coin flip at 2–4°.

**Undertrained checkpoint.** These results are from epoch 6 of an 80-epoch
schedule. All fallback thresholds are relative to this checkpoint's quality and
should be regenerated as training continues.

**Separator ceiling.** The fallback plateaus at 8–14 dB regardless of how easy
the scene is, because SepFormer's output quality bounds it. At wide separations
the beamformer alone comfortably beats that — hence the gate.
