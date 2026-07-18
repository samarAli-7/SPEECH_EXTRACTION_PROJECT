"""
End-to-end check of the close-speaker fallback, with audio you can listen to.

Part 1 -- geometry gating (no models): verifies analyze_geometry triggers only
         when it should, and picks the 2- vs 3-speaker pipeline correctly.
Part 2 -- audio: for a set of separations (very close -> comfortably apart),
         renders one scene per separation through app.run_extraction, both with
         and without the fallback, and writes to ./close_speaker_tests/:
             sep_XX_mixture.wav        what the array hears
             sep_XX_beamformer.wav     direction-conditioned model alone
             sep_XX_final.wav          after separation + ECAPA identification
             sep_XX_target.wav         ground truth
             report.txt                SI-SDR for every case
Part 3 -- app smoke test: drives app.py headlessly (streamlit AppTest) to make
         sure the UI path itself runs without exceptions.

Run:  python test_close_speakers.py
"""

import os
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio

import app
import fallback_separation as fbs
import train as tr
from analyze_angular_safety import load_clip, speaker_id
from dataset import FS

OUTPUT_DIR = "./close_speaker_tests"
SEPARATIONS = [3.0, 6.0, 10.0, 15.0, 30.0]
DISTANCE_M = 8.0
N_FAR = 3                      # extra far-away speakers in the room
SEED = 4242
# The angular gate is a population-level rule (benchmark_fallback.py), so a
# single scene can go either way; average a few voice draws per separation
# before judging it.
N_SCENES = 5


# ---------------------------------------------------------------------------
def test_geometry():
    print("=== Part 1: geometry gating ===")
    print(f"trigger table: {fbs.trigger_table()} (cluster cone {fbs.CLUSTER_CONE_DEG}°)")
    far = [90.0, 140.0, 190.0, 240.0, 290.0]
    checks = [
        # (azimuths, target, expect_fallback, expect_n_cluster, description)
        ([0.0, 90.0, 180.0], 0, False, 1, "well separated -> direct path"),
        ([0.0, 6.0, 180.0], 0, True, 2, "one neighbour at 6° -> 2-spk pipeline"),
        ([0.0, 8.0, 352.0], 0, True, 3, "two neighbours (8°) -> 3-spk pipeline"),
        ([0.0, 5.0, 10.0, 15.0, 200.0], 0, True, 3, "crowded cluster -> capped at 3"),
        ([0.0, 19.0], 0, False, 2, "19° in a 2-speaker room -> beamformer handles it"),
        # the same separation decides differently depending on room density
        ([0.0, 12.0, 180.0], 0, False, 2, "12°, sparse room (3 spk) -> no fallback"),
        ([0.0, 12.0] + far, 0, True, 2, "12°, crowded room (7 spk) -> fallback"),
        ([5.0, 358.0], 0, True, 2, "wrap-around across 0° is handled (7° apart)"),
        ([10.0, 355.0], 0, False, 2, "wrap-around, 15° apart in a 2-speaker room"),
    ]
    failures = 0
    for azimuths, tgt, want_fb, want_n, desc in checks:
        g = fbs.analyze_geometry(azimuths, tgt)
        ok = (g["needs_fallback"] == want_fb) and (g["n_cluster"] == want_n)
        failures += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc}: nearest "
              f"{g['nearest_sep_deg']:.1f}°, {g['n_speakers']} spk, trigger "
              f"{g['trigger_deg']:.1f}°, n_cluster {g['n_cluster']}, "
              f"fallback {g['needs_fallback']}")
    print(f"  {len(checks) - failures}/{len(checks)} geometry checks passed\n")
    return failures


# ---------------------------------------------------------------------------
def save_wav(path, wav):
    peak = float(np.abs(wav).max()) or 1e-8
    data = torch.tensor((wav * (0.9 / peak)).astype(np.float32)).unsqueeze(0)
    torchaudio.save(path, data, FS, encoding="PCM_S", bits_per_sample=16)


def test_audio():
    print("=== Part 2: end-to-end audio ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt = app.find_checkpoint()
    model, epoch = app.load_model(ckpt)
    mic_positions = app.build_mic_positions(app.mic_spacing_for(ckpt))
    fb = fbs.SeparationFallback(device="cpu")
    print(f"checkpoint: {ckpt} (epoch {epoch})")

    # distinct validation speakers, one clip each
    files = tr.collect_audio_files(tr.TARGET_AUDIO_DIR)
    shuffled = files.copy()
    random.Random(tr.SPLIT_SEED).shuffle(shuffled)
    val = shuffled[:max(1, int(len(shuffled) * tr.VAL_FRACTION))]
    by_spk = defaultdict(list)
    for f in val:
        by_spk[speaker_id(f)].append(f)
    lines = [f"Close-speaker fallback -- end-to-end audio test",
             f"checkpoint: {ckpt} (epoch {epoch})",
             f"{2 + N_FAR} speakers: one neighbour at the listed separation, "
             f"{N_FAR} others at least 45° away.",
             f"Mean over {N_SCENES} random scenes (voices, orientation, and far "
             f"speakers redrawn); the saved wavs are draw 0.",
             "",
             f"{'sep':>6} | {'mixture':>9} | {'beamformer':>11} | {'final':>9} | "
             f"{'gain':>7} | path",
             "-" * 68]
    failures = 0

    for sep in SEPARATIONS:
        mix_db, bf_db, fin_db = [], [], []
        for scene in range(N_SCENES):
            rng = random.Random(SEED + 1000 * scene + int(sep))
            spks = rng.sample(sorted(by_spk), k=2 + N_FAR)
            signals = [load_clip(rng.choice(by_spk[s]), rng) for s in spks]

            # Randomize the whole scene's orientation and the far speakers'
            # placement per draw, so this measures the same population the
            # angular gate was fitted on rather than one fixed layout.
            base = rng.uniform(0.0, 360.0)
            azimuths = [base, (base + sep) % 360.0]
            for _ in range(N_FAR):
                a = rng.uniform(0.0, 360.0)
                for _try in range(200):
                    if all(fbs.angular_diff_deg(a, b) >= 45.0 for b in azimuths):
                        break
                    a = rng.uniform(0.0, 360.0)
                azimuths.append(a)
            rel = [(DISTANCE_M * np.cos(np.radians(a)),
                    DISTANCE_M * np.sin(np.radians(a))) for a in azimuths]
            geo = fbs.analyze_geometry(azimuths, 0)

            res = app.run_extraction(model, mic_positions, signals, rel, 0,
                                     fallback=fb if geo["needs_fallback"] else None,
                                     use_enrolment=True)
            mix_db.append(res["sisdr_mix"])
            bf_db.append(res["sisdr_ext"])
            fin_db.append(res["sisdr_final"])

            if scene == 0:
                tag = f"sep_{int(sep):02d}"
                save_wav(os.path.join(OUTPUT_DIR, f"{tag}_mixture.wav"), res["mixture_ref"])
                save_wav(os.path.join(OUTPUT_DIR, f"{tag}_beamformer.wav"), res["extracted"])
                save_wav(os.path.join(OUTPUT_DIR, f"{tag}_target.wav"), res["target_ref"])
                if res["path"] == "fallback":
                    save_wav(os.path.join(OUTPUT_DIR, f"{tag}_final.wav"), res["final"])
                    for k, s in enumerate(res["fallback"]["streams"]):
                        save_wav(os.path.join(OUTPUT_DIR, f"{tag}_stream{k + 1}.wav"), s)
                path = res["path"]

        mix_m, bf_m, fin_m = np.mean(mix_db), np.mean(bf_db), np.mean(fin_db)
        gain = fin_m - bf_m
        lines.append(f"{sep:6.1f} | {mix_m:9.2f} | {bf_m:11.2f} | {fin_m:9.2f} | "
                     f"{gain:+7.2f} | {path}")
        note = ""
        if path == "fallback" and gain < 0.0:
            failures += 1
            note = "  <-- REGRESSION (fallback should help where it fires)"
        print(f"  sep {sep:5.1f}° | mixture {mix_m:6.2f} | beamformer {bf_m:6.2f} | "
              f"final {fin_m:6.2f} dB (mean of {N_SCENES}, {path}){note}")

    with open(os.path.join(OUTPUT_DIR, "report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  audio written to {OUTPUT_DIR}/\n")
    return failures


# ---------------------------------------------------------------------------
def test_app_smoke():
    print("=== Part 3: app smoke test ===")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=600)
    at.run()
    if at.exception:
        print(f"  [FAIL] app raised: {at.exception}")
        return 1

    # place two speakers 6° apart so the fallback path is exercised in the UI
    at.session_state["x_0"], at.session_state["y_0"] = 8.0, 0.0
    at.session_state["x_1"], at.session_state["y_1"] = 7.956, 0.836   # ~6°
    at.run()
    if at.exception:
        print(f"  [FAIL] app raised after positioning: {at.exception}")
        return 1

    infos = [i.value for i in at.info]
    close_msg = any("Close-speaker geometry" in str(v) for v in infos)
    print(f"  [{'PASS' if close_msg else 'FAIL'}] close-speaker notice shown "
          f"for a 6° pair")

    at.button[0].click().run()
    if at.exception:
        print(f"  [FAIL] extraction raised: {at.exception}")
        return 1
    ran = "results" in at.session_state
    used = at.session_state["results"]["path"] if ran else None
    print(f"  [{'PASS' if ran else 'FAIL'}] extraction completed (path: {used})")
    return 0 if (close_msg and ran) else 1


if __name__ == "__main__":
    fails = test_geometry() + test_audio() + test_app_smoke()
    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} CHECK(S) FAILED'}")
    raise SystemExit(1 if fails else 0)
