#!/usr/bin/env python3
"""
Compare three tests to isolate cache-dit's impact:
- Test A: No cache-dit baseline (10 steps)
- Test B: Cache-dit warmup only (10 warmup, 10 total)
- Test C: Cache-dit with caching (10 warmup, 20 total)

This reveals:
1. Does cache-dit wrapper affect warmup? (Compare A vs B)
2. Do cached steps pollute? (Compare B vs C)
"""

import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("Installing scikit-image...")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity as ssim


def load_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compute_metrics(frame1, frame2):
    mse = np.mean((frame1.astype(float) - frame2.astype(float)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf")
    ssim_val = ssim(frame1, frame2, channel_axis=2, data_range=255)
    return mse, psnr, ssim_val


def extract_mouth_region(frame):
    h, w = frame.shape[:2]
    y_start, y_end = int(h * 0.6), h
    x_start, x_end = int(w * 0.3), int(w * 0.7)
    return frame[y_start:y_end, x_start:x_end]


def compare_videos(frames1, frames2, label1, label2):
    print(f"\nComparing: {label1} vs {label2}")
    print("=" * 60)

    # Full frame
    psnr_full = [compute_metrics(f1, f2)[1] for f1, f2 in zip(frames1, frames2)]
    ssim_full = [compute_metrics(f1, f2)[2] for f1, f2 in zip(frames1, frames2)]

    # Mouth region
    psnr_mouth = []
    ssim_mouth = []
    for f1, f2 in zip(frames1, frames2):
        _, p, s = compute_metrics(extract_mouth_region(f1), extract_mouth_region(f2))
        psnr_mouth.append(p)
        ssim_mouth.append(s)

    print(f"Full Frame - PSNR: {np.mean(psnr_full):.2f} dB, SSIM: {np.mean(ssim_full):.4f}")
    print(f"Mouth Region - PSNR: {np.mean(psnr_mouth):.2f} dB, SSIM: {np.mean(ssim_mouth):.4f}")

    return {
        "psnr_full": np.mean(psnr_full),
        "ssim_full": np.mean(ssim_full),
        "psnr_mouth": np.mean(psnr_mouth),
        "ssim_mouth": np.mean(ssim_mouth),
    }


if __name__ == "__main__":
    base_path = Path("examples/offline_inference/speech_to_video")

    video_no_cache = base_path / "s2v_480p_tp4_no_cache_40steps.mp4"
    video_warmup_only = base_path / "s2v_480p_tp4_cache_w10_s10.mp4"
    video_with_cache = base_path / "s2v_480p_tp4_cache_w10_s20.mp4"

    print("\n" + "=" * 80)
    print("CACHE-DIT WARMUP IMPACT ANALYSIS")
    print("=" * 80)
    print("Test A: No cache-dit (10 steps) - BASELINE")
    print("Test B: Cache-dit warmup only (10 warmup, 10 total) - NO CACHING")
    print("Test C: Cache-dit with caching (10 warmup, 20 total) - WITH CACHING")
    print("=" * 80)

    # Check if all videos exist
    if not video_no_cache.exists():
        print(f"\n❌ Baseline not ready: {video_no_cache}")
        sys.exit(1)
    if not video_warmup_only.exists():
        print(f"\n❌ Warmup test not found: {video_warmup_only}")
        sys.exit(1)
    if not video_with_cache.exists():
        print(f"\n❌ Cache test not found: {video_with_cache}")
        sys.exit(1)

    # Load all frames
    print("\nLoading videos...")
    frames_no_cache = load_video_frames(video_no_cache)
    frames_warmup = load_video_frames(video_warmup_only)
    frames_cached = load_video_frames(video_with_cache)

    print(f"Loaded: {len(frames_no_cache)} frames each")

    # Compare A vs B (cache-dit wrapper impact)
    metrics_ab = compare_videos(frames_no_cache, frames_warmup, "Test A (no cache-dit)", "Test B (warmup only)")

    # Compare B vs C (cached steps impact)
    metrics_bc = compare_videos(frames_warmup, frames_cached, "Test B (warmup only)", "Test C (with caching)")

    # Compare A vs C (overall cache-dit impact)
    metrics_ac = compare_videos(frames_no_cache, frames_cached, "Test A (no cache-dit)", "Test C (with caching)")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\n1. Does cache-dit wrapper affect warmup? (A vs B)")
    if metrics_ab["psnr_mouth"] < 35:
        print("   ❌ YES - Warmup differs from baseline")
        print(f"      Mouth PSNR: {metrics_ab['psnr_mouth']:.2f} dB (should be >35 dB for 'same')")
        print("      → cache_dit.enable_cache() changes computation even during warmup!")
    else:
        print("   ✅ NO - Warmup matches baseline")
        print(f"      Mouth PSNR: {metrics_ab['psnr_mouth']:.2f} dB")

    print("\n2. Do cached steps pollute lip sync? (B vs C)")
    if metrics_bc["psnr_mouth"] < 30:
        print("   ❌ YES - Cached steps degrade lip sync")
        print(f"      Mouth PSNR: {metrics_bc['psnr_mouth']:.2f} dB")
        print("      → Cached steps (11-20) pollute generation")
    else:
        print("   ✅ NO - Cached steps maintain quality")
        print(f"      Mouth PSNR: {metrics_bc['psnr_mouth']:.2f} dB")

    print("\n3. Overall cache-dit quality impact (A vs C)")
    print(f"   Mouth PSNR: {metrics_ac['psnr_mouth']:.2f} dB")
    print(f"   Mouth SSIM: {metrics_ac['ssim_mouth']:.4f}")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if metrics_ab["psnr_mouth"] < 35:
        print("❌ cache-dit wrapper is problematic - changes computation during warmup")
        print("   Action: Investigate cache_dit.enable_cache() wrapper behavior")

    if metrics_bc["psnr_mouth"] < 30:
        print("❌ Cached steps pollute lip sync quality")
        print("   Action: Implement audio-aware adaptive caching")

    if metrics_ac["psnr_mouth"] < 30:
        print("\n⚠️  Overall: cache-dit is NOT suitable for S2V lip sync")
        print("   Consider alternative acceleration strategies")

    print("=" * 80 + "\n")
