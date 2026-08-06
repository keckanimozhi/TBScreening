"""Verify the exact manual algorithm (reflect-pad -> frame -> Hann window
-> radix-2 FFT -> power spectrum -> mel filterbank matmul -> power_to_db
-> standardize) that will be ported line-for-line into Kotlin for
on-device cough feature extraction (Task 17), by comparing it against
librosa's own librosa.feature.melspectrogram + power_to_db on a real
cough clip.

There's no Android device/emulator available in this environment, so the
Kotlin port can't be tested by actually running it -- this script is the
substitute: verify the algorithm in Python first (fast iteration, exact
comparison against the ground-truth library), then port the verified
steps to Kotlin unchanged. If this script's max error is small, the
Kotlin port following the same steps should behave equivalently modulo
floating-point representation differences (both are IEEE754 doubles).

Also exports the mel filterbank matrix itself (librosa.filters.mel) as a
bundled Android asset -- reproducing librosa's exact Slaney-mel-scale
filter construction from scratch in Kotlin would be a second, unrelated
source of numerical risk, so the verified matrix is precomputed here and
just loaded as a lookup table on-device instead.

Usage:
    .venv/Scripts/python.exe -m src.export.verify_mel_algorithm
"""

import numpy as np
from src.common.config import PROJECT_ROOT
from src.cough.features import N_FFT, N_MELS, HOP_LENGTH, SAMPLE_RATE, extract_log_mel, load_audio
import librosa
import glob

ASSETS_DIR = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "assets"


def manual_fft(x: np.ndarray) -> np.ndarray:
    """Iterative radix-2 Cooley-Tukey FFT, real input, length must be a
    power of two (1024 here). Written to mirror what will be hand-coded
    in Kotlin (no numpy.fft) -- this is the exact algorithm being
    verified, not just a stand-in."""
    n = len(x)
    assert n & (n - 1) == 0, "length must be a power of two"

    # bit-reversal permutation
    real = x.astype(np.float64).copy()
    imag = np.zeros(n, dtype=np.float64)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]

    length = 2
    while length <= n:
        half = length // 2
        angle_step = -2.0 * np.pi / length
        for start in range(0, n, length):
            for k in range(half):
                angle = angle_step * k
                wr, wi = np.cos(angle), np.sin(angle)
                idx_even = start + k
                idx_odd = start + k + half
                er, ei = real[idx_even], imag[idx_even]
                tr = real[idx_odd] * wr - imag[idx_odd] * wi
                ti = real[idx_odd] * wi + imag[idx_odd] * wr
                real[idx_even] = er + tr
                imag[idx_even] = ei + ti
                real[idx_odd] = er - tr
                imag[idx_odd] = ei - ti
        length *= 2

    return real + 1j * imag


def periodic_hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def manual_log_mel(audio: np.ndarray, mel_basis: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    padded = np.pad(audio, pad, mode="constant")  # librosa >=0.10 default pad_mode for stft

    window = periodic_hann(N_FFT)
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH

    mel_spec = np.zeros((N_MELS, n_frames), dtype=np.float64)
    for t in range(n_frames):
        start = t * HOP_LENGTH
        frame = padded[start : start + N_FFT] * window
        spectrum = manual_fft(frame)[: N_FFT // 2 + 1]
        power = (spectrum.real ** 2 + spectrum.imag ** 2)
        mel_spec[:, t] = mel_basis @ power

    amin = 1e-10
    top_db = 80.0
    log_spec = 10.0 * np.log10(np.maximum(amin, mel_spec))
    log_spec -= 10.0 * np.log10(np.maximum(amin, mel_spec.max()))
    log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    standardized = (log_spec - log_spec.mean()) / (log_spec.std() + 1e-8)
    return standardized.astype(np.float32)


def main():
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)

    sample_files = glob.glob(str(PROJECT_ROOT / "dataset_200" / "cough" / "tb" / "*.wav"))[:5]
    sample_files += glob.glob(str(PROJECT_ROOT / "dataset_200" / "cough" / "notb" / "*.wav"))[:5]

    max_abs_diff = 0.0
    max_rel_files = []
    for f in sample_files:
        reference = extract_log_mel(f)
        audio = load_audio(f)
        manual = manual_log_mel(audio, mel_basis)
        diff = np.abs(reference - manual)
        d = diff.max()
        max_abs_diff = max(max_abs_diff, d)
        max_rel_files.append((f, d))

    print(f"Tested {len(sample_files)} files.")
    print(f"Max abs difference (standardized log-mel, manual vs librosa): {max_abs_diff:.6f}")
    for f, d in sorted(max_rel_files, key=lambda x: -x[1])[:3]:
        print(f"  {f}: max_diff={d:.6f}")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    mel_basis.astype(np.float32).tofile(ASSETS_DIR / "mel_filterbank_64x513.bin")
    print(f"Saved mel filterbank ({mel_basis.shape}, float32) to {ASSETS_DIR / 'mel_filterbank_64x513.bin'}")

    assert max_abs_diff < 0.05, "Manual algorithm diverges from librosa beyond acceptable tolerance"
    print("Manual algorithm verified: safe to port to Kotlin unchanged.")


if __name__ == "__main__":
    main()
