"""Log-mel spectrogram feature extraction for cough recordings.

All 200 source clips are already mono/44.1kHz/1.0s (verified during Task 5
data audit), but load_audio still resamples/pads/trims defensively so the
pipeline doesn't silently break if a differently-recorded clip is added
later.
"""

import librosa
import numpy as np

SAMPLE_RATE = 44100
DURATION_SEC = 1.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256


def load_audio(filepath) -> np.ndarray:
    audio, _ = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    target_len = int(SAMPLE_RATE * DURATION_SEC)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        audio = audio[:target_len]
    return audio


def extract_log_mel(filepath) -> np.ndarray:
    """Returns a (N_MELS, T) log-mel spectrogram, standardized to zero
    mean / unit variance per-clip."""
    audio = load_audio(filepath)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)
