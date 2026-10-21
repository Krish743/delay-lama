
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

data = np.load("../outputs/lampmel.npz")

times = data["times"]
f0 = data["f0"].astype(float)
voiced = data["voiced"].astype(bool)
sr = int(data["sr"])
hop = int(data["hop_length"])

# Recover the audio-length estimate from the frame timeline.
duration = times[-1] + hop / sr
n_samples = int(round(duration * sr))
audio_times = np.arange(n_samples) / sr

# Interpolate pitch and voicing onto individual audio samples.
pitch = np.interp(audio_times, times, f0)
gate = np.interp(audio_times, times, voiced.astype(float))

# Smooth the gate to avoid abrupt clicks.
gate = np.clip(gate, 0, 1)
gate[gate < 0.15] = 0

# Harmonic oscillator with a simple vowel-like spectral envelope.
phase = np.cumsum(2 * np.pi * np.maximum(pitch, 0) / sr)
audio = np.zeros(n_samples)

# Relative strengths of harmonics; these give a buzzy timbre.
harmonics = np.arange(1, 25)
formants = [700, 1200, 2600]  # Approximate vowel resonances
formant_weights = (
    np.exp(-0.5 * ((harmonics[:, None] * 150 - np.array(formants)[None, :])
                    / 250) ** 2)
).sum(axis=1)

for h, weight in zip(harmonics, formant_weights):
    audio += (weight / h) * np.sin(h * phase)

# Normalize and apply the voicing gate.
peak = np.max(np.abs(audio))
if peak > 0:
    audio /= peak

audio *= gate

# Gentle low-pass filtering to tame harsh upper harmonics.
sos = butter(4, 5000, fs=sr, output="sos")
audio = sosfilt(sos, audio)

# Prevent clipping and export.
peak = np.max(np.abs(audio))
if peak > 0:
    audio = 0.9 * audio / peak

sf.write("../outputs/lampmonk.wav", audio, sr)
print("Saved outputs/lampmonk.wav")
