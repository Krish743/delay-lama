
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

INPUT = "../data/lamptrim.wav"
OUTPUT = "../outputs/lampmel.wav"

Path("outputs").mkdir(exist_ok=True)

# Load mono audio at a consistent sample rate
y, sr = librosa.load(INPUT, sr=22050, mono=True)

# Estimate fundamental frequency
f0, voiced_flag, voiced_prob = librosa.pyin(
    y,
    fmin=librosa.note_to_hz("C2"),
    fmax=librosa.note_to_hz("C7"),
    sr=sr,
    frame_length=2048,
    hop_length=256,
)

# Convert detected pitches to MIDI note numbers
midi = np.full_like(f0, np.nan)
valid = np.isfinite(f0) & voiced_flag
midi[valid] = librosa.hz_to_midi(f0[valid])

# Smooth pitch in MIDI space to reduce jitter
# Only smooth within contiguous voiced regions.
from scipy.ndimage import median_filter

clean_midi = midi.copy()
indices = np.flatnonzero(valid)

if len(indices):
    groups = np.split(
        indices, np.where(np.diff(indices) > 1)[0] + 1
    )
    for group in groups:
        if len(group) >= 3:
            clean_midi[group] = median_filter(
                midi[group], size=5, mode="nearest"
            )

# Convert back to frequency
clean_f0 = np.zeros_like(f0)
valid_clean = np.isfinite(clean_midi)
clean_f0[valid_clean] = librosa.midi_to_hz(
    clean_midi[valid_clean]
)

# Save a diagnostic pitch contour
times = librosa.times_like(
    f0, sr=sr, hop_length=256
)

np.savez(
    "../outputs/lampmel.npz",
    times=times,
    f0=clean_f0,
    voiced=valid_clean,
    sr=sr,
    hop_length=256,
)

# Save the original audio for reference
sf.write(OUTPUT, y, sr)

print(f"Audio duration: {len(y) / sr:.2f} seconds")
print(f"Detected voiced frames: {valid_clean.mean():.1%}")
print("Saved pitch data to outputs/melody.npz")
