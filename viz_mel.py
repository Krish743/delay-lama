
import numpy as np
import matplotlib.pyplot as plt
import librosa

data = np.load("../outputs/lampmel.npz")

for key in data.files:
    arr = data[key]
    print(f"{key}: shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim == 0:
        print(f"    value={arr.item()}")

times = data["times"]
f0 = data["f0"]
voiced = data["voiced"]

# Plot detected pitches
plt.figure(figsize=(14, 5))
plt.scatter(
    times[voiced],
    librosa.hz_to_midi(f0[voiced]),
    s=5,
)
plt.xlabel("Time (seconds)")
plt.ylabel("MIDI note number")
plt.title("Extracted melody")
plt.grid(True)
plt.tight_layout()
plt.show()
