
from pathlib import Path

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parent.parent
DATA = np.load(ROOT / "outputs" / "somebodyfull.npz")

times = DATA["times"]
f0 = DATA["f0"]
voiced = DATA["voiced"].astype(bool)

# Convert pitch in Hz to MIDI note numbers.
valid = voiced & np.isfinite(f0) & (f0 > 0)

midi_pitch = np.full(len(f0), -1, dtype=int)
midi_pitch[valid] = np.rint(
    69 + 12 * np.log2(f0[valid] / 440.0)
).astype(int)

# Smooth out small pitch fluctuations by snapping to semitones.
# Group consecutive frames with the same MIDI pitch.
MIN_DURATION = 0.09  # seconds
GAP_TOLERANCE = 0.04 # seconds

notes = []
start = None
current_pitch = -1

for i, pitch in enumerate(midi_pitch):
    pitch = int(pitch)
    t = float(times[i])

    if pitch != current_pitch:
        if start is not None:
            end = t
            if current_pitch >= 0 and end - start >= MIN_DURATION:
                notes.append((current_pitch, start, end))

        start = t if pitch >= 0 else None
        current_pitch = pitch

# Close the last note.
if start is not None:
    end = float(times[-1]) + 0.02
    if current_pitch >= 0 and end - start >= MIN_DURATION:
        notes.append((current_pitch, start, end))

# Merge same-pitch notes separated by tiny gaps.
merged = []
for pitch, start, end in notes:
    if (
        merged
        and merged[-1][0] == pitch
        and start - merged[-1][2] <= GAP_TOLERANCE
    ):
        merged[-1] = (pitch, merged[-1][1], end)
    else:
        merged.append((pitch, start, end))

# Create MIDI.
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)

for pitch, start, end in merged:
    instrument.notes.append(
        pretty_midi.Note(
            velocity=100,
            pitch=int(np.clip(pitch, 0, 127)),
            start=max(0, start),
            end=max(start + 0.05, end),
        )
    )

midi.instruments.append(instrument)

output = ROOT / "outputs" / "somebodyfull.mid"
midi.write(str(output))

print(f"Saved {output}")
print(f"Notes generated: {len(instrument.notes)}")
