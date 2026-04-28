# Audio-Reactive Engine

Native audio-tensor processing inside ComfyUI. Load WAV/MP3, analyze
into onsets/BPM/centroid/RMS, derive per-frame float curves, and use
them to drive masks, sampler CFG, or any other parameter.

## Nodes

- **Audio Load & Analyze** — `path` -> `AUDIO_FEATURES` + BPM. Uses
  soundfile/librosa when available, falls back to stdlib `wave`.
- **Audio → Float Curve** — extract per-frame float list at given fps,
  selectable band (bass/mid/treble/onsets/centroid).
- **Audio Drive Mask** — modulate intensity / dilation / feather of a
  per-frame mask from a curve.
- **Audio Drive Schedule** — emit a JSON schedule for sampler CFG /
  denoise hooks.
- **Audio Spectrogram** — debug visualization.
