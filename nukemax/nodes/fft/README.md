# Frequency-Guided Generation

Operate in the FFT domain. Match noise spectra to image context for
seamless inpainting; band-filter; synthesize seamless tiles.

## Nodes

- **FFT Analyze / FFT Synthesize** — round-trip an `IMAGE` through an
  `FFT_TENSOR` (magnitude + phase, centered).
- **Frequency Mask** — radial band-pass with raised-cosine roll-off.
- **Latent Frequency Match** — warps noise spectrum to match context;
  inputs `LATENT` + `IMAGE`, outputs corrected `LATENT` for KSampler.
- **FFT Texture Synthesis** — seamless tile by re-randomizing phase.
