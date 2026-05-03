# NOTICE — ComfyUI-NukeMaxNodes

Copyright (c) 2025-2026 Code2Collapse  
License: Apache License 2.0 (see LICENSE)

---

## Trademark Disclaimer

**"Nuke"** is a registered trademark of **The Foundry Visionmongers Ltd**.
This project (ComfyUI-NukeMaxNodes) is an independent open-source ComfyUI
node pack. It is **not affiliated with, endorsed by, or associated with
The Foundry Visionmongers Ltd** in any way.

The name reflects conceptual inspiration from professional VFX compositing
workflows; no code, algorithms, or proprietary assets from The Foundry's
Nuke software are included or reproduced in this project.

---

## Third-Party Libraries

The following open-source libraries are used by this project. Their
copyrights and licenses are reproduced for compliance.

### 1. PyTorch

**Repository**: <https://github.com/pytorch/pytorch>  
**Copyright**: Copyright (c) 2016-2024 Facebook, Inc. and its affiliates  
**License**: BSD-style license  
<https://github.com/pytorch/pytorch/blob/main/LICENSE>

### 2. NumPy

**Repository**: <https://github.com/numpy/numpy>  
**Copyright**: Copyright (c) 2005-2024 NumPy Developers  
**License**: BSD 3-Clause  
<https://github.com/numpy/numpy/blob/main/LICENSE.txt>

### 3. librosa

**Repository**: <https://github.com/librosa/librosa>  
**Copyright**: Copyright (c) 2013-2024 librosa development team  
**License**: ISC License  
<https://github.com/librosa/librosa/blob/main/LICENSE.md>

### 4. SoundFile (PySoundFile)

**Repository**: <https://github.com/bastibe/python-soundfile>  
**Copyright**: Copyright (c) 2013 Bastian Bechtold  
**License**: BSD 3-Clause  
<https://github.com/bastibe/python-soundfile/blob/master/LICENSE>

### 5. imageio

**Repository**: <https://github.com/imageio/imageio>  
**Copyright**: Copyright (c) 2014-2024 imageio contributors  
**License**: BSD 2-Clause  
<https://github.com/imageio/imageio/blob/master/LICENSE>

### 6. OpenCV

**Repository**: <https://github.com/opencv/opencv>  
**License**: Apache License 2.0 (since OpenCV 4.5)  
<https://github.com/opencv/opencv/blob/master/LICENSE>

### 7. SciPy

**Repository**: <https://github.com/scipy/scipy>  
**Copyright**: Copyright (c) 2001-2024 SciPy Developers  
**License**: BSD 3-Clause  
<https://github.com/scipy/scipy/blob/main/LICENSE.txt>

### 8. OpenEXR / Imath (optional dependency)

**Repository**: <https://github.com/AcademySoftwareFoundation/openexr>  
**Copyright**: Copyright (c) Contributors to the OpenEXR Project  
**License**: BSD 3-Clause  
<https://github.com/AcademySoftwareFoundation/openexr/blob/main/LICENSE.md>

---

## FFT Algorithms

The FFT routines in `nukemax/nodes/fft/fft_tensor.py` use PyTorch's built-in
`torch.fft` module which is based on the FFTPACK / cuFFT libraries included
in PyTorch under its own BSD license.

---

## Optical Flow

Optical-flow estimation in `nukemax/nodes/flow/flow_field.py` uses standard
PyTorch and OpenCV algorithms. No code is copied from proprietary sources.
