"""Custom data types for the NukeMax pack.

Each type is a frozen dataclass holding `torch.Tensor` fields. Types are
registered as ComfyUI socket type strings (e.g. `("ROTO_SHAPE",)`) so
they get colored sockets and type-checked wires. Live wire payloads are
the dataclass instances themselves; ComfyUI does not enforce
socket/payload identity, so this works.

For workflow-JSON portability there is an optional safetensors+base64
serializer in `nukemax.types.serialize`.
"""
from .roto_shape import RotoShape
from .tracking_data import TrackingData
from .fft_tensor import FFTTensor
from .material_set import MaterialSet
from .light_probe import LightProbe
from .light_rig import Light, LightRig
from .audio_features import AudioFeatures
from .flow_field import FlowField
from .deep_image import DeepImage

# Registered ComfyUI socket type names (uppercase string conventions).
TYPE_NAMES = (
    "ROTO_SHAPE",
    "TRACKING_DATA",
    "FFT_TENSOR",
    "MATERIAL_SET",
    "LIGHT_PROBE",
    "LIGHT_RIG",
    "AUDIO_FEATURES",
    "FLOW_FIELD",
    "DEEP_IMAGE",
)

__all__ = [
    "RotoShape",
    "TrackingData",
    "FFTTensor",
    "MaterialSet",
    "LightProbe",
    "Light",
    "LightRig",
    "AudioFeatures",
    "FlowField",
    "DeepImage",
    "TYPE_NAMES",
]
