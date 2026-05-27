# Loader for the 768-dim / 20-layer nenad1002/lite-conformer-de-200m variant.
# Mirrors nemotron_encoder_load.py but with the right architecture constants.

import sys
from pathlib import Path

# Apply LiteConformer patches BEFORE NeMo loads any model.
# Scoped to this loader only — no other export pipeline is affected.
sys.path.insert(0, "/home/nebanfic")
from lite_conformer import apply_lite_conformer_patches
apply_lite_conformer_patches()

import torch

CHUNK_SIZE = 0.56
LEFT_CHUNKS = 10
MEL_FEATURES = 128
SUBSAMPLING_FACTOR = 8

N_LAYERS = 20
D_MODEL = 768
CONV_CONTEXT = 14  # conv_kernel_size(15) - 1

_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")


def _get_streaming_shapes():
    chunk_encoded_frames = int(CHUNK_SIZE * 100) // SUBSAMPLING_FACTOR
    left_context = LEFT_CHUNKS * chunk_encoded_frames
    pre_encode_cache = 9
    chunk_mel_frames = int(CHUNK_SIZE * 100)
    static_mel_frames = chunk_mel_frames + pre_encode_cache
    return {
        "last_channel_cache_size": left_context,
        "static_mel_frames": static_mel_frames,
    }


def model_loader(model_name):
    sys.path.insert(0, _SCRIPTS_DIR)
    try:
        from export_nemotron_to_onnx_static_shape import (
            _make_streaming_encoder_wrapper,
            get_att_context_size,
        )
    finally:
        sys.path.pop(0)

    import nemo.collections.asr as nemo_asr

    if model_name.endswith(".nemo"):
        asr_model = nemo_asr.models.ASRModel.restore_from(model_name)
    else:
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    asr_model = asr_model.cpu()
    asr_model.eval()

    encoder = asr_model.encoder
    encoder.eval()
    att_context_size = get_att_context_size(CHUNK_SIZE, LEFT_CHUNKS)
    if hasattr(encoder, "set_default_att_context_size"):
        encoder.set_default_att_context_size(att_context_size)

    wrapper = _make_streaming_encoder_wrapper(encoder)
    wrapper.eval()
    return wrapper


def generate_dummy_inputs(model):
    shapes = _get_streaming_shapes()
    static_mel_frames = shapes["static_mel_frames"]
    last_channel_cache_size = shapes["last_channel_cache_size"]

    batch = 1
    dummy_audio = torch.randn(batch, static_mel_frames, MEL_FEATURES)
    dummy_length = torch.tensor([static_mel_frames], dtype=torch.int64)
    dummy_cache_ch = torch.zeros(batch, N_LAYERS, last_channel_cache_size, D_MODEL)
    dummy_cache_tm = torch.zeros(batch, N_LAYERS, D_MODEL, CONV_CONTEXT)
    dummy_cache_len = torch.zeros(batch, dtype=torch.int64)

    return (dummy_audio, dummy_length, dummy_cache_ch, dummy_cache_tm, dummy_cache_len)
