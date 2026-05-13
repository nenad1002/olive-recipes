# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Model loaders and dummy input generators for Cohere Transcribe components.

Used by Olive's ``OnnxConversion`` pass via the ``model_script`` /
``model_loader`` mechanism. Each component (encoder, decoder) has its own
loader and dummy inputs function, referenced from separate Olive JSON configs.

Mirrors the wrappers used in ``scripts/export_cohere_to_onnx.py`` (kept for
reference) so that the ONNX I/O layout matches what onnxruntime-genai's
``cohere_transcribe`` model type expects.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared constants (architecture of cohere-transcribe-03-2026)
# ---------------------------------------------------------------------------
MODEL_NAME = "CohereLabs/cohere-transcribe-03-2026"

NUM_DECODER_LAYERS = 8
NUM_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = NUM_HEADS * HEAD_DIM  # 1024
NUM_MELS = 128
MAX_SEQ = 1024  # static self-attention KV cache length
VOCAB_SIZE = 16384


def _load_hf_model(model_name: str = MODEL_NAME):
    """Load the HuggingFace ``CohereAsrForConditionalGeneration`` model in fp32."""
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except TypeError:
        # Older custom model classes don't accept ``torch_dtype``.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            config=cfg,
            trust_remote_code=True,
        ).to(torch.float32)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class EncoderWrapper(nn.Module):
    """``mel [B, 128, T]`` + ``mel_length [B]`` -> per-layer cross KV caches.

    Runs the conformer encoder, projects to the decoder hidden size and then
    materializes per-layer cross-attention K/V using each decoder layer's
    ``second_sub_layer`` (cross-attn) projections.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.encoder_decoder_proj  # may be None if dims match
        decoder_layers = model.transf_decoder._decoder.layers
        self.cross_key_projs = nn.ModuleList(
            [layer.second_sub_layer.key_net for layer in decoder_layers]
        )
        self.cross_value_projs = nn.ModuleList(
            [layer.second_sub_layer.value_net for layer in decoder_layers]
        )
        any_cross = decoder_layers[0].second_sub_layer
        self.num_heads = any_cross.num_heads
        self.head_dim = any_cross.head_dim
        self.num_layers = len(decoder_layers)

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor):
        enc_out, _ = self.encoder(input_features=mel, length=mel_length)
        if self.proj is not None:
            enc_out = self.proj(enc_out)
        b, t, _ = enc_out.shape
        keys, values = [], []
        for i in range(self.num_layers):
            k = (
                self.cross_key_projs[i](enc_out)
                .view(b, t, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.cross_value_projs[i](enc_out)
                .view(b, t, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            keys.append(k)
            values.append(v)
        return tuple(keys + values)


def encoder_model_loader(model_name: str):
    """Olive entry point for the encoder."""
    model = _load_hf_model(model_name)
    wrapper = EncoderWrapper(model).float().eval()
    return wrapper


def encoder_dummy_inputs(model: nn.Module):
    """Dummy inputs for ONNX export of the encoder.

    The mel time axis is dynamic at runtime; here we trace with a small
    representative length. Olive will combine these inputs with the
    ``dynamic_axes`` declared in the JSON config.
    """
    dummy_t = 300
    mel = torch.randn(1, NUM_MELS, dummy_t, dtype=torch.float32)
    mel_length = torch.tensor([dummy_t], dtype=torch.int64)
    return (mel, mel_length)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
class DecoderWrapper(nn.Module):
    """``input_ids`` + past KV caches -> logits + updated self KV caches.

    Implements the decoder forward in plain ops (no HF cache classes) so that
    ONNX tracing produces a graph with explicit past/present tensors that the
    onnxruntime-genai runtime can manage.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.embedding = model.transf_decoder._embedding
        self.layers = model.transf_decoder._decoder.layers
        self.final_ln = model.transf_decoder._decoder.final_layer_norm
        # Logits head: Linear(hidden, vocab) tied to token embedding weight.
        # Skip the log_softmax — genai expects raw logits.
        self.cls_head = model.log_softmax.mlp.layer0
        first_attn = self.layers[0].first_sub_layer
        self.num_heads = first_attn.num_heads
        self.head_dim = first_attn.head_dim
        self.hidden_size = first_attn.hidden_size
        self.num_layers = len(self.layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_sequence_length: torch.Tensor,
        *past_kv_args: torch.Tensor,
    ):
        nl = self.num_layers
        ns = nl * 2
        past_self = past_kv_args[:ns]
        past_cross = past_kv_args[ns:]

        b, sl = input_ids.shape
        # Cast to int64 so all subsequent index arithmetic / Slice ops use a
        # single int type (mixing int32 and int64 in onnx::Slice's Tind
        # produces "Type parameter (Tind) bound to different types" errors).
        pl = past_sequence_length.reshape(1)[0].to(torch.int64)
        positions = torch.arange(sl, device=input_ids.device) + pl
        hidden = self.embedding(input_ids, positions.unsqueeze(0).expand(b, -1))

        total_len = pl + sl
        # Static-shape causal mask shaped [sl, MAX_SEQ] then sliced to total_len.
        row_idx = torch.arange(sl, device=input_ids.device).unsqueeze(1) + pl
        col_idx = torch.arange(MAX_SEQ, device=input_ids.device).unsqueeze(0)
        causal = col_idx > row_idx  # mask future positions
        attn_mask = torch.where(
            causal[:, :total_len],
            torch.tensor(float("-inf"), device=input_ids.device, dtype=hidden.dtype),
            torch.tensor(0.0, device=input_ids.device, dtype=hidden.dtype),
        )

        scale = self.head_dim ** -0.5
        present_self = []
        for i, layer in enumerate(self.layers):
            pk, pv = past_self[i * 2], past_self[i * 2 + 1]
            xk, xv = past_cross[i * 2], past_cross[i * 2 + 1]

            # Self-attention.
            residual = hidden
            hidden = layer.layer_norm_1(hidden)
            sa = layer.first_sub_layer
            q = (
                sa.query_net(hidden)
                .view(b, sl, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            k = (
                sa.key_net(hidden)
                .view(b, sl, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                sa.value_net(hidden)
                .view(b, sl, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            new_k, new_v = pk.clone(), pv.clone()
            new_k[:, :, pl : pl + sl, :] = k
            new_v[:, :, pl : pl + sl, :] = v
            kf = new_k[:, :, :total_len, :]
            vf = new_v[:, :, :total_len, :]
            aw = torch.matmul(q, kf.transpose(-2, -1)) * scale
            aw = aw + attn_mask.unsqueeze(0).unsqueeze(0)
            ao = torch.matmul(torch.softmax(aw, dim=-1), vf)
            ao = ao.transpose(1, 2).contiguous().view(b, sl, self.hidden_size)
            hidden = residual + sa.out_projection(ao)

            # Cross-attention.
            residual = hidden
            hidden = layer.layer_norm_2(hidden)
            ca = layer.second_sub_layer
            qc = (
                ca.query_net(hidden)
                .view(b, sl, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )
            cw = torch.softmax(
                torch.matmul(qc, xk.transpose(-2, -1)) * scale, dim=-1
            )
            co = (
                torch.matmul(cw, xv)
                .transpose(1, 2)
                .contiguous()
                .view(b, sl, self.hidden_size)
            )
            hidden = residual + ca.out_projection(co)

            # Feed-forward.
            residual = hidden
            hidden = layer.layer_norm_3(hidden)
            hidden = residual + layer.third_sub_layer(hidden)

            present_self.extend([new_k, new_v])

        if self.final_ln is not None:
            hidden = self.final_ln(hidden)
        logits = self.cls_head(hidden)
        return (logits,) + tuple(present_self)


def decoder_model_loader(model_name: str):
    """Olive entry point for the decoder."""
    model = _load_hf_model(model_name)
    wrapper = DecoderWrapper(model).float().eval()
    return wrapper


def decoder_dummy_inputs(model: nn.Module):
    """Dummy inputs for ONNX export of the decoder."""
    nl = NUM_DECODER_LAYERS
    nh = NUM_HEADS
    hd = HEAD_DIM
    ms = MAX_SEQ

    input_ids = torch.tensor([[7]], dtype=torch.int32)
    past_seq_len = torch.tensor([0], dtype=torch.int32)
    past_self = [torch.zeros(1, nh, ms, hd, dtype=torch.float32) for _ in range(nl * 2)]
    past_cross = [torch.randn(1, nh, 100, hd, dtype=torch.float32) for _ in range(nl * 2)]
    return (input_ids, past_seq_len, *past_self, *past_cross)
