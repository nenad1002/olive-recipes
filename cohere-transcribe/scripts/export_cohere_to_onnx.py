#!/usr/bin/env python3
"""Export CohereLabs/cohere-transcribe-03-2026 to ONNX (encoder + decoder).

Produces a pair of ONNX models compatible with the ``cohere_transcribe`` model
type in onnxruntime-genai:

  encoder.onnx
      mel [B, 128, T_mel] + mel_length [B] -> cross-attention KV caches for
      every decoder layer (present_key_cross_i / present_value_cross_i).
  decoder.onnx
      input_ids + past_sequence_length + past self/cross KV caches
          -> logits + updated self-attention KV caches.

The official model is a HuggingFace ``CohereAsrForConditionalGeneration``:
  - ConformerEncoder (48 layers, d_model=1280)
  - Linear projection 1280 -> 1024 (``encoder_decoder_proj``)
  - 8-layer transformer decoder (hidden_size=1024, 8 heads, head_dim=128)
  - tied logits head ``log_softmax.mlp.layer0`` (Linear 1024 -> 16384)

We trace thin ``nn.Module`` wrappers that only expose the I/O the genai
``cohere_model.cpp`` runtime expects — same naming convention as the previous
private export so the C++ side and ``genai_config.json`` schema stay
unchanged.

Usage::

    python export_cohere_to_onnx.py \
        --model_dir /datadisks/disk2/nebanfic/hf_models/CohereLabs--cohere-transcribe-03-2026 \
        --output_dir /datadisks/disk2/nebanfic/hf_models/cohere-transcribe-onnx-fp32
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Enable faulthandler so we can `kill -SIGUSR1 <pid>` to dump Python stacks of
# every thread on demand (useful when the legacy ONNX exporter spends ages in
# C++ shape inference / constant folding without printing anything).
import faulthandler  # noqa: E402
import signal as _signal  # noqa: E402

faulthandler.enable()
try:
    faulthandler.register(_signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass

_orig_print = print


def print(*a, **kw):  # noqa: A001 - flushed printer for long-running export
    kw.setdefault("flush", True)
    _orig_print(*a, **kw)


import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# onnxscript shims for the dynamo exporter: register no-op translations for
# `aten::sym_storage_offset` / `aten::sym_stride` / `aten::sym_numel` so the
# rel-pos conformer can be traced with a dynamic mel-time axis. These ops
# carry only metadata and have no ONNX runtime semantics.
import onnxscript  # noqa: E402

_op18 = onnxscript.values.Opset("", 18)


@onnxscript.script()
def _sym_storage_offset(self: onnxscript.FLOAT) -> onnxscript.INT64:
    return _op18.Constant(value_int=0)


@onnxscript.script()
def _sym_one(self: onnxscript.FLOAT, dim: int) -> onnxscript.INT64:
    return _op18.Constant(value_int=1)


_DYNAMO_CUSTOM_OPS = {
    "aten::sym_storage_offset": _sym_storage_offset,
    "aten::sym_stride": _sym_one,
    "aten::sym_numel": _sym_one,
}


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class EncoderWrapper(nn.Module):
    """mel [B, 128, T] + mel_length [B] -> cross KV caches.

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


class DecoderWrapper(nn.Module):
    """input_ids + past KV caches -> logits + updated self KV caches.

    Implements the decoder forward in plain ops (no HF cache classes) so that
    ONNX tracing produces a graph with explicit past/present tensors that the
    onnxruntime-genai runtime can manage.
    """

    MAX_SEQ = 1024  # genai static cache length

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
        # Keep past_sequence_length as a tensor so it remains dynamic in ONNX.
        # Cast to int64 so all subsequent index arithmetic / Slice ops use a
        # single int type (mixing int32 and int64 in onnx::Slice's Tind
        # produces "Type parameter (Tind) bound to different types" errors).
        pl = past_sequence_length.reshape(1)[0].to(torch.int64)
        positions = torch.arange(sl, device=input_ids.device) + pl
        hidden = self.embedding(input_ids, positions.unsqueeze(0).expand(b, -1))

        total_len = pl + sl
        # Static-shape causal mask shaped [sl, MAX_SEQ] then sliced to total_len.
        row_idx = torch.arange(sl, device=input_ids.device).unsqueeze(1) + pl
        col_idx = torch.arange(self.MAX_SEQ, device=input_ids.device).unsqueeze(0)
        causal = col_idx > row_idx  # mask future positions
        attn_mask = torch.where(
            causal[:, :total_len],
            torch.tensor(float("-inf"), device=input_ids.device, dtype=hidden.dtype),
            torch.tensor(0.0, device=input_ids.device, dtype=hidden.dtype),
        )

        scale = self.head_dim**-0.5
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
            co = torch.matmul(cw, xv).transpose(1, 2).contiguous().view(
                b, sl, self.hidden_size
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


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def export_encoder(model: nn.Module, out_dir: str, opset: int) -> None:
    """Export the encoder with the dynamo exporter using DYNAMIC mel length.

    The dynamo exporter trips on a few ``aten::sym_*`` ops emitted while
    tracing the rel-pos conformer with dynamic shapes; we register no-op
    translations for them via ``custom_translation_table``. After that the
    ``T_mel`` axis can be dynamic so the genai runtime accepts arbitrary
    audio lengths instead of being pinned to 35s.
    """
    import time

    print("\n  Exporting encoder (dynamo, dynamic T_mel)...")
    t0 = time.time()
    wrapper = EncoderWrapper(model).float().eval()
    nl = wrapper.num_layers

    # Small dummy length is fine — dynamic axis means ORT will accept any T at runtime.
    dummy_t = 300
    mel = torch.randn(1, 128, dummy_t, dtype=torch.float32)
    mel_length = torch.tensor([dummy_t], dtype=torch.int64)

    out_names = [f"present_key_cross_{i}" for i in range(nl)] + [
        f"present_value_cross_{i}" for i in range(nl)
    ]
    dynamic_axes = {
        "mel": {0: "batch", 2: "T_mel"},
        "mel_length": {0: "batch"},
    }
    for n in out_names:
        dynamic_axes[n] = {0: "batch", 2: "T_enc"}

    print(f"  [t={time.time()-t0:.1f}s] sanity forward (shape={tuple(mel.shape)})...")
    with torch.no_grad():
        sample_out = wrapper(mel, mel_length)
    print(
        f"  [t={time.time()-t0:.1f}s] forward OK, "
        f"first KV shape={tuple(sample_out[0].shape)}"
    )
    del sample_out

    path = os.path.join(out_dir, "encoder.onnx")
    print(f"  [t={time.time()-t0:.1f}s] calling torch.onnx.export(dynamo=True)...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (mel, mel_length),
            path,
            input_names=["mel", "mel_length"],
            output_names=out_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=True,
            do_constant_folding=True,
            custom_translation_table=_DYNAMO_CUSTOM_OPS,
            verbose=False,
        )
    print(f"  [t={time.time()-t0:.1f}s] torch.onnx.export returned.")

    # External-data save (consolidated single .data file).
    import onnx

    m = onnx.load(path)
    onnx.save(
        m,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=1024,
    )

    if os.path.exists(path):
        sz = os.path.getsize(path)
        if os.path.exists(path + ".data"):
            sz += os.path.getsize(path + ".data")
        print(f"  [t={time.time()-t0:.1f}s] encoder.onnx written ({sz/1e6:.0f} MB)")


def export_decoder(model: nn.Module, out_dir: str, opset: int) -> None:
    print("\n  Exporting decoder...")
    wrapper = DecoderWrapper(model).float().eval()
    nl = wrapper.num_layers
    nh = wrapper.num_heads
    hd = wrapper.head_dim
    ms = wrapper.MAX_SEQ

    input_ids = torch.tensor([[7]], dtype=torch.int32)
    past_seq_len = torch.tensor([0], dtype=torch.int32)
    past_self = [torch.zeros(1, nh, ms, hd) for _ in range(nl * 2)]
    past_cross = [torch.randn(1, nh, 100, hd) for _ in range(nl * 2)]

    inputs = ["input_ids", "past_sequence_length"]
    for i in range(nl):
        inputs += [f"past_key_self_{i}", f"past_value_self_{i}"]
    for i in range(nl):
        inputs += [f"past_key_cross_{i}", f"past_value_cross_{i}"]

    outputs = ["logits"]
    for i in range(nl):
        outputs += [f"present_key_self_{i}", f"present_value_self_{i}"]

    dynamic = {
        "input_ids": {0: "batch", 1: "n_tokens"},
        "logits": {0: "batch", 1: "n_tokens"},
    }
    for i in range(nl):
        dynamic[f"past_key_cross_{i}"] = {0: "batch", 2: "T_enc"}
        dynamic[f"past_value_cross_{i}"] = {0: "batch", 2: "T_enc"}

    path = os.path.join(out_dir, "decoder.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, past_seq_len, *past_self, *past_cross),
            path,
            input_names=inputs,
            output_names=outputs,
            dynamic_axes=dynamic,
            opset_version=opset,
            do_constant_folding=False,
            dynamo=False,
        )

    import onnx

    model_proto = onnx.load(path)
    if os.path.getsize(path) > 100 * 1024 * 1024:
        onnx.save(
            model_proto,
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="decoder.onnx.data",
            size_threshold=1024,
        )
    size_mb = os.path.getsize(path) / 1e6
    if os.path.exists(path + ".data"):
        size_mb += os.path.getsize(path + ".data") / 1e6
    print(f"  decoder.onnx written ({size_mb:.0f} MB)")


def write_config_and_tokenizer(
    model_dir: str, out_dir: str, num_decoder_layers: int
) -> None:
    cfg = {
        "model": {
            "bos_token_id": 7,
            "context_length": 1024,
            "decoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "decoder.onnx",
                "head_size": 128,
                "hidden_size": 1024,
                "inputs": {
                    "input_ids": "input_ids",
                    "past_key_names": "past_key_self_%d",
                    "past_value_names": "past_value_self_%d",
                    "cross_past_key_names": "past_key_cross_%d",
                    "cross_past_value_names": "past_value_cross_%d",
                    "past_sequence_length": "past_sequence_length",
                },
                "outputs": {
                    "logits": "logits",
                    "present_key_names": "present_key_self_%d",
                    "present_value_names": "present_value_self_%d",
                },
                "num_attention_heads": 8,
                "num_hidden_layers": num_decoder_layers,
                "num_key_value_heads": 8,
            },
            "encoder": {
                "session_options": {
                    "log_id": "onnxruntime-genai",
                    "provider_options": [],
                },
                "filename": "encoder.onnx",
                "head_size": 128,
                "hidden_size": 1024,
                "audio_stride": 1280,
                "inputs": {"audio_features": "mel"},
                "outputs": {
                    "cross_present_key_names": "present_key_cross_%d",
                    "cross_present_value_names": "present_value_cross_%d",
                },
                "num_attention_heads": 8,
                "num_hidden_layers": num_decoder_layers,
            },
            "eos_token_id": 3,
            "pad_token_id": 3,
            "type": "cohere_transcribe",
            "vocab_size": 16384,
            "num_mels": 128,
            "fft_size": 512,
            "hop_length": 160,
            "win_length": 400,
            "sample_rate": 16000,
            "preemph": 0.97,
            "log_eps": 5.96046448e-08,
            "norm_eps": 1e-05,
        },
        "search": {
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": True,
            "length_penalty": 1.0,
            "max_length": 1024,
            "min_length": 0,
            "no_repeat_ngram_size": 0,
            "num_beams": 1,
            "num_return_sequences": 1,
            "past_present_share_buffer": True,
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
        },
    }
    with open(os.path.join(out_dir, "genai_config.json"), "w") as f:
        json.dump(cfg, f, indent=4)

    with open(os.path.join(out_dir, "audio_processor_config.json"), "w") as f:
        json.dump(
            {
                "feature_extraction": {
                    "sequence": [
                        {"operation": {"name": "audio_decoder", "type": "AudioDecoder"}},
                        {
                            "operation": {
                                "name": "STFT",
                                "type": "STFTNorm",
                                "attrs": {
                                    "n_fft": 512,
                                    "frame_length": 400,
                                    "hop_length": 160,
                                },
                            }
                        },
                        {
                            "operation": {
                                "name": "log_mel_spectrogram",
                                "type": "LogMelSpectrum",
                                "attrs": {
                                    "n_fft": 512,
                                    "n_mel": 128,
                                    "hop_length": 160,
                                    "chunk_size": 30,
                                    "feature_first": 1,
                                    "no_padding": 1,
                                },
                            }
                        },
                    ]
                }
            },
            f,
            indent=4,
        )

    for fn in (
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ):
        src = os.path.join(model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out_dir, fn))

    # TODO: see should we use CohereTokenizer
    # Rewrite tokenizer_class so onnxruntime-genai's tokenizer factory accepts
    # it. The HF model declares ``CohereAsrTokenizer`` (a custom trust_remote_code
    # subclass), but genai only recognizes a fixed set of classes. The
    # underlying tokenizer.json is a standard SentencePiece-style tokenizer and
    # works fine when loaded as a WhisperTokenizer.
    tcfg_path = os.path.join(out_dir, "tokenizer_config.json")
    if os.path.exists(tcfg_path):
        with open(tcfg_path) as f:
            tcfg = json.load(f)
        tcfg["tokenizer_class"] = "WhisperTokenizer"
        # Drop auto_map so HF doesn't try to re-import the trust_remote_code class.
        tcfg.pop("auto_map", None)
        with open(tcfg_path, "w") as f:
            json.dump(tcfg, f, indent=2)

    tj = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tj):
        with open(tj) as f:
            td = json.load(f)
        tokens: dict[str, int] = {}
        for it in td.get("model", {}).get("vocab", []):
            if isinstance(it, list) and len(it) == 2:
                tokens[it[0]] = len(tokens)
        for it in td.get("added_tokens", []):
            if isinstance(it, dict):
                tokens[it["content"]] = it["id"]
        with open(os.path.join(out_dir, "tokens.txt"), "w", encoding="utf-8") as f:
            for tok, idx in sorted(tokens.items(), key=lambda x: x[1]):
                f.write(f"{tok} {idx}\n")
        print(f"  tokens.txt: {len(tokens)} entries")
    print("  config + tokenizer done")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--model_dir",
        required=True,
        help="Path to the downloaded CohereLabs/cohere-transcribe-03-2026 snapshot.",
    )
    p.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write encoder.onnx, decoder.onnx and genai config.",
    )
    p.add_argument("--opset", type=int, default=17)
    p.add_argument(
        "--skip_decoder",
        action="store_true",
        help="Only export the encoder (useful for iterating on encoder issues).",
    )
    p.add_argument(
        "--skip_encoder",
        action="store_true",
        help="Skip encoder export (reuse existing encoder.onnx).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Cohere Transcribe ONNX export")
    print("=" * 60)
    print(f"  Model:  {args.model_dir}")
    print(f"  Output: {args.output_dir}")

    print("\n  Loading model (trust_remote_code=True)...")
    sys.path.insert(0, args.model_dir)
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

    cfg = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_dir,
            config=cfg,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
    except TypeError:
        # Some custom model classes don't accept torch_dtype kwarg.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model_dir,
            config=cfg,
            trust_remote_code=True,
        ).to(torch.float32)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded: {n_params / 1e6:.0f}M params")

    num_decoder_layers = len(model.transf_decoder._decoder.layers)

    export_encoder(model, args.output_dir, args.opset) if not args.skip_encoder else print("\n  Skipping encoder export (--skip_encoder)")
    if not args.skip_decoder:
        export_decoder(model, args.output_dir, args.opset)
    write_config_and_tokenizer(args.model_dir, args.output_dir, num_decoder_layers)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    for f in sorted(Path(args.output_dir).iterdir()):
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()
