#!/usr/bin/env python3
"""End-to-end optimization pipeline for Cohere Transcribe.

The encoder and decoder are exported through Olive's declarative pass system:

  - Encoder: ``OnnxConversion`` (dynamo, FP32, dynamic mel time axis)
  - Decoder: ``OnnxConversion`` (legacy torchscript, FP32)

After the Olive pipelines, tokenizer and config files are generated and
Silero VAD is downloaded. Fusion and quantization are intentionally omitted
in this first cut and will be added in a follow-up.

Usage::

    # Full pipeline
    python scripts/optimize.py --output-dir build/onnx_models_fp32

    # Or use Olive CLI directly for an individual component
    python -m olive run --config cpu/cohere_encoder_fp32_cpu.json
    python -m olive run --config cpu/cohere_decoder_fp32_cpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make ``cpu.cohere_model_load`` importable regardless of where the script is
# invoked from.
_SCRIPT_DIR = Path(__file__).resolve().parent
_RECIPE_ROOT = _SCRIPT_DIR.parent
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

_CPU_DIR = _RECIPE_ROOT / "cpu"

DEFAULT_OUTPUT_DIR = "build/onnx_models_fp32"


def _resolve(path: str) -> Path:
    """Resolve a path relative to the recipe root."""
    p = Path(path)
    return p if p.is_absolute() else _RECIPE_ROOT / p


def _run_olive_pipeline(config_name: str, output_path: Path) -> None:
    """Run an Olive pipeline from a JSON config, overriding ``output_dir``."""
    from olive import run as olive_run

    config_path = _CPU_DIR / config_name
    with open(config_path) as f:
        config = json.load(f)

    config["output_dir"] = str(output_path)

    # Olive resolves ``model_script`` relative to the cwd, so write the
    # temporary config next to ``cohere_model_load.py`` and run from there.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_CPU_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=4)
        tmp_path = Path(tmp.name)

    cwd = os.getcwd()
    try:
        os.chdir(_RECIPE_ROOT)
        olive_run(str(tmp_path))
    finally:
        os.chdir(cwd)
        tmp_path.unlink(missing_ok=True)

    if not output_path.exists():
        raise RuntimeError(
            f"Olive pipeline '{config_name}' did not produce expected output at "
            f"{output_path}. Check the log above for the failing pass (common "
            f"causes: ENOSPC on the cache disk, missing dependencies)."
        )


def _prune_unused_initializers(model_path: Path, external_data_name: str) -> None:
    """Drop initializers not referenced by any node and rewrite external data.

    ``OnnxBlockWiseRtnQuantization`` rewires consumers from ``MatMul`` to
    ``MatMulNBits`` but leaves the original FP32 weight initializers in the
    graph. They get serialized into the external-data blob, bloating the file
    far beyond the int4 size.
    """
    import onnx

    print(f"=== Pruning unused initializers in {model_path.name} ===")
    model = onnx.load(str(model_path), load_external_data=True)
    used: set[str] = set()
    for n in model.graph.node:
        used.update(n.input)
    keep = [init for init in model.graph.initializer if init.name in used]
    dropped = len(model.graph.initializer) - len(keep)
    if dropped == 0:
        print("  (no orphan initializers)")
        return
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)

    # Re-save with external data, overwriting the existing blob.
    data_file = model_path.parent / external_data_name
    if data_file.exists():
        data_file.unlink()
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=1024,
    )
    new_mb = data_file.stat().st_size / (1024 * 1024) if data_file.exists() else 0
    print(f"  pruned {dropped} initializers; new {external_data_name}: {new_mb:.1f} MB")


def _export_and_quantize_encoder_int4(
    model_name: str, output_dir: Path, block_size: int = 32
) -> None:
    """Encoder int4: standalone dynamo export → MatMul4Bits quantization.

    Olive's ``OnnxConversion`` does not forward ``custom_translation_table``
    to ``torch.onnx.export(dynamo=True)``, so the conformer's symbolic shape
    ops cause a RecursionError. We bypass Olive for the encoder and use the
    proven standalone exporter (``scripts/export_cohere_to_onnx.py``) which
    registers the needed onnxscript shims, then run the same K-quant
    quantizer that Olive's ``OnnxBlockWiseRtnQuantization`` wraps.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )

    sys.path.insert(0, str(_SCRIPT_DIR))
    from export_cohere_to_onnx import export_encoder
    from cpu.cohere_model_load import _load_hf_model

    print("=== Stage 1a: Standalone dynamo export (fp32 encoder) ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_hf_model(model_name)

    # torch >= 2.10's torch.export rejects the ``view`` calls in the
    # conformer's ``rel_shift`` because the shape relations are
    # data-dependent (GuardOnDataDependentSymNode). Patching ``view`` to
    # ``reshape`` is semantically equivalent (the tensors are contiguous
    # after the pad) and exports cleanly. Patch every attention module's
    # bound method so all conformer layers pick it up.
    import types as _types
    import torch.nn.functional as _F

    def _rel_shift_reshape(self, x):
        b, h, qlen, pos_len = x.size()
        x = _F.pad(x, pad=(1, 0))
        x = x.reshape(b, h, -1, qlen)
        x = x[:, :, 1:].reshape(b, h, qlen, pos_len)
        return x

    patched = 0
    for m in model.modules():
        if hasattr(m, "rel_shift") and callable(getattr(m, "rel_shift")):
            m.rel_shift = _types.MethodType(_rel_shift_reshape, m)
            patched += 1
    print(f"  patched rel_shift on {patched} attention modules")

    export_encoder(model, str(output_dir), opset=17)
    del model

    enc_path = output_dir / "encoder.onnx"
    print("\n=== Stage 1b: K-quant int4 quantization (block_size=%d) ===" % block_size)
    m = onnx.load(str(enc_path), load_external_data=True)
    quantizer = MatMulNBitsQuantizer(
        model=m,
        bits=4,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        algo_config=RTNWeightOnlyQuantConfig(),
    )
    quantizer.process()
    qmodel = quantizer.model.model

    # Re-save with consolidated external data, replacing the fp32 blob.
    data_file = output_dir / "encoder.onnx.data"
    if data_file.exists():
        data_file.unlink()
    onnx.save_model(
        qmodel,
        str(enc_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="encoder.onnx.data",
        size_threshold=1024,
    )
    sz = data_file.stat().st_size / (1024 * 1024) if data_file.exists() else 0
    print(f"  encoder.onnx + .data ({sz:.0f} MB)")


def run_olive_pipelines(output_dir: Path) -> None:
    """Run encoder and decoder Olive pipelines."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: Olive Encoder (OnnxConversion, FP32) ===")
    _run_olive_pipeline(
        "cohere_encoder_fp32_cpu.json", output_dir / "encoder.onnx"
    )
    print()

    print("=== Stage 2: Olive Decoder (OnnxConversion, FP32) ===")
    _run_olive_pipeline(
        "cohere_decoder_fp32_cpu.json", output_dir / "decoder.onnx"
    )
    print()


# ---------------------------------------------------------------------------
# Tokenizer + configs
# ---------------------------------------------------------------------------
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
)


def _hf_snapshot(model_name: str) -> Path:
    """Resolve a HuggingFace repo id (or local dir) to a local snapshot path."""
    if Path(model_name).exists():
        return Path(model_name)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=model_name, allow_patterns=list(_TOKENIZER_FILES))
    )


def export_tokenizer(model_name: str, output_dir: Path) -> None:
    """Copy HF tokenizer files, fix up ``tokenizer_class`` and emit ``tokens.txt``.

    ``onnxruntime-genai``'s tokenizer factory only recognizes a fixed set of
    classes. The HF model declares ``CohereAsrTokenizer`` (a custom
    trust_remote_code subclass), but the underlying tokenizer.json is a
    standard SentencePiece-style tokenizer and works fine when loaded as a
    ``WhisperTokenizer``.
    """
    print("=== Stage 3: Exporting tokenizer ===")
    src = _hf_snapshot(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fn in _TOKENIZER_FILES:
        s = src / fn
        if s.exists():
            shutil.copy2(s, output_dir / fn)

    tcfg_path = output_dir / "tokenizer_config.json"
    if tcfg_path.exists():
        with open(tcfg_path) as f:
            tcfg = json.load(f)
        tcfg["tokenizer_class"] = "WhisperTokenizer"
        # Drop auto_map so HF doesn't try to re-import the trust_remote_code class.
        tcfg.pop("auto_map", None)
        with open(tcfg_path, "w") as f:
            json.dump(tcfg, f, indent=2)

    tj = src / "tokenizer.json"
    if tj.exists():
        with open(tj) as f:
            td = json.load(f)
        tokens: dict[str, int] = {}
        for it in td.get("model", {}).get("vocab", []):
            if isinstance(it, list) and len(it) == 2:
                tokens[it[0]] = len(tokens)
        for it in td.get("added_tokens", []):
            if isinstance(it, dict):
                tokens[it["content"]] = it["id"]
        with open(output_dir / "tokens.txt", "w", encoding="utf-8") as f:
            for tok, idx in sorted(tokens.items(), key=lambda x: x[1]):
                f.write(f"{tok} {idx}\n")
        print(f"  tokens.txt: {len(tokens)} entries")
    print()


def generate_configs(output_dir: Path, num_decoder_layers: int) -> None:
    """Write ``genai_config.json`` and ``audio_processor_config.json``."""
    print("=== Stage 4: Generating config files ===")
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
            "vad": {
                "filename": "silero_vad.onnx",
                "threshold": 0.3,
                "silence_duration_ms": 3360,
                "prefix_padding_ms": 560,
            },
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
    with open(output_dir / "genai_config.json", "w") as f:
        json.dump(cfg, f, indent=4)
    print("  [OK] genai_config.json")

    audio_cfg = {
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
    }
    with open(output_dir / "audio_processor_config.json", "w") as f:
        json.dump(audio_cfg, f, indent=4)
    print("  [OK] audio_processor_config.json")
    print()


def download_silero_vad(output_dir: Path) -> None:
    """Download the Silero VAD ONNX model from ``onnx-community/silero-vad``."""
    from huggingface_hub import hf_hub_download

    print("=== Stage 5: Downloading Silero VAD ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / "silero_vad.onnx"

    cached = hf_hub_download(
        repo_id="onnx-community/silero-vad",
        filename="onnx/model.onnx",
    )
    shutil.copy2(cached, str(dst))
    size_mb = dst.stat().st_size / (1024 * 1024)
    print(f"  Saved Silero VAD model to: {dst} ({size_mb:.1f} MB)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    from cpu.cohere_model_load import MODEL_NAME, NUM_DECODER_LAYERS

    parser = argparse.ArgumentParser(
        description="Optimize Cohere Transcribe for CPU inference (FP32)."
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help=(
            "HuggingFace repo id or local snapshot path of the cohere-transcribe "
            "model (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for the exported artifacts (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--skip-encoder",
        action="store_true",
        help="Skip the encoder Olive pipeline.",
    )
    parser.add_argument(
        "--encoder-precision",
        choices=["fp32", "int4"],
        default="fp32",
        help=(
            "Encoder precision. 'fp32' runs OnnxConversion only; 'int4' adds "
            "OnnxKQuantQuantization (block_size=32, MatMulNBits) on top."
        ),
    )
    parser.add_argument(
        "--skip-decoder",
        action="store_true",
        help="Skip the decoder Olive pipeline.",
    )
    parser.add_argument(
        "--skip-vad",
        action="store_true",
        help="Skip downloading Silero VAD.",
    )
    args = parser.parse_args()

    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Cohere Transcribe — Olive optimize (FP32 CPU)")
    print("=" * 60)
    print(f"  Model:  {args.model_name}")
    print(f"  Output: {output_dir}")
    print()

    if not args.skip_encoder:
        if args.encoder_precision == "int4":
            _export_and_quantize_encoder_int4(args.model_name, output_dir)
        else:
            encoder_cfg = "cohere_encoder_fp32_cpu.json"
            print(f"=== Encoder pipeline: {encoder_cfg} ===")
            _run_olive_pipeline(encoder_cfg, output_dir / "encoder.onnx")
    else:
        print("=== Skipping encoder Olive pipeline ===\n")

    if not args.skip_decoder:
        _run_olive_pipeline("cohere_decoder_fp32_cpu.json", output_dir / "decoder.onnx")
    else:
        print("=== Skipping decoder Olive pipeline ===\n")

    export_tokenizer(args.model_name, output_dir)
    generate_configs(output_dir, num_decoder_layers=NUM_DECODER_LAYERS)

    if not args.skip_vad:
        try:
            download_silero_vad(output_dir)
        except Exception as exc:
            print(
                f"  Warning: Silero VAD download failed ({exc}).\n"
                f"  Download manually from https://huggingface.co/onnx-community/silero-vad\n"
                f"  and place silero_vad.onnx at: {output_dir / 'silero_vad.onnx'}"
            )

    # Summary
    if output_dir.exists():
        files = sorted(f for f in output_dir.iterdir() if f.is_file())
        total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        print(f"=== Done! Optimized artifacts → {output_dir} ===")
        print(f"    Total size: {total_mb:.1f} MB")
        for f in files:
            print(f"    {f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
