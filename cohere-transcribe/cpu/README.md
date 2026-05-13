# Cohere Transcribe (CPU EP, FP32)

This recipe exports **CohereLabs/cohere-transcribe-03-2026** to ONNX for CPU
inference. Only the FP32 export path is wired up — graph fusion and
quantization will be added in a later iteration.

All model components are handled through Olive's declarative pass system:

- **Encoder**: `OnnxConversion` (dynamo, dynamic mel time axis)
- **Decoder**: `OnnxConversion` (legacy torchscript)

## Files
- `cpu/cohere_encoder_fp32_cpu.json` – Olive encoder config
- `cpu/cohere_decoder_fp32_cpu.json` – Olive decoder config
- `cpu/cohere_model_load.py` – model loaders + dummy inputs for both components
- `scripts/optimize.py` – full pipeline (Olive × 2 + tokenizer + configs + VAD)
- `scripts/export_cohere_to_onnx.py` – legacy stand-alone export script (kept for reference)

## Setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r cohere-transcribe/cpu/requirements.txt
```

## Run

From the `cohere-transcribe` directory:

```bash
cd cohere-transcribe
python scripts/optimize.py --output-dir build/onnx_models_fp32
```

This runs the full pipeline:

1. **Encoder** — Olive: `OnnxConversion` (FP32, dynamic mel)
2. **Decoder** — Olive: `OnnxConversion` (FP32)
3. **Tokenizer** — copies HF tokenizer files and rewrites `tokenizer_class`
4. **Configs** — generates `genai_config.json` + `audio_processor_config.json`
5. **VAD** — downloads Silero VAD ONNX model

Or run individual components directly with the Olive CLI:

```bash
python -m olive run --config cpu/cohere_encoder_fp32_cpu.json
python -m olive run --config cpu/cohere_decoder_fp32_cpu.json
```

## Output

Expected artifacts in `build/onnx_models_fp32/`:

- `encoder.onnx` (+ `encoder.onnx.data`, FP32)
- `decoder.onnx` (+ `decoder.onnx.data`, FP32)
- `silero_vad.onnx`
- `genai_config.json`
- `audio_processor_config.json`
- `tokenizer.json`, `tokenizer_config.json`, `tokenizer.model`, `special_tokens_map.json`
- `tokens.txt`
