"""ZH-512 variant of optimize.py — wraps the same stages but uses
nemotron_speech_int4_cpu_zh512.json and points at the patched local .nemo."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
_EXPORT_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_nemotron_to_onnx_static_shape.py"
_TOKENIZER_SCRIPT = _SCRIPT_DIR.parent / "scripts" / "export_tokenizer.py"

DEFAULT_MODEL = "/datadisks/disk2/nebanfic/hf_models/nemotron-zh-512/best_model_fixed.nemo"
DEFAULT_EXPORT_DIR = "build/zh_fp32"
DEFAULT_OUTPUT_DIR = "build/zh_int4"
CONFIG_NAME = "nemotron_speech_int4_cpu_zh512.json"


def _resolve(path):
    p = Path(path)
    return p if p.is_absolute() else _SCRIPT_DIR / p


def run_export(model_name, export_dir, chunk_size, left_chunks):
    print(f"=== Stage 1: NeMo → ONNX (fp32) ===")
    subprocess.run([
        sys.executable, str(_EXPORT_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(export_dir)),
        "--streaming", "--chunk_size", str(chunk_size),
        "--left_chunks", str(left_chunks),
        "--device", "cpu",
    ], check=True, cwd=str(_SCRIPT_DIR))
    print("=== Stage 1b: Tokenizer ===")
    subprocess.run([
        sys.executable, str(_TOKENIZER_SCRIPT),
        "--model_name", model_name,
        "--output_dir", str(_resolve(export_dir)),
    ], check=True, cwd=str(_SCRIPT_DIR))


def run_olive_encoder(output_dir):
    from olive import run as olive_run
    print("=== Stage 2: Olive encoder (convert → fuse → INT4) ===")
    config_path = _SCRIPT_DIR / CONFIG_NAME
    config = json.loads(config_path.read_text())
    config["output_dir"] = str(_resolve(output_dir) / "encoder.onnx")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", dir=str(_SCRIPT_DIR), delete=False
    ) as tmp:
        json.dump(config, tmp, indent=2)
        tmp_path = tmp.name
    try:
        olive_run(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def download_silero(output_dir):
    from huggingface_hub import hf_hub_download
    dst_dir = _resolve(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "silero_vad.onnx"
    cached = hf_hub_download(repo_id="onnx-community/silero-vad", filename="onnx/model.onnx")
    shutil.copy2(cached, str(dst))
    print(f"  Silero VAD → {dst}")


def copy_supporting(export_dir, output_dir):
    src = _resolve(export_dir)
    dst = _resolve(output_dir)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.name.startswith("encoder"):
            continue
        shutil.copy2(str(f), str(dst / f.name))
        n += 1
    print(f"  Copied {n} supporting files → {dst}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--export-dir", default=DEFAULT_EXPORT_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--chunk-size", type=float, default=0.56)
    p.add_argument("--left-chunks", type=int, default=10)
    p.add_argument("--skip-export", action="store_true")
    args = p.parse_args()

    if not args.skip_export:
        run_export(args.model_name, args.export_dir, args.chunk_size, args.left_chunks)
    run_olive_encoder(args.output_dir)
    copy_supporting(args.export_dir, args.output_dir)
    download_silero(args.output_dir)
    print("\nDone. Output:", _resolve(args.output_dir))


if __name__ == "__main__":
    main()
