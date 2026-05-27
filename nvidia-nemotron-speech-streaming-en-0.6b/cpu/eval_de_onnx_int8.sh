#!/bin/bash
set -e

RECIPE_DIR=/home/nebanfic/myOliveRecipes/olive-recipes/nvidia-nemotron-speech-streaming-en-0.6b
SCRIPT=$RECIPE_DIR/cpu/eval_onnx_de.py
MODEL_DIR=${MODEL_DIR:-$RECIPE_DIR/cpu/build/de_int8_full}

# Force CPU-only
export CUDA_VISIBLE_DEVICES=""

for ds in fleurs mls; do
    echo ""
    echo "================================================================="
    echo "  $(basename $MODEL_DIR) / $ds  (CPU, ONNX)"
    echo "================================================================="
    python3 "$SCRIPT" --model_dir "$MODEL_DIR" --dataset "$ds" --lang de_de
done
