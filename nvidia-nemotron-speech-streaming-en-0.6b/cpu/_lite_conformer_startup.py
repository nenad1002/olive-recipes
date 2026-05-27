# Activated via PYTHONSTARTUP by optimize_de200m.py only.
# Applies LiteConformer monkey-patches to NeMo Conformer modules before
# any model is loaded. Scoped to the de-200m export subprocesses.
import sys
sys.path.insert(0, "/home/nebanfic")
try:
    from lite_conformer import apply_lite_conformer_patches
    apply_lite_conformer_patches()
except Exception as e:
    print(f"[lite_conformer_startup] failed: {e}", file=sys.stderr)
