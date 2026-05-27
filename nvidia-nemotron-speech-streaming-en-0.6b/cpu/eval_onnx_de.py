#!/usr/bin/env python3
"""Self-contained WER eval for the lite-conformer-de ONNX-genai build.

Runs on CPU. Uses onnxruntime-genai streaming exactly like nemotron_speech.py.
Supports FLEURS de_de and MLS german out of the box.

Usage:
  python eval_onnx_de.py --model_dir cpu/build/de_int8_full --dataset fleurs --output_dir cpu/build/de_int8_full/eval_fleurs
  python eval_onnx_de.py --model_dir cpu/build/de_int8_full --dataset mls    --output_dir cpu/build/de_int8_full/eval_mls
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np


# ---------------- text normalization ----------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


# ---------------- WER with S/D/I breakdown ----------------

def compute_sdi(ref_words, hyp_words):
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1; i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            dels += 1; i -= 1
        else:
            ins += 1; j -= 1
    return subs, dels, ins


# ---------------- ONNX-genai streaming transcribe ----------------

def transcribe_audio(audio_f32: np.ndarray, model, tokenizer, params, chunk_samples: int) -> str:
    import onnxruntime_genai as og
    processor = og.StreamingProcessor(model)
    generator = og.Generator(model, params)
    tok_stream = tokenizer.create_stream()
    out = []
    for start in range(0, len(audio_f32), chunk_samples):
        chunk = audio_f32[start:start + chunk_samples].astype(np.float32)
        inputs = processor.process(chunk)
        if inputs is not None:
            generator.set_inputs(inputs)
            while not generator.is_done():
                generator.generate_next_token()
                toks = generator.get_next_tokens()
                if len(toks) > 0:
                    t = tok_stream.decode(toks[0])
                    if t:
                        out.append(t)
    inputs = processor.flush()
    if inputs is not None:
        generator.set_inputs(inputs)
        while not generator.is_done():
            generator.generate_next_token()
            toks = generator.get_next_tokens()
            if len(toks) > 0:
                t = tok_stream.decode(toks[0])
                if t:
                    out.append(t)
    del generator
    del processor
    return "".join(out).strip()


# ---------------- dataset loaders ----------------

def load_dataset_split(name: str, lang: str = "de_de", cache_dir: str | None = None,
                       cv_local: str | None = None):
    from datasets import load_dataset, Audio
    if name == "fleurs":
        ds = load_dataset("google/fleurs", lang, split="test",
                          trust_remote_code=True, cache_dir=cache_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds, "transcription"
    if name == "mls":
        mls_lang = {"de_de": "german"}.get(lang, lang)
        ds = load_dataset("facebook/multilingual_librispeech", mls_lang, split="test",
                          cache_dir=cache_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds, "transcript"
    if name == "voxpopuli":
        vp_lang = {"de_de": "de"}.get(lang, lang)
        # Prefer pre-saved local copy (~2.2GB for de) created by the
        # ForeignlangNemotronTraining eval scripts via Dataset.save_to_disk.
        local_dir = os.environ.get("VOXPOPULI_SAVE_BASE",
                                   "/datadisks/disk4/nebanfic/voxpopuli") + f"_{vp_lang}_test"
        saved_path = os.path.join(local_dir, "dataset")
        if os.path.exists(saved_path):
            from datasets import load_from_disk
            ds = load_from_disk(saved_path)
            # Note: do NOT cast_column("audio", Audio(...)) here — the saved
            # copy stores audio as a plain Sequence[float64] dict (not the HF
            # Audio feature), so casting would rewrite all ~2.2GB of arrow
            # files. The main loop already handles sample["audio"]["array"]
            # / sample["audio"]["sampling_rate"] dicts directly.
        else:
            # NOTE: do NOT pass trust_remote_code=True — that uses the legacy
            # loader script which downloads ALL splits (~50GB for de).
            ds = load_dataset("facebook/voxpopuli", vp_lang, split="test",
                              cache_dir=cache_dir)
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ref_field = "normalized_text" if "normalized_text" in ds.column_names else "raw_text"
        return ds, ref_field
    if name == "cv":
        if cv_local:
            return _load_local_cv(cv_local), "sentence"
        cv_lang = {"de_de": "de"}.get(lang, lang)
        ds = load_dataset("mozilla-foundation/common_voice_17_0", cv_lang,
                          split="test", trust_remote_code=True, cache_dir=cache_dir)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds, "sentence"
    raise ValueError(f"unsupported dataset: {name}")


def _load_local_cv(cv_dir: str):
    """Iterable dataset over a local CV directory with test.tsv + clips/.
    Yields {audio: {array, sampling_rate}, sentence, gender, age, accents, client_id}.
    """
    import soundfile as sf

    tsv_path = os.path.join(cv_dir, "test.tsv")
    clips_dir = os.path.join(cv_dir, "clips")
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    idx = {k: header.index(k) if k in header else -1
           for k in ("path", "sentence", "gender", "age", "accents", "accent", "client_id")}

    samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(idx["path"], idx["sentence"]):
                continue
            clip_file = parts[idx["path"]]
            clip_path = os.path.join(clips_dir, clip_file)
            if not os.path.exists(clip_path):
                continue
            samples.append({
                "audio_path": clip_path,
                "sentence": parts[idx["sentence"]],
                "gender": parts[idx["gender"]] if idx["gender"] >= 0 and idx["gender"] < len(parts) else "",
                "age": parts[idx["age"]] if idx["age"] >= 0 and idx["age"] < len(parts) else "",
                "accents": (parts[idx["accents"]] if idx["accents"] >= 0 and idx["accents"] < len(parts)
                            else (parts[idx["accent"]] if idx["accent"] >= 0 and idx["accent"] < len(parts) else "")),
                "client_id": parts[idx["client_id"]] if idx["client_id"] >= 0 and idx["client_id"] < len(parts) else "",
            })

    class LocalCV:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            s = self.items[i]
            audio, sr = sf.read(s["audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                new_len = int(len(audio) * 16000 / sr)
                audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                                  np.arange(len(audio)), audio).astype(np.float32)
                sr = 16000
            return {
                "audio": {"array": audio, "sampling_rate": sr},
                "sentence": s["sentence"],
                "gender": s["gender"],
                "age": s["age"],
                "accents": s["accents"],
                "client_id": s["client_id"],
            }

    return LocalCV(samples)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--dataset", choices=["fleurs", "mls", "cv", "voxpopuli"], required=True)
    ap.add_argument("--lang", default="de_de")
    ap.add_argument("--cv_local", default="/datadisks/disk3/nebanfic/cv_de_test/cv-corpus-25.0-2026-03-09/de",
                    help="Path to local CV dir with test.tsv + clips/. Used only for --dataset cv.")
    ap.add_argument("--output_dir", default=None,
                    help="Write per-utt TSV + summary JSON here. Default: <model_dir>/eval_<dataset>")
    ap.add_argument("--limit", type=int, default=None, help="Eval at most N samples")
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--cache_dir", default=os.environ.get("HF_DATASETS_CACHE")
                    or os.environ.get("HF_HOME")
                    or "/datadisks/disk3/nebanfic/hf_cache")
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.model_dir) / f"eval_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read chunk_samples + sample_rate from genai_config
    cfg = json.loads((Path(args.model_dir) / "genai_config.json").read_text())
    sample_rate = cfg["model"]["sample_rate"]
    chunk_samples = cfg["model"]["chunk_samples"]
    print(f"  sample_rate={sample_rate} chunk_samples={chunk_samples}")

    import onnxruntime_genai as og
    print(f"  Loading model from {args.model_dir} (CPU)...")
    model = og.Model(args.model_dir)
    tokenizer = og.Tokenizer(model)
    params = og.GeneratorParams(model)

    print(f"  Loading dataset {args.dataset} ({args.lang})...")
    ds, ref_field = load_dataset_split(args.dataset, args.lang, args.cache_dir,
                                       cv_local=args.cv_local)
    n_total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"  Samples: {n_total}")

    tsv_path = out_dir / "utterances.tsv"
    summary_path = out_dir / "summary.json"
    tsv = open(tsv_path, "w", encoding="utf-8")
    has_demo = args.dataset == "cv"
    if has_demo:
        tsv.write("idx\tgender\tage\taccents\tclient_id\tn_ref\tsubs\tdels\tins\tref\thyp\n")
    else:
        tsv.write("idx\tn_ref\tsubs\tdels\tins\tref\thyp\n")

    total_s = total_d = total_i = total_w = errors = 0
    audio_secs = 0.0
    # subgroup buckets: key -> {value -> [subs, dels, ins, words, n_utt]}
    subgroups: dict[str, dict[str, list[int]]] = {"gender": {}, "age": {}, "accents": {}}
    t0 = time.time()

    for idx in range(n_total):
        sample = ds[idx]
        try:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != sample_rate:
                new_len = int(len(audio) * sample_rate / sr)
                audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                                  np.arange(len(audio)), audio).astype(np.float32)
            audio_secs += len(audio) / sample_rate

            hyp = transcribe_audio(audio, model, tokenizer, params, chunk_samples)
        except Exception as e:
            errors += 1
            hyp = ""
            if errors <= 3:
                print(f"    [error idx={idx}] {type(e).__name__}: {e}")

        ref_n = normalize(sample[ref_field])
        hyp_n = normalize(hyp)
        ref_w = ref_n.split()
        hyp_w = hyp_n.split()
        s = d = i = 0
        if ref_w:
            s, d, i = compute_sdi(ref_w, hyp_w)
            total_s += s; total_d += d; total_i += i
            total_w += len(ref_w)

        if has_demo:
            g = (sample.get("gender") or "").strip() or "unknown"
            a = (sample.get("age") or "").strip() or "unknown"
            ac = (sample.get("accents") or "").strip() or "unknown"
            cid = sample.get("client_id", "")
            for key, val in (("gender", g), ("age", a), ("accents", ac)):
                b = subgroups[key].setdefault(val, [0, 0, 0, 0, 0])
                b[0] += s; b[1] += d; b[2] += i; b[3] += len(ref_w); b[4] += 1
            ref_safe = ref_n.replace("\t", " ")
            hyp_safe = hyp_n.replace("\t", " ")
            tsv.write(f"{idx}\t{g}\t{a}\t{ac}\t{cid}\t{len(ref_w)}\t{s}\t{d}\t{i}\t{ref_safe}\t{hyp_safe}\n")
        else:
            tsv.write(f"{idx}\t{len(ref_w)}\t{s}\t{d}\t{i}\t{ref_n}\t{hyp_n}\n")
        if (idx + 1) % args.log_every == 0 or idx + 1 == n_total:
            edits = total_s + total_d + total_i
            wer = edits / max(total_w, 1) * 100
            wall = time.time() - t0
            rtf = audio_secs / wall if wall > 0 else 0
            print(f"    [{idx+1}/{n_total}] WER={wer:.2f}%  words={total_w}  "
                  f"RTF={rtf:.2f}x  errs={errors}", flush=True)

    tsv.close()

    edits = total_s + total_d + total_i
    wer = edits / max(total_w, 1) * 100
    sub = total_s / max(total_w, 1) * 100
    dele = total_d / max(total_w, 1) * 100
    ins = total_i / max(total_w, 1) * 100
    wall = time.time() - t0

    summary = {
        "model_dir": args.model_dir,
        "dataset": args.dataset,
        "lang": args.lang,
        "n_samples": n_total,
        "errors": errors,
        "total_words": total_w,
        "subs": total_s, "dels": total_d, "ins": total_i,
        "wer": wer, "sub_rate": sub, "del_rate": dele, "ins_rate": ins,
        "audio_seconds": audio_secs,
        "wall_seconds": wall,
        "rtf": (audio_secs / wall) if wall > 0 else 0.0,
    }

    if has_demo:
        breakdown = {}
        for key in ("gender", "age", "accents"):
            rows = {}
            for val, (s, d, i, w, n) in sorted(subgroups[key].items()):
                rows[val] = {
                    "n_utt": n, "words": w,
                    "subs": s, "dels": d, "ins": i,
                    "wer": (s + d + i) / max(w, 1) * 100,
                    "sub_rate": s / max(w, 1) * 100,
                    "del_rate": d / max(w, 1) * 100,
                    "ins_rate": i / max(w, 1) * 100,
                }
            breakdown[key] = rows
        summary["breakdown"] = breakdown

    summary_path.write_text(json.dumps(summary, indent=2))

    print()
    print("=" * 80)
    print(f"  {args.dataset.upper()} {args.lang}  ({n_total} samples)")
    print("=" * 80)
    print(f"  WER={wer:.2f}%  (S={sub:.2f}% D={dele:.2f}% I={ins:.2f}%)")
    print(f"  Counts: subs={total_s} dels={total_d} ins={total_i} / {total_w} ref words")
    print(f"  Audio={audio_secs:.1f}s  Wall={wall:.1f}s  RTF={summary['rtf']:.2f}x")

    if has_demo:
        for key in ("gender", "age", "accents"):
            print(f"\n  --- breakdown by {key} ---")
            print(f"  {'value':<22} {'n_utt':>6} {'words':>7} {'WER':>7} {'S%':>6} {'D%':>6} {'I%':>6}")
            for val, row in sorted(summary["breakdown"][key].items(),
                                   key=lambda kv: -kv[1]["n_utt"]):
                print(f"  {val:<22} {row['n_utt']:>6} {row['words']:>7} "
                      f"{row['wer']:>6.2f}% {row['sub_rate']:>5.2f}% "
                      f"{row['del_rate']:>5.2f}% {row['ins_rate']:>5.2f}%")
    print(f"  Output: {out_dir}")


if __name__ == "__main__":
    main()
