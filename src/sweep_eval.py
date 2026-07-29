"""Evaluate every checkpoint on the VALIDATION templates per EXPERIMENT.md.

Fixed 25 rows PER validation template (sampled once, same rows for every
checkpoint), scored with the real execute-and-compare metric. Reports, per
checkpoint: train loss, in-distribution val loss (parsed from the train log),
and validation MACRO-average execution-match (per-template rates averaged with
equal weight). This is the pre-registered model-selection signal.
"""

import json
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from mlx_lm import generate, load

import infer

ROOT = infer.ROOT
BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SWEEP_DIR = ROOT / "adapters_sweep"
VAL = ROOT / "data" / "dataset" / "valid.jsonl"
TRAIN_LOG = SWEEP_DIR / "train.log"
OUT = ROOT / "data" / "sweep_results.json"
ROWS_PER_TEMPLATE = 25
SAMPLE_SEED = 20260726


def fixed_sample():
    import random
    rows = [json.loads(l) for l in VAL.read_text().splitlines()]
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["template_id"]].append(r)
    rng = random.Random(SAMPLE_SEED)
    sample = []
    for tid in sorted(by_t):
        pool = by_t[tid]
        rng.shuffle(pool)
        sample.extend(pool[:ROWS_PER_TEMPLATE])
    return sample


def parse_losses(log_text):
    """iter -> (train_loss, val_loss) from the MLX training log."""
    tl, vl = {}, {}
    for m in re.finditer(r"Iter (\d+): Train loss ([\d.]+)", log_text):
        tl[int(m.group(1))] = float(m.group(2))
    for m in re.finditer(r"Iter (\d+): Val loss ([\d.]+)", log_text):
        vl[int(m.group(1))] = float(m.group(2))
    return tl, vl


def eval_checkpoint(ckpt, sample, conn, gold):
    import shutil
    # make this checkpoint the active adapter, then load
    shutil.copy(ckpt, SWEEP_DIR / "adapters.safetensors")
    model, tok = load(BASE, adapter_path=str(SWEEP_DIR))
    per_t = defaultdict(lambda: {"n": 0, "exec": 0, "exact": 0})
    for r in sample:
        prompt = infer.build_prompt(tok, r["question"], oneshot=False)
        comp = generate(model, tok, prompt=prompt, max_tokens=320, verbose=False)
        sql, how = infer.extract_sql(comp)
        pred, err = (infer.run_sql(conn, sql) if sql else (None, "no sql"))
        s = per_t[r["template_id"]]
        s["n"] += 1
        s["exec"] += int(infer.result_match(gold[r["sql"]], pred))
        s["exact"] += int(_norm(sql) == _norm(r["sql"]))
    del model, tok
    rates = {t: s["exec"] / s["n"] for t, s in per_t.items()}
    exact = {t: s["exact"] / s["n"] for t, s in per_t.items()}
    macro = sum(rates.values()) / len(rates)
    macro_exact = sum(exact.values()) / len(exact)
    return macro, macro_exact, rates


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main():
    sample = fixed_sample()
    conn = sqlite3.connect(infer.DB_PATH)
    gold = {}
    for r in sample:
        if r["sql"] not in gold:
            gold[r["sql"]], _ = infer.run_sql(conn, r["sql"])

    n_templates = len({r["template_id"] for r in sample})
    print(f"validation sample: {len(sample)} rows across {n_templates} templates "
          f"({ROWS_PER_TEMPLATE}/template)\n")

    tl, vl = parse_losses(TRAIN_LOG.read_text())
    ckpts = sorted(SWEEP_DIR.glob("0*_adapters.safetensors"))

    results = []
    print(f"{'iter':>5}{'train':>8}{'val_loss':>10}{'val_macro_exec':>16}")
    for ckpt in ckpts:
        it = int(ckpt.name.split("_")[0])
        macro, macro_exact, rates = eval_checkpoint(ckpt, sample, conn, gold)
        results.append({"iter": it, "train_loss": tl.get(it),
                        "val_loss": vl.get(it), "val_macro_exec": macro,
                        "val_macro_exact": macro_exact, "per_template": rates})
        print(f"{it:>5}{(tl.get(it) or 0):>8.3f}{(vl.get(it) or 0):>10.3f}"
              f"{macro*100:>15.1f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2))

    # pre-registered selection: max macro-exec, tie -> macro-exact, tie -> earlier
    best = max(results, key=lambda r: (round(r["val_macro_exec"], 4),
                                       round(r["val_macro_exact"], 4),
                                       -r["iter"]))
    print(f"\nSELECTED (pre-registered rule): iter {best['iter']}  "
          f"val_macro_exec {best['val_macro_exec']*100:.1f}%")
    print("per-template exec at selected checkpoint:")
    for t, rate in sorted(best["per_template"].items()):
        print(f"  {t:<34} {rate*100:5.1f}%")

    # is in-distribution val loss correlated with held-out generalization?
    import statistics
    vls = [r["val_loss"] for r in results if r["val_loss"]]
    ms = [r["val_macro_exec"] for r in results if r["val_loss"]]
    if len(vls) > 2 and statistics.pstdev(vls) > 0 and statistics.pstdev(ms) > 0:
        mv, mm = statistics.mean(vls), statistics.mean(ms)
        cov = sum((a - mv) * (b - mm) for a, b in zip(vls, ms)) / len(vls)
        corr = cov / (statistics.pstdev(vls) * statistics.pstdev(ms))
        print(f"\ncorr(in-distribution val_loss, held-out macro_exec): {corr:+.2f}")
        print("  (near 0 or positive = val loss FAILED to predict generalization)")
    conn.close()


if __name__ == "__main__":
    main()
