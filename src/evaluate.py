"""Full test-set eval: base Qwen (one-shot) vs fine-tuned Qwen (LoRA).

Primary metric: result-set equivalence (rows match gold, order-insensitive).
Reported alongside: execution-success rate, format validity, and -- for the
base model -- the rate at which it fabricates an answer object instead of
emitting SQL. Models are loaded one at a time; the M4 can't hold two at once.
"""

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

from mlx_lm import generate, load

import infer

ROOT = infer.ROOT
BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER = str(ROOT / "adapters" / "0000050_adapters.safetensors")
TEST = ROOT / "data" / "dataset" / "test.jsonl"
RESULTS = ROOT / "data" / "eval_results.jsonl"


def load_test():
    return [json.loads(l) for l in TEST.read_text().splitlines()]


def gold_cache(conn, rows):
    cache = {}
    for r in rows:
        if r["sql"] not in cache:
            res, err = infer.run_sql(conn, r["sql"])
            cache[r["sql"]] = res
    return cache


def eval_model(tag, model, tokenizer, rows, gold, conn, oneshot):
    t0 = time.time()
    per_tier = defaultdict(lambda: {"n": 0, "match": 0, "exec_ok": 0,
                                    "valid_json": 0, "hallucinated": 0})
    out_lines = []
    for i, r in enumerate(rows, 1):
        prompt = infer.build_prompt(tokenizer, r["question"], oneshot=oneshot)
        comp = generate(model, tokenizer, prompt=prompt, max_tokens=320, verbose=False)
        sql, how = infer.extract_sql(comp)
        pred, err = (infer.run_sql(conn, sql) if sql else (None, "no sql"))
        match = infer.result_match(gold[r["sql"]], pred)

        t = r["tier"]
        s = per_tier[t]
        s["n"] += 1
        s["match"] += int(match)
        s["exec_ok"] += int(sql is not None and err is None)
        s["valid_json"] += int(how == "json")
        s["hallucinated"] += int(how == "hallucinated")

        out_lines.append(json.dumps({
            "model": tag, "template_id": r["template_id"], "tier": t,
            "question": r["question"], "gold_sql": r["sql"], "pred_sql": sql,
            "extract": how, "exec_error": err, "match": match}))

        if i % 50 == 0:
            el = time.time() - t0
            print(f"  [{tag}] {i}/{len(rows)}  {el:.0f}s  "
                  f"eta {el/i*(len(rows)-i):.0f}s", flush=True)

    with RESULTS.open("a") as f:
        f.write("\n".join(out_lines) + "\n")
    return per_tier


def report(tag, per_tier):
    tiers = sorted(per_tier)
    tot = {k: sum(per_tier[t][k] for t in tiers)
           for k in ["n", "match", "exec_ok", "valid_json", "hallucinated"]}
    print(f"\n===== {tag} =====")
    print(f"{'tier':<6}{'n':>5}{'match':>8}{'exec':>8}{'json':>8}")
    for t in tiers:
        s = per_tier[t]
        print(f"{t:<6}{s['n']:>5}{s['match']/s['n']*100:>7.1f}%"
              f"{s['exec_ok']/s['n']*100:>7.1f}%{s['valid_json']/s['n']*100:>7.1f}%")
    print(f"{'ALL':<6}{tot['n']:>5}{tot['match']/tot['n']*100:>7.1f}%"
          f"{tot['exec_ok']/tot['n']*100:>7.1f}%{tot['valid_json']/tot['n']*100:>7.1f}%")
    if tot["hallucinated"]:
        print(f"  hallucinated answer objects: {tot['hallucinated']} "
              f"({tot['hallucinated']/tot['n']*100:.1f}%)")
    return tot["match"] / tot["n"] * 100


def main():
    rows = load_test()
    conn = sqlite3.connect(infer.DB_PATH)
    gold = gold_cache(conn, rows)
    RESULTS.unlink(missing_ok=True)
    print(f"test rows: {len(rows)}   unique gold queries: {len(gold)}")

    print("\nloading BASE ...", flush=True)
    m, t = load(BASE)
    base_tier = eval_model("base", m, t, rows, gold, conn, oneshot=True)
    del m, t

    print("\nloading FINE-TUNED (adapter 0000050) ...", flush=True)
    m, t = load(BASE, adapter_path=str(ROOT / "adapters"))
    # point at the specific checkpoint
    import shutil
    ft_tier = eval_model("finetuned", m, t, rows, gold, conn, oneshot=False)
    del m, t

    base_acc = report("BASE (one-shot)", base_tier)
    ft_acc = report("FINE-TUNED", ft_tier)
    print(f"\n>>> result-set accuracy:  base {base_acc:.1f}%  ->  "
          f"fine-tuned {ft_acc:.1f}%   (+{ft_acc-base_acc:.1f} pts)")
    conn.close()


if __name__ == "__main__":
    main()
