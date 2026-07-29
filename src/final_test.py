import json, re, sqlite3
from collections import defaultdict
from pathlib import Path
from mlx_lm import generate, load
import infer

ROOT = infer.ROOT
BASE = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SWEEP_DIR = ROOT / "adapters_sweep"
TEST = ROOT / "data" / "dataset" / "test.jsonl"
OUT = ROOT / "data" / "final_test_results.jsonl"

def _norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def main():
    rows = [json.loads(l) for l in TEST.read_text().splitlines()]
    conn = sqlite3.connect(infer.DB_PATH)
    gold = {}
    for r in rows:
        if r["sql"] not in gold:
            gold[r["sql"]], _ = infer.run_sql(conn, r["sql"])
    print(f"final test: {len(rows)} rows / {len({r['template_id'] for r in rows})} templates (iter 200)")
    print("loading selected checkpoint ...", flush=True)
    model, tok = load(BASE, adapter_path=str(SWEEP_DIR))
    per_t = defaultdict(lambda: {"n": 0, "exec": 0, "exact": 0, "tier": None})
    out_lines = []
    for i, r in enumerate(rows, 1):
        prompt = infer.build_prompt(tok, r["question"], oneshot=False)
        comp = generate(model, tok, prompt=prompt, max_tokens=320, verbose=False)
        sql, how = infer.extract_sql(comp)
        pred, err = (infer.run_sql(conn, sql) if sql else (None, "no sql"))
        match = infer.result_match(gold[r["sql"]], pred)
        s = per_t[r["template_id"]]
        s["n"] += 1; s["tier"] = r["tier"]
        s["exec"] += int(match)
        s["exact"] += int(_norm(sql) == _norm(r["sql"]))
        out_lines.append(json.dumps({"template_id": r["template_id"], "tier": r["tier"],
            "question": r["question"], "gold_sql": r["sql"], "pred_sql": sql, "match": match}))
        if i % 100 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    OUT.write_text("\n".join(out_lines) + "\n")
    print("\n--- per-template (test) ---")
    print(f"{'template':<34}{'tier':>5}{'n':>5}{'exec%':>8}{'exact%':>8}")
    for t in sorted(per_t, key=lambda x: (per_t[x]["tier"], x)):
        s = per_t[t]
        print(f"{t:<34}{s['tier']:>5}{s['n']:>5}{s['exec']/s['n']*100:>7.1f}%{s['exact']/s['n']*100:>7.1f}%")
    tier = defaultdict(lambda: {"n": 0, "exec": 0})
    for t, s in per_t.items():
        tier[s["tier"]]["n"] += s["n"]; tier[s["tier"]]["exec"] += s["exec"]
    print("\n--- per-tier (exploratory) ---")
    for k in sorted(tier):
        v = tier[k]
        print(f"  tier {k}: {v['exec']/v['n']*100:5.1f}%  (n={v['n']})")
    macro = sum(s["exec"]/s["n"] for s in per_t.values()) / len(per_t)
    row_exec = sum(s["exec"] for s in per_t.values()) / sum(s["n"] for s in per_t.values())
    row_exact = sum(s["exact"] for s in per_t.values()) / sum(s["n"] for s in per_t.values())
    print(f"\n===== FINAL TEST RESULT (iter 200) =====")
    print(f"  macro-average execution-match : {macro*100:.1f}%   <-- primary")
    print(f"  row-level execution-match     : {row_exec*100:.1f}%")
    print(f"  row-level exact-match         : {row_exact*100:.1f}%")
    conn.close()

if __name__ == "__main__":
    main()
