"""Frontier baseline: Gemini 2.5 Flash on the frozen 8 test templates.
Identical prompt (imported SYSTEM_ONESHOT) + identical harness to the Qwen
runs; only the model call differs. 25 rows/template -> macro-average
comparable to 6.8% (base-FT) and 17.1% (Exp-2)."""
import json, os, random, time, sqlite3
from collections import defaultdict
from google import genai
from google.genai import types
import infer

MODEL = "gemini-2.5-flash-lite"
N_PER = 5
ROOT = infer.ROOT
TEST = ROOT / "data" / "dataset" / "test.jsonl"
OUT = ROOT / "data" / "gemini_baseline_results.jsonl"
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def ask(q):
    for a in range(6):
        try:
            r = client.models.generate_content(
                model=MODEL, contents=q,
                config=types.GenerateContentConfig(
                    system_instruction=infer.SYSTEM_ONESHOT,
                    temperature=0, max_output_tokens=512))
            return r.text or ""
        except Exception as e:
            w = 5 * (a + 1)
            print(f"   retry {a+1} in {w}s ({str(e)[:60]})", flush=True)
            time.sleep(w)
    return ""

def main():
    rows = [json.loads(l) for l in TEST.read_text().splitlines()]
    by_t = defaultdict(list)
    for r in rows: by_t[r["template_id"]].append(r)
    rng = random.Random(42)
    sample = []
    for tid, rs in by_t.items():
        rng.shuffle(rs); sample.extend(rs[:N_PER])
    conn = sqlite3.connect(infer.DB_PATH)
    gold = {}
    for r in sample:
        if r["sql"] not in gold:
            gold[r["sql"]], _ = infer.run_sql(conn, r["sql"])
    per_t = defaultdict(lambda: {"n": 0, "exec": 0, "tier": None})
    out = []
    print(f"gemini baseline: {len(sample)} rows / {len(by_t)} templates", flush=True)
    for i, r in enumerate(sample, 1):
        sql, how = infer.extract_sql(ask(r["question"]))
        pred, err = (infer.run_sql(conn, sql) if sql else (None, "no sql"))
        match = infer.result_match(gold[r["sql"]], pred)
        s = per_t[r["template_id"]]
        s["n"] += 1; s["tier"] = r["tier"]; s["exec"] += int(match)
        out.append(json.dumps({"template_id": r["template_id"], "tier": r["tier"],
            "question": r["question"], "gold_sql": r["sql"], "pred_sql": sql, "match": match}))
        if i % 20 == 0: print(f"  {i}/{len(sample)}", flush=True)
        time.sleep(6)
    OUT.write_text("\n".join(out) + "\n")
    print("\n--- per-template (gemini) ---")
    print(f"{'template':<40}{'tier':>5}{'n':>5}{'exec%':>8}")
    for t in sorted(per_t, key=lambda x: (per_t[x]['tier'], x)):
        s = per_t[t]; print(f"{t:<40}{s['tier']:>5}{s['n']:>5}{s['exec']/s['n']*100:>7.1f}%")
    tier = defaultdict(lambda: {"n": 0, "exec": 0})
    for t, s in per_t.items():
        tier[s["tier"]]["n"] += s["n"]; tier[s["tier"]]["exec"] += s["exec"]
    print("\n--- per-tier ---")
    for k in sorted(tier):
        v = tier[k]; print(f"  tier {k}: {v['exec']/v['n']*100:5.1f}%  (n={v['n']})")
    macro = sum(s["exec"]/s["n"] for s in per_t.values()) / len(per_t)
    row = sum(s["exec"] for s in per_t.values()) / sum(s["n"] for s in per_t.values())
    print(f"\n===== GEMINI BASELINE ({MODEL}) =====")
    print(f"  macro-average execution-match : {macro*100:.1f}%   <-- vs 6.8% base-FT / 17.1% Exp-2")
    print(f"  row-level execution-match     : {row*100:.1f}%")
    conn.close()

if __name__ == "__main__":
    main()
