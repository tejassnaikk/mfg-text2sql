"""Shared inference + execution path. Used by the smoke test and the real eval.

Keeping generation, JSON extraction, SQL extraction, and execution in one place
guarantees the base model, the fine-tuned model, and the frontier baseline are
all measured through identical plumbing -- differences in scores then reflect
the models, not three subtly different harnesses.
"""

import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "db" / "mfg.db"
META = json.loads((ROOT / "data" / "dataset" / "meta.json").read_text())
ONESHOT = (
    '\n\nExample of the required output format (for a different question):\n'
    'Question: How many machines are there?\n'
    '{"tool": "execute_sql", "arguments": {"query": "SELECT COUNT(*) FROM machines;"}}'
)
SYSTEM = META["instruction"] + "\n\n" + META["schema_prompt"]
SYSTEM_ONESHOT = SYSTEM + ONESHOT


def build_prompt(tokenizer, question, oneshot=False):
    msgs = [{"role": "system", "content": SYSTEM_ONESHOT if oneshot else SYSTEM},
            {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False)


def extract_sql(text):
    """Pull the SQL string out of a model completion.

    Preferred path: parse the JSON tool call and read arguments.query. Fallback:
    if JSON is malformed, grab the first SELECT... so a formatting slip doesn't
    get scored as a SQL failure it isn't.
    """
    text = text.strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            q = obj.get("arguments", {}).get("query")
            if isinstance(q, str) and q.strip():
                return q.strip(), "json"
        except Exception:
            pass
    m = re.search(r"(SELECT\b.*?;)", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip(), "fallback"
    # No SQL at all. Distinguish a fabricated answer object from other misses,
    # since "the model invented rows instead of querying" is a finding worth counting.
    if re.search(r'"(rows|result|results|data|answer)"\s*:', text):
        return None, "hallucinated"
    return None, "none"


def run_sql(conn, sql):
    """Return (rows, error). rows is a list; error is None on success."""
    try:
        cur = conn.execute(sql)
        return cur.fetchall(), None
    except Exception as e:
        return None, str(e)


def result_match(a, b):
    """Order-insensitive multiset comparison of two result sets."""
    if a is None or b is None:
        return False
    return sorted(map(repr, a)) == sorted(map(repr, b))


if __name__ == "__main__":
    import random
    from mlx_lm import generate, load

    rows = [json.loads(l) for l in
            (ROOT / "data" / "dataset" / "test.jsonl").read_text().splitlines()]
    # one example per held-out template, up to 5, for a spread of difficulty
    by_t = {}
    for r in rows:
        by_t.setdefault(r["template_id"], r)
    sample = list(by_t.values())[:5]

    print("loading base Qwen2.5-3B ...")
    model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
    conn = sqlite3.connect(DB_PATH)

    ok = 0
    for r in sample:
        prompt = build_prompt(tokenizer, r["question"], oneshot=True)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=False)
        sql, how = extract_sql(out)
        gold_rows, _ = run_sql(conn, r["sql"])
        pred_rows, err = (run_sql(conn, sql) if sql else (None, "no sql"))
        match = result_match(gold_rows, pred_rows)
        ok += match
        print(f"\n[{r['template_id']}  tier {r['tier']}]")
        print("  Q:", r["question"])
        print(f"  extract={how}  exec={'ERR' if err else 'ok'}  match={match}")
        if sql:
            print("  SQL:", sql[:140].replace("\n", " "))
        if err:
            print("  err:", err[:100])

    print(f"\nbase model result-set matches: {ok}/{len(sample)}")
    conn.close()
