"""Assemble train/valid/test splits from the template bank.

Split design:
  test  - template-disjoint. Entire templates are held out, so the test set
          measures generalisation to unseen query shapes rather than to unseen
          phrasings of shapes already memorised. This is the headline metric.
  valid - drawn from TRAINING templates with different parameters/phrasings.
          Its role is early stopping, so it is deliberately in-distribution
          and is never reported as a generalisation result.
  train - everything else.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from templates import TEMPLATES, DB_PATH, generate
from schema_text import schema_prompt

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "dataset"
SEED = 1729
TEST_FRACTION = 0.20
VALID_FRACTION = 0.10
VARIATIONS_PER_TEMPLATE = 40

INSTRUCTION = (
    "You are a SQL analyst for a manufacturing operations database. "
    "Given the schema and a question, respond with a single JSON tool call "
    "that answers the question. Respond with JSON only."
)


def tool_call(sql):
    return json.dumps({"tool": "execute_sql", "arguments": {"query": sql}})


def pick_test_templates(rng):
    """Hold out whole templates, stratified by tier."""
    by_tier = defaultdict(list)
    for t in TEMPLATES:
        by_tier[t.tier].append(t.tid)
    held = []
    for tier in sorted(by_tier):
        ids = sorted(by_tier[tier])
        rng.shuffle(ids)
        k = max(1, round(len(ids) * TEST_FRACTION))
        held.extend(ids[:k])
    return set(held)


def main():
    rng = random.Random(SEED)
    schema = schema_prompt()
    test_tids = pick_test_templates(rng)

    seen = set()
    rows = []
    for var in range(VARIATIONS_PER_TEMPLATE):
        for rec in generate(n_per_template=1, seed=SEED + var):
            for q in rec["questions"]:
                key = (q, rec["sql"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "template_id": rec["template_id"],
                    "tier": rec["tier"],
                    "tags": rec["tags"],
                    "question": q,
                    "sql": rec["sql"],
                    "completion": tool_call(rec["sql"]),
                })

    test = [r for r in rows if r["template_id"] in test_tids]
    pool = [r for r in rows if r["template_id"] not in test_tids]
    rng.shuffle(pool)
    n_valid = int(len(pool) * VALID_FRACTION)
    valid, train = pool[:n_valid], pool[n_valid:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"schema_prompt": schema, "instruction": INSTRUCTION,
            "test_templates": sorted(test_tids), "seed": SEED}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        with (OUT_DIR / f"{name}.jsonl").open("w") as f:
            for r in split:
                f.write(json.dumps(r) + "\n")

    print(f"held-out test templates ({len(test_tids)}): {sorted(test_tids)}")
    print()
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        tiers = Counter(r["tier"] for r in split)
        tmpl = len({r["template_id"] for r in split})
        print(f"{name:6} {len(split):5} pairs   {tmpl:3} templates   "
              f"tiers {dict(sorted(tiers.items()))}")
    print(f"\ntotal {len(rows)} unique pairs")

    overlap = {r["template_id"] for r in train} & {r["template_id"] for r in test}
    print(f"train/test template overlap: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
