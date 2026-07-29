"""Derive the convention spec for Arm 2 from TRAINING gold SQL only.

Reads data/dataset/train.jsonl (never test.jsonl) and surfaces the conventions
the model failed to infer in Experiment 1: which columns get projected for
lookup-style queries, and how ordering words map to ASC/DESC. Output is appended
to the system prompt for the Arm 2 / Arm 1+2 runs. Test set is untouched.
"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "dataset" / "train.jsonl"


def main():
    rows = [json.loads(l) for l in TRAIN.read_text().splitlines()]

    # 1) For product lookups, what does gold project?
    proj = Counter()
    for r in rows:
        s = r["sql"]
        m = re.match(r"SELECT\s+(.*?)\s+FROM\s+products", s, re.I | re.S)
        if m and "JOIN" not in s.upper()[:s.upper().find("WHERE") if "WHERE" in s.upper() else len(s)]:
            cols = tuple(c.strip() for c in m.group(1).split(","))
            proj[cols] += 1

    # 2) Do gold queries ever use bare product_id / table aliases in simple lookups?
    alias_use = sum(1 for r in rows if re.search(r"\bFROM\s+products\s+p\b", r["sql"], re.I))
    bare = sum(1 for r in rows if re.search(r"\bFROM\s+products\b(?!\s+p\b)", r["sql"], re.I))

    # 3) ASC vs DESC association with wording (from the questions)
    asc_words = Counter(); desc_words = Counter()
    for r in rows:
        q = r["question"].lower()
        if "ORDER BY" in r["sql"].upper():
            direction = "ASC" if re.search(r"ORDER BY[^;]*\bASC\b", r["sql"], re.I) else "DESC"
            for w in ["shortest", "lowest", "fewest", "least", "smallest",
                      "longest", "highest", "most", "greatest", "largest", "top"]:
                if w in q:
                    (asc_words if direction == "ASC" else desc_words)[w] += 1

    print("=== projection for simple products lookups (gold) ===")
    for cols, n in proj.most_common():
        print(f"  {n:4}  SELECT {', '.join(cols)}")
    print(f"\nsimple products FROM: aliased 'p' {alias_use}, bare {bare}")
    print("\n=== ordering word -> direction (gold) ===")
    print("  ASC :", dict(asc_words))
    print("  DESC:", dict(desc_words))


if __name__ == "__main__":
    main()
