"""Validate the Experiment-2 expansion templates: execute all, check for empties,
and confirm none replicates a test-template SQL shape."""

import json
import random
import re
import sqlite3
from pathlib import Path

import templates  # registers base templates
import templates_expansion  # registers x_ templates
from templates import TEMPLATES, DB_PATH, sample_values

ROOT = Path(__file__).resolve().parents[1]
meta = json.loads((ROOT / "data" / "dataset" / "meta.json").read_text())
test_ids = set(meta["test_templates"])


def shape(sql):
    s = re.sub(r"'[^']*'", "'?'", sql)   # blank string literals
    s = re.sub(r"\b\d+\b", "N", s)        # blank numbers
    return re.sub(r"\s+", " ", s.strip().lower())


def main():
    conn = sqlite3.connect(DB_PATH)
    vals = sample_values(conn)

    # collect test-template shapes to guard against
    test_shapes = set()
    for t in TEMPLATES:
        if t.tid in test_ids:
            for seed in range(10):
                _, sql = t.build(vals, random.Random(seed))
                test_shapes.add(shape(sql))

    new = [t for t in TEMPLATES if t.tid.startswith("x_")]
    print(f"new templates: {len(new)}")

    bad = empty = collide = 0
    new_shapes = set()
    for t in new:
        for seed in range(20):
            qs, sql = t.build(vals, random.Random(seed))
            try:
                if not conn.execute(sql).fetchall():
                    empty += 1
                    print("  EMPTY", t.tid)
            except Exception as e:
                bad += 1
                print("  FAIL", t.tid, str(e)[:60])
            sh = shape(sql)
            new_shapes.add(sh)
            if sh in test_shapes:
                collide += 1
                print("  COLLISION with test shape:", t.tid)

    print(f"\nexecution failures: {bad}")
    print(f"empty result sets:  {empty}")
    print(f"test-shape collisions: {collide}  (must be 0)")
    print(f"distinct SQL shapes from new templates: {len(new_shapes)}")
    conn.close()


if __name__ == "__main__":
    main()
