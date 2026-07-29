"""Exp-2 step 1: generate raw rows for the 12 x_ expansion templates ONLY
and append them to train.jsonl. Paraphrase + re-split run afterward.
Frozen 8 test templates are never touched."""
import json
from collections import Counter
from pathlib import Path

import templates            # noqa: F401  registers base templates
import templates_expansion  # noqa: F401  registers x_ templates
from templates import generate
from build_dataset import tool_call, SEED, VARIATIONS_PER_TEMPLATE

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset"

existing_tids = {json.loads(l)["template_id"]
                 for name in ("train", "valid", "test")
                 for l in (DATA / f"{name}.jsonl").read_text().splitlines()}

seen, new_rows = set(), []
for var in range(VARIATIONS_PER_TEMPLATE):
    for rec in generate(n_per_template=1, seed=SEED + var):
        if not rec["template_id"].startswith("x_"):
            continue
        for q in rec["questions"]:
            key = (q, rec["sql"])
            if key in seen:
                continue
            seen.add(key)
            new_rows.append({"template_id": rec["template_id"], "tier": rec["tier"],
                             "tags": rec["tags"], "question": q, "sql": rec["sql"],
                             "completion": tool_call(rec["sql"])})

new_tids = {r["template_id"] for r in new_rows}
assert new_tids & existing_tids == set(), f"COLLISION with existing split: {new_tids & existing_tids}"
assert len(new_tids) == 12, f"expected 12 new templates, got {len(new_tids)}"

with (DATA / "train.jsonl").open("a") as f:
    for r in new_rows:
        f.write(json.dumps(r) + "\n")

print(f"appended {len(new_rows)} raw rows across {len(new_tids)} templates")
print("per template:", dict(sorted(Counter(r['template_id'] for r in new_rows).items())))
