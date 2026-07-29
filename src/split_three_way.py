"""Regenerate a template-stratified three-way split per EXPERIMENT.md.

TEST templates are read from the existing meta.json (untouched, 8 templates).
VALIDATION = one template per tier, the median-row-count template among that
tier's non-test templates (ties -> alphabetically first id). TRAIN = the rest.

Every split stays template-disjoint. Row assignment follows each row's template.
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
MLX = ROOT / "data" / "mlx"
meta = json.loads((DATA / "meta.json").read_text())
TEST_TEMPLATES = set(meta["test_templates"])

# all rows across the current splits, keyed by template
all_rows = []
for name in ["train", "valid", "test"]:
    for l in (DATA / f"{name}.jsonl").read_text().splitlines():
        all_rows.append(json.loads(l))

rows_by_template = defaultdict(list)
for r in all_rows:
    rows_by_template[r["template_id"]].append(r)

tier_of = {tid: rs[0]["tier"] for tid, rs in rows_by_template.items()}

# tier -> non-test templates, to pick validation by median row count
by_tier = defaultdict(list)
for tid in rows_by_template:
    if tid not in TEST_TEMPLATES:
        by_tier[tier_of[tid]].append(tid)

val_templates = set()
print("validation template selection (median row count per tier):")
for tier in sorted(by_tier):
    cands = by_tier[tier]
    counts = sorted(((len(rows_by_template[t]), t) for t in cands))
    # median by position; ties in count broken alphabetically via the sort key
    mid = (len(counts) - 1) // 2
    chosen = counts[mid][1]
    val_templates.add(chosen)
    print(f"  tier {tier}: candidates {len(cands)}  "
          f"row counts {[c for c, _ in counts]}  -> {chosen} "
          f"({len(rows_by_template[chosen])} rows)")

train_templates = set(rows_by_template) - TEST_TEMPLATES - val_templates

def rows_for(tids):
    out = []
    for t in tids:
        out.extend(rows_by_template[t])
    return out

splits = {
    "train": rows_for(train_templates),
    "valid": rows_for(val_templates),
    "test": rows_for(TEST_TEMPLATES),
}

# rewrite rich splits
for name, rows in splits.items():
    with (DATA / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

# rewrite MLX chat-format splits (system carries schema+instruction)
SYSTEM = meta["instruction"] + "\n\n" + meta["schema_prompt"]
MLX.mkdir(parents=True, exist_ok=True)
for name, rows in splits.items():
    with (MLX / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": r["question"]},
                {"role": "assistant", "content": r["completion"]},
            ]}) + "\n")

# record the split composition into meta for reproducibility
meta["val_templates"] = sorted(val_templates)
meta["train_templates"] = sorted(train_templates)
(DATA / "meta.json").write_text(json.dumps(meta, indent=2))

print()
def summarize(name, rows, tids):
    tiers = defaultdict(int)
    for r in rows:
        tiers[r["tier"]] += 1
    print(f"{name:6} {len(tids):3} templates  {len(rows):5} rows  "
          f"tiers {dict(sorted(tiers.items()))}")

summarize("train", splits["train"], train_templates)
summarize("valid", splits["valid"], val_templates)
summarize("test", splits["test"], TEST_TEMPLATES)

# integrity: no template in two splits
overlap = (train_templates & val_templates) | (train_templates & TEST_TEMPLATES) | (val_templates & TEST_TEMPLATES)
print(f"\ntemplate overlap across splits: {len(overlap)} (must be 0)")

# training tier coverage warning (per EXPERIMENT.md known limitation)
train_tier_counts = defaultdict(int)
for t in train_templates:
    train_tier_counts[tier_of[t]] += 1
thin = {tier: n for tier, n in train_tier_counts.items() if n <= 5}
if thin:
    print(f"thin training tiers (<=5 templates, expected per pre-registration): {dict(sorted(thin.items()))}")
