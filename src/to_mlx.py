"""Convert rich dataset records into MLX-LM chat format for LoRA training.

MLX-LM's trainer reads data/train.jsonl and data/valid.jsonl where each line is
{"messages": [{"role","content"}, ...]}. It ignores any other fields, so the
template_id / tier / tags the eval needs are kept only in the rich
data/dataset/ files -- this writes a training-only view to data/mlx/.

The schema goes in the system turn (once), not the user turn, so the prompt the
model trains on is byte-identical to the prompt it will be served at inference.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "dataset"
OUT = ROOT / "data" / "mlx"

meta = json.loads((SRC / "meta.json").read_text())
SYSTEM = meta["instruction"] + "\n\n" + meta["schema_prompt"]


def convert(name):
    rows = [json.loads(l) for l in (SRC / f"{name}.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        out.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["completion"]},
        ]})
    return rows, out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # MLX-LM expects file stems train / valid / test inside the data dir.
    for name in ["train", "valid", "test"]:
        rows, conv = convert(name)
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in conv:
                f.write(json.dumps(r) + "\n")
        print(f"{name:6} {len(conv):5} examples")

    sample = json.loads((OUT / "train.jsonl").read_text().splitlines()[0])
    print("\n--- system turn (first 300 chars) ---")
    print(sample["messages"][0]["content"][:300])
    print("\n--- user turn ---")
    print(sample["messages"][1]["content"])
    print("\n--- assistant turn ---")
    print(sample["messages"][2]["content"])

    # token-length sanity: longest example drives the training sequence length
    import statistics
    lens = []
    for name in ["train", "valid"]:
        for l in (OUT / f"{name}.jsonl").read_text().splitlines():
            m = json.loads(l)["messages"]
            chars = sum(len(t["content"]) for t in m)
            lens.append(chars // 4)  # rough token estimate
    print(f"\napprox tokens/example: median {int(statistics.median(lens))}  "
          f"p95 {int(sorted(lens)[int(len(lens)*0.95)])}  max {max(lens)}")


if __name__ == "__main__":
    main()
