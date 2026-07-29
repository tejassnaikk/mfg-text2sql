# mfg-text2sql

Fine-tuning a small language model (Qwen2.5-3B) to answer natural-language questions about a manufacturing-operations database, emitted as a structured `execute_sql(...)` function call. Built and evaluated locally on an M4 Mac with LoRA via MLX.

**What this project actually is:** a study of whether a 3B model can be fine-tuned to *generalize* to query shapes it has never seen — and a rigorous, pre-registered evaluation that catches it failing to. The accuracy numbers are modest by design and by outcome. The value is in the experimental method and in what it lets me say precisely about *why* the model fails.

---

## Headline result

The primary metric is **macro-average execution-match** on a **template-disjoint** test set: entire query *shapes* are held out of training, so the score measures generalization to unseen shapes rather than memorized phrasing. Per-template rates are averaged with equal weight (not row-level, which is dominated by whichever templates paraphrasing happened to multiply).

| Model | Macro-avg exec-match (primary) | Row-level exec-match |
|---|---:|---:|
| Base Qwen2.5-3B, one-shot (no fine-tune) | — † | 22.2% |
| Fine-tune v1 (baseline template coverage) | 6.8% | 13.9% |
| Fine-tune v2 (coverage-expanded training set) | **17.1%** | 29.7% |

† The base model was measured on the row-level metric only, in the initial eval. Worth stating plainly: at row-level, the un-fine-tuned base model (22.2%) *beat* the first fine-tune (13.9%). The fine-tune reliably learned the output *format* but initially degraded correctness — an honest, non-flattering data point that motivated the second experiment.

Both fine-tuned numbers come from **one-time** evaluations of a checkpoint selected by a rule frozen and committed **before** training (git SHAs below). The test templates and their gold SQL were never inspected during development.

---

## The finding that matters more than the number

Fine-tune v2 raised the macro score from 6.8% to 17.1% — a 2.5× lift. But the per-tier breakdown shows the improvement did **not** come from where the intervention was aimed:

| Tier | Query complexity | v1 | v2 |
|---|---|---:|---:|
| 1 | single-table (lookups, ordering) | 0.0% | 0.0% |
| 2 | two-table joins | 29.9% | 50.0% |
| 3 | multi-join aggregates | 5.7% | 35.4% |
| 4 | date arithmetic, subqueries, superlatives | 0.0% | 0.0% |

Experiment 2 specifically added training coverage for the query families that failed in Experiment 1 (ascending-order sorts, product lookups, superlatives) — the families concentrated in tiers 1 and 4. **Those tiers stayed at exactly 0%.** Every point of the aggregate gain came from tiers 2 and 3.

The mechanism is visible when validation and test are compared. On the *validation* set, one of the newly added ascending-order templates scored 28% — the model clearly *can* learn that family when the exact shape appears in training. But on the *test* set, the unseen tier-1 shape that also needs ascending-order logic remained at 0%. So:

> The model learns the specific query shapes it is trained on. It does not abstract the underlying SQL conventions and transfer them to novel shapes in the same family.

This extends the Experiment 1 result rather than overturning it. More training diversity lifts test templates that are structurally close to well-covered training shapes; it confers no compositional generalization to genuinely novel ones. The failure is now localized: it is **not** insufficient coverage (v2 tested that) and **not** insufficient optimization (v1 tested that — see below). What remains is compositional generalization at this model size and adapter budget.

---

## The methodology (the real contribution)

**Pre-registration.** The evaluation protocol — the checkpoint-selection rule and every metric — was written into an `EXPERIMENT.md` and committed to git *before* any training run. Results therefore cannot be fit to the rule after the fact. Experiment 2's plan and a later scope amendment were likewise committed before their training run.

**Template-disjoint three-way split.** The 47 query templates are partitioned so that whole *shapes* are held out: training, a validation set for checkpoint selection, and a frozen 8-template test set. Because no test shape appears in training in any phrasing, the test measures generalization, not recall. Template overlap across splits is asserted to be zero.

**Execution-match as the primary metric.** SQL is scored by executing it and comparing result sets, not by string match. This correctly credits a semantically-equivalent query that differs from the gold SQL. (Concretely: one tier-3 test template scored 58.4% execution-match at 0% exact-match — the model wrote different-but-correct SQL. A string metric would have called every one of those wrong.) In Experiment 1, exact-match and execution-match agreed closely (≈12% vs ≈14%), which rules out "the metric is too strict" — the model was writing genuinely wrong SQL, not correct SQL phrased unusually.

**In-distribution loss does not predict generalization.** A checkpoint sweep (iters 50→500, saved every 50) showed held-out accuracy flat and noisy across the entire run — training 10× longer did not help. In-distribution validation loss converged smoothly to ~0.08 and stayed there, while held-out execution-match wandered with no trend. The correlation between the two was approximately zero (and slightly negative in v2). **A converged-looking loss curve carried no information about whether the model could generalize.** This is the single most useful thing the project demonstrates about evaluating fine-tunes.

---

## Data pipeline

The dataset is synthetic and built to be *answerable* and *diagnosable*, not to look impressive.

- **Schema:** 15 foreign-key-constrained SQLite tables spanning plants, production lines, machines, products, work orders, production runs, quality inspections, defects, maintenance, downtime, and sensor readings. Analyst-style questions span 2–5 joins.
- **Deterministic seeder with injected signal.** Scrap rates vary by product category and shift, and some machines are seeded as degraded, so aggregate queries have something real to find. *These correlations are invented for the exercise and do not reflect real manufacturing patterns.*
- **Inverted generation.** Rather than have an LLM write SQL and then execution-validate it (which catches crashes but not silently-wrong-but-runnable queries), the SQL is generated deterministically from parameterized templates, and a local LLM (Mistral-7B, a deliberately different model family from the fine-tune target) only **paraphrases the English question**. The gold SQL is correct by construction.
- **Two-way paraphrase guard.** Each paraphrase is validated to preserve required entities (with time-word synonyms) and to reject any filter value not present in the original question, with GROUP-BY awareness. The guard caught and discarded 555 paraphrases that dropped or invented a filter — mislabeled training data that never reached the model.

The paraphrased corpus is 2,686 question/SQL pairs across 47 templates.

---

## Stack

- **Base model:** `mlx-community/Qwen2.5-3B-Instruct-4bit`
- **Fine-tuning:** LoRA via `mlx-lm`, rank 8 / scale 16, last 8 layers, lr 1e-4, batch 4, sequence length 1024, on Apple M4 (Metal)
- **Paraphraser:** `mlx-community/Mistral-7B-Instruct-v0.3-4bit`
- **Database / eval:** SQLite, execution-based result-set comparison

---

## Limitations (stated plainly)

- **Small test n.** Eight held-out templates. The macro-average is defensible; the per-tier numbers are exploratory, not statistically strong.
- **Single seed.** No variance estimate across training seeds.
- **Synthetic data.** The schema and the injected correlations are constructed. Findings are about model behavior on this controlled task, not about production text-to-SQL.
- **No frontier baseline yet.** A same-prompt, same-harness Gemini run on the frozen test set is scripted (`src/gemini_baseline.py`) but not yet completed. It would resolve the remaining ambiguity: if a frontier model also fails tiers 1 and 4, those gold conventions are underdetermined by the question (a dataset finding); if it clears them, the failure is cleanly attributable to model scale.
- **No live demo is claimed.** This repository is a research and evaluation artifact.

---

## Reproduce

```zsh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/seed.py                 # build + seed the SQLite database
python src/build_dataset.py        # generate templated question/SQL pairs
python src/paraphrase.py           # paraphrase questions (guarded)
python src/split_three_way.py      # template-disjoint train/val/test split
python src/to_mlx.py               # convert to MLX chat format

mlx_lm.lora --config src/lora_config_sweep.yaml   # sweep-train (checkpoints every 50 iters)
python src/sweep_eval.py           # score each checkpoint on validation; select by frozen rule
python src/final_test.py           # one-time evaluation on the frozen 8 test templates
```

Long MLX runs should be prefixed with `caffeinate -i` and run with the lid open; validation passes stall for hours if the machine sleeps mid-run.

---

## Repository layout

```
src/
  seed.py                  15-table schema + deterministic seeder with injected signal
  templates.py             35 base NL→SQL templates across 4 difficulty tiers
  templates_expansion.py   12 coverage templates added in Experiment 2 (authored blind to test)
  paraphrase.py            LLM paraphraser + two-way entity/filter validator
  build_dataset.py         assembles question/SQL pairs
  split_three_way.py       template-disjoint train/val/test split
  to_mlx.py                convert to MLX chat format
  infer.py                 shared prompt/extract/execute/compare harness
  sweep_eval.py            per-checkpoint validation scoring + frozen selection rule
  final_test.py            one-time frozen test evaluation
  gemini_baseline.py       frontier baseline (same prompt + harness)
EXPERIMENT.md              pre-registered protocol (committed before training)
```

The pre-registered protocol — the selection rule and every metric, fixed before any training run — is in `EXPERIMENT.md`. The development repository was reset to a clean single-commit history before publishing (regenerable adapter checkpoints and generated datasets are not tracked; see `.gitignore`), so the original per-experiment commit timestamps live in the local development history rather than here.
