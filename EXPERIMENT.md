# Experiment: template-disjoint generalization of a LoRA text-to-SQL fine-tune

Pre-registered before training. Committed prior to any checkpoint sweep so the
selection rule provably precedes the results. Do not edit after training begins;
record deviations in a dated appendix instead.

## Question
Can a LoRA fine-tune of Qwen2.5-3B generalize to *unseen SQL query templates*
(not just unseen phrasings of seen templates)? A prior run trained to iter 50
scored 9.1% execution-match on held-out templates while reaching val loss 0.079
on in-distribution validation — i.e. in-distribution loss looked converged while
held-out generalization was poor. This experiment tests whether more training
optimization changes that, and whether in-distribution val loss tracks
template-disjoint accuracy at all.

## Split (template-stratified, three-way, template-disjoint)
- TEST: the existing 8 held-out templates (2 per tier). Untouched — never
  trained or selected on. Used exactly once, at the very end.
- VALIDATION: 4 templates, one per tier, carved out of the former 27 training
  templates. Selection rule: within each tier, the template whose row count is
  the MEDIAN among that tier's eligible (non-test) templates; ties broken by the
  alphabetically-first template id. Deterministic and reproducible.
- TRAIN: the remaining 23 templates.

## Known limitation (stated, not hidden)
Removing validation templates leaves tiers 1 and 4 with ~5 training templates
each. Generalization failure in those tiers is therefore confounded with limited
training diversity, and will be reported as such. Per-tier results are
exploratory given 8–10 templates per tier; only large, consistent differences
warrant claims.

## Training
- Base: mlx-community/Qwen2.5-3B-Instruct-4bit, LoRA rank 8 / scale 16, last 8
  layers, lr 1e-4, batch 4, seq len 1024 (all unchanged from the prior run).
- Train to 500 iterations. Checkpoint every 50.

## Checkpoint sweep (model selection)
- At each checkpoint, evaluate on a FIXED sample of 25 rows PER validation
  template (100 rows total), the same rows at every checkpoint. Sampling is
  per-template, not per-row, so paraphrase counts do not weight the signal.
- Log per checkpoint: train loss, in-distribution val loss, and validation
  macro-average execution-match (mean of per-template exec-match rates, each
  validation template weighted equally).

## Selection rule (PRE-REGISTERED — do not change after seeing curves)
Select the checkpoint that MAXIMIZES macro-average execution-match across the 4
validation templates on the fixed 100-row sample. Break ties by macro-average
exact-match, then by earlier checkpoint (favor less training). Then re-run the
FULL validation set once on the chosen checkpoint to confirm the sampled ranking
was not a sampling artifact.

## Final evaluation (done ONCE)
Run the selected checkpoint on the FULL untouched TEST templates. Report:
- macro-average execution-match (primary headline),
- row-level execution-match,
- exact-match,
- per-template table and per-tier table (exploratory),
- the train-loss / val-loss / validation-exec-match learning curve, and whether
  in-distribution val loss tracked template-disjoint accuracy.

## Primary reported metric
Macro-average execution-match on the test templates. Not row-level — row-level is
dominated by templates that paraphrasing happened to multiply.

---

# Experiment 2 — decomposing the generalization failure (added 2026-07-26)

Pre-registered before writing any new template or convention spec. Baseline
(Exp 1 result) is fixed at 6.8% macro-average execution-match on the 8 test
templates. Test templates and their gold SQL remain FROZEN and unlooked-at while
new templates and the convention spec are authored.

## Hypothesis
The 6.8% has two separable causes: (a) starved training diversity in tiers 1 & 4
(~5 templates each), and (b) unspecified gold conventions (column projection,
sort direction) that the metric scores as failures even when the SQL is
defensible (e.g. `SELECT sku` vs gold `SELECT sku, name`).

## Prediction (pre-registered)
Convention specification (Arm 2) yields a LARGER gain than template expansion,
because error analysis showed most zeros were convention mismatches, not
reasoning errors. Stacking (Arm 1+2) may exceed Arm 2 alone but with diminishing
returns.

## Design: 3 cells (test set identical across all)
- BASELINE: original 23-template train, no convention spec. = 6.8% (done).
- ARM 2: original templates, convention spec appended to system prompt.
- ARM 1+2: expanded templates (~15 added, blind to test) + convention spec.

## What changes, per arm
- Template expansion (in Arm 1+2): ~15 new templates authored from schema join
  structure and generic query patterns, NEVER consulting test-template SQL.
  Rebalance so every tier has comparable template counts (no starved tier).
- Convention spec (Arm 2 and 1+2): explicit rules appended to the system prompt,
  derived ONLY from TRAINING gold conventions, not from inspecting test gold:
  projection columns for "show/list/which" queries, and that
  "shortest/lowest/fewest/least" implies ASC ordering.

## Frozen (identical to Experiment 1)
Base model, LoRA hyperparameters, schema, paraphrase pipeline, three-way
template-stratified split method, pre-registered selection rule (max validation
macro-exec -> tie macro-exact -> tie earlier checkpoint), all metrics.

## Per-arm procedure
Regenerate dataset -> paraphrase new questions -> three-way stratified split ->
sweep-train (500 iters, checkpoint every 50) -> checkpoint-sweep on THAT arm's
validation templates -> select one checkpoint by the frozen rule -> evaluate
ONCE on the frozen 8 test templates.

## Reporting
3-cell table of test macro-average execution-match, per-tier deltas, and
attribution: how much gain came from conventions vs diversity, and whether they
stack. Flat or negative arms reported as-is.

## Integrity constraint
Test templates and gold SQL are not inspected during authoring of new templates
or the convention spec. Convention rules come from training gold only.

---

# Experiment 2 — AMENDED (2026-07-27, before authoring templates)

## Why amended
`src/derive_conventions.py` (reads TRAINING gold only) revealed that the
conventions the test set fails on are ABSENT from training entirely, not merely
underspecified:
- Zero simple product-lookup examples in training (both product-lookup templates
  are in the test split), so the `sku, name` projection convention appears
  nowhere in training gold.
- Ordering words in training gold map only to DESC (most/top/highest: 64/47/56);
  ASC has ZERO training examples, so "shortest/lowest/fewest" was never taught.

Therefore the Exp-1 failures were largely QUERY FAMILIES WITH ZERO TRAINING
COVERAGE, not conventions the model was taught and ignored. A convention spec
derived honestly from training (the pre-registration's integrity rule) CANNOT
address them, because training is silent on them. Writing those rules anyway
would require inspecting test needs = teach-to-the-test. So Arm 2 as originally
framed is hollow.

## Revised design (supersedes the 3-cell plan above)
Primary lever becomes TEMPLATE EXPANSION (coverage), authored blind to test:
- BASELINE: 6.8% (done).
- ARM A (primary): add training templates covering the query FAMILIES that had
  zero/low coverage -- ASC ordering, product-column projection, and additional
  join-path and superlative patterns -- authored generically from the schema,
  NEVER by copying test-template SQL. Rebalance tiers so none is starved.
- ARM A+conv (secondary): Arm A plus a MINIMAL convention line that is genuinely
  grounded in training gold -- only the DESC-superlative mapping that already
  exists; NOT the ASC or projection rules (those aren't in training).

## Revised hypothesis / prediction
Giving the model ANY exposure to a query family (vs zero) is what enables
within-family generalization to held-out shapes. Expansion should lift the
tiers/families that were previously uncovered; families already covered in
Exp 1 (e.g. count+single-join, which already worked) change little.

## Integrity (unchanged)
Test templates and gold SQL remain frozen and unlooked-at. New templates are
authored from schema structure and generic query patterns only. Everything else
frozen as in Exp 1.
