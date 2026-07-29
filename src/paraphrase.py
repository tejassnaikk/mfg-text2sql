"""Diversify question phrasing with a local MLX model. Gold SQL is never touched.

The risk this guards against: a fluent paraphrase that silently drops a
parameter ("runs in the last 30 days at Rivermill" -> "recent runs") produces a
question whose gold SQL no longer answers it -- the same
silently-wrong-but-runnable failure the template approach exists to avoid. Every
candidate is checked for entity preservation and discarded if it fails, so a
weaker local paraphraser raises the rejection rate rather than corrupting labels.

The paraphraser is deliberately a different model family from the fine-tune
target. Generating training questions with the same family being trained would
place them inside that model's own distribution and flatter its eval numbers.

Run is resumable: results are cached by question and flushed periodically.
"""

import json
import os
import re
import signal
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dataset"
CACHE_PATH = DATA / "paraphrase_cache.json"

os.environ.setdefault("HF_HOME", "/Volumes/Tejas SSD/hf-cache")

MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
N_PARAPHRASES = 4
MAX_TOKENS = 220
FLUSH_EVERY = 20
ASOF = "2026-06-30"

PROMPT = """Rewrite this factory analyst's question in {n} different ways.

Vary the wording and sentence structure. Some should be terse and lowercase, \
some full polite sentences, some clipped and informal. Keep every \
specific detail exactly as given: plant names, shift names, category names, \
numbers, time windows, severities. Do not add or remove any filter. Never \
invent codes, abbreviations, IDs, times, or any detail that is not in the \
original question. Do not mention SQL or database tables.

Reply with a JSON array of {n} strings and nothing else.

Question: {q}"""


DAY_WORDS = {
    "7": ["7", "week"],
    "14": ["14", "two weeks", "2 weeks", "fortnight"],
    "21": ["21", "three weeks", "3 weeks"],
    "30": ["30", "month"],
    "60": ["60", "two months", "2 months"],
    "90": ["90", "quarter", "three months", "3 months"],
    "180": ["180", "six months", "6 months", "half year"],
}


def required_entities(sql):
    """Groups of alternatives; a faithful paraphrase must match one per group.

    Natural time expressions count: "last week" is a legitimate rendering of a
    7-day window, and rejecting it would discard exactly the phrasing variety
    this pass exists to create.
    """
    req = []
    for lit in re.findall(r"'([^']*)'", sql):
        if lit == ASOF or not lit.strip():
            continue
        m = re.fullmatch(r"-(\d+) days", lit)
        if m:
            req.append(DAY_WORDS.get(m.group(1), [m.group(1)]))
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", lit):
            req.append([lit[:4]])
        else:
            req.append([lit])
    for n in re.findall(r"LIMIT (\d+)", sql):
        if n != "1":
            req.append([n])
    return req


def norm(s):
    return re.sub(r"[\s_]+", " ", s.lower())


def preserves(text, groups):
    t = norm(text)
    return all(any(norm(a) in t for a in g) for g in groups)


def literal_vocab(all_sql):
    """Every string literal used as a filter anywhere in the corpus.

    Any of these appearing in a paraphrase whose own SQL does not filter on it
    is a hallucinated constraint. The presence check alone cannot catch that:
    it verifies nothing was dropped, never that nothing was added.
    """
    vocab = set()
    for sql in all_sql:
        for lit in re.findall(r"'([^']*)'", sql):
            if lit == ASOF or not lit.strip():
                continue
            if re.fullmatch(r"-?\d+( days)?|\d{4}-\d{2}-\d{2}", lit):
                continue
            vocab.add(lit)
    return vocab


CANDIDATE_COLUMNS = [
    ("plants", "name"), ("plants", "city"), ("plants", "country"),
    ("production_lines", "name"), ("shifts", "name"), ("employees", "role"),
    ("machines", "machine_type"), ("machines", "manufacturer"),
    ("products", "category"), ("work_orders", "status"),
    ("defect_categories", "name"), ("defect_categories", "default_severity"),
    ("defect_logs", "severity"), ("maintenance_orders", "maint_type"),
    ("maintenance_schedules", "task_type"), ("downtime_events", "category"),
    ("downtime_events", "reason"), ("sensor_readings", "machine_state"),
]


def column_domains(db_path):
    """Distinct values per low-cardinality text column."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    out = {}
    for table, col in CANDIDATE_COLUMNS:
        vals = [r[0] for r in conn.execute(
            f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL")]
        if len(vals) <= 25:
            out[(table, col)] = {norm(v) for v in vals if isinstance(v, str)}
    conn.close()
    return out


def grouped_domains(sql, domains):
    """Values legitimately nameable because the query GROUPs BY their column.

    A query that groups by shift returns every shift, so a paraphrase saying
    "Day, Swing, Night" is describing the breakdown, not adding a filter.
    Filtered columns are not grouped, so a wrong filter value is still caught.
    """
    aliases = {}
    for tbl, alias in re.findall(r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?!ON\b|WHERE\b|GROUP\b)(\w+))?", sql):
        aliases[(alias or tbl).lower()] = tbl.lower()

    allowed = set()
    for clause in re.findall(r"GROUP BY\s+(.*?)(?:\s+ORDER BY|\s+HAVING|\s+LIMIT|;|$)",
                             sql, flags=re.I | re.S):
        for ref in clause.split(","):
            ref = ref.strip()
            m = re.fullmatch(r"(?:(\w+)\.)?(\w+)", ref)
            if not m:
                continue
            alias, col = m.group(1), m.group(2)
            tables = [aliases[alias.lower()]] if alias and alias.lower() in aliases \
                else list(set(aliases.values()))
            for t in tables:
                if (t, col) in domains:
                    allowed |= domains[(t, col)]
    return allowed


def foreign_entities(text, groups, vocab, allowed=frozenset()):
    """Filter values named in the paraphrase that its SQL neither filters nor groups on."""
    own = {norm(a) for g in groups for a in g} | set(allowed)
    t = norm(text)
    hits = []
    for term in vocab:
        n = norm(term)
        if n in own:
            continue
        if re.search(r"\b" + re.escape(n) + r"\b", t):
            hits.append(term)
    return hits


def parse_array(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    i, j = text.find("["), text.rfind("]")
    if i == -1 or j == -1:
        return []
    try:
        return [x.strip() for x in json.loads(text[i:j + 1]) if isinstance(x, str)]
    except Exception:
        return []


def fmt(sec):
    return f"{int(sec // 60)}m{int(sec % 60):02d}s"


def main():
    from mlx_lm import generate, load

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    splits = {}
    todo = {}
    for name in ["train", "valid", "test"]:
        rows = [json.loads(l) for l in (DATA / f"{name}.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r.get("source", "template") == "template"]
        splits[name] = rows
        for r in rows:
            if r["question"] not in cache:
                todo[r["question"]] = r["sql"]

    total = sum(len(v) for v in splits.values())
    print(f"template questions: {total}   cached: {len(cache)}   to generate: {len(todo)}")

    if todo:
        print(f"loading {MODEL} ...")
        model, tokenizer = load(MODEL)

        stop = {"flag": False}

        def on_sigint(*_):
            stop["flag"] = True
            print("\n  interrupt received -- flushing cache, rerun to resume")

        signal.signal(signal.SIGINT, on_sigint)

        t0 = time.time()
        for i, (question, _sql) in enumerate(todo.items(), 1):
            if stop["flag"]:
                break
            prompt = tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": PROMPT.format(n=N_PARAPHRASES, q=question)}],
                add_generation_prompt=True, tokenize=False)
            try:
                out = generate(model, tokenizer, prompt=prompt,
                               max_tokens=MAX_TOKENS, verbose=False)
                cache[question] = parse_array(out)
            except Exception as e:
                cache[question] = []
                print(f"  ! {type(e).__name__}: {str(e)[:70]}")

            if i % FLUSH_EVERY == 0 or i == len(todo):
                CACHE_PATH.write_text(json.dumps(cache))
                el = time.time() - t0
                eta = el / i * (len(todo) - i)
                print(f"  {i}/{len(todo)}  elapsed {fmt(el)}  eta {fmt(eta)}")

        CACHE_PATH.write_text(json.dumps(cache))
        if stop["flag"]:
            sys.exit("stopped early -- progress saved")

    stats = Counter()
    empty_models = 0
    vocab = literal_vocab(r["sql"] for rows in splits.values() for r in rows)
    domains = column_domains(ROOT / "data" / "db" / "mfg.db")
    for name, rows in splits.items():
        seen = {r["question"].strip().lower() for r in rows}
        added = []
        for r in rows:
            cands = cache.get(r["question"], [])
            if not cands:
                empty_models += 1
            ents = required_entities(r["sql"])
            for cand in cands:
                cand = cand.strip()
                key = cand.lower()
                if not cand or key in seen:
                    stats["duplicate"] += 1
                    continue
                if not preserves(cand, ents):
                    stats["rejected_dropped"] += 1
                    continue
                if foreign_entities(cand, ents, vocab,
                                    grouped_domains(r["sql"], domains)):
                    stats["rejected_invented"] += 1
                    continue
                seen.add(key)
                stats["accepted"] += 1
                added.append({**r, "question": cand, "source": "paraphrase"})
        rows = [{**r, "source": "template"} for r in rows]
        out = rows + added
        with (DATA / f"{name}.jsonl").open("w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"{name:6} {len(rows):5} -> {len(out):5} pairs  (+{len(added)})")

    judged = stats["accepted"] + stats["rejected_dropped"] + stats["rejected_invented"]
    pct = lambda k: 100 * stats[k] / judged if judged else 0
    print(f"\naccepted {stats['accepted']} ({pct('accepted'):.1f}%)   "
          f"rejected -- dropped a filter {stats['rejected_dropped']} "
          f"({pct('rejected_dropped'):.1f}%)   "
          f"invented a filter {stats['rejected_invented']} "
          f"({pct('rejected_invented'):.1f}%)   "
          f"duplicates {stats['duplicate']}")
    if empty_models:
        print(f"questions the model returned nothing usable for: {empty_models}")


if __name__ == "__main__":
    main()
