"""Parameterized NL/SQL template pairs.

SQL is generated programmatically rather than by a language model, so the
question and the query originate from the same source and cannot silently
disagree. Execution validation catches crashes; construction guarantees intent.

Tiers reflect query complexity:
  1 = single table          2 = one or two joins
  3 = multi-join aggregate  4 = date arithmetic / subquery / superlative
"""

import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "mfg.db"
ASOF = "2026-06-30"


@dataclass
class Template:
    tid: str
    tier: int
    build: Callable[[dict, random.Random], tuple]
    tags: list = field(default_factory=list)


def sample_values(conn):
    """Real values pulled from the DB so generated questions are answerable."""
    q = lambda s: [r[0] for r in conn.execute(s).fetchall()]
    return {
        "category": q("SELECT DISTINCT category FROM products"),
        "machine_type": q("SELECT DISTINCT machine_type FROM machines"),
        "machine_name": q("SELECT name FROM machines"),
        "plant": q("SELECT name FROM plants"),
        "shift": q("SELECT name FROM shifts"),
        "severity": ["low", "medium", "high", "critical"],
        "defect_cat": q("SELECT name FROM defect_categories"),
        "maint_type": ["preventive", "corrective"],
        "wo_status": ["open", "in_progress", "completed", "cancelled"],
        "role": ["operator", "inspector", "technician", "supervisor"],
        "state": ["running", "idle", "down", "setup"],
        "manufacturer": q("SELECT DISTINCT manufacturer FROM machines"),
        "country": q("SELECT DISTINCT country FROM plants"),
        "line_name": q("SELECT name FROM production_lines"),
        "task_type": q("SELECT DISTINCT task_type FROM maintenance_schedules"),
        "inspector": q("SELECT DISTINCT e.name FROM employees e "
                       "JOIN inspections i ON i.inspector_id = e.employee_id"),
    }


TEMPLATES = []


def template(tid, tier, tags=()):
    def deco(fn):
        TEMPLATES.append(Template(tid, tier, fn, list(tags)))
        return fn
    return deco


# ---------- tier 1: single table ----------

@template("t1_machines_by_type", 1, ["filter"])
def _(v, r):
    t = r.choice(v["machine_type"])
    return ([f"List all {t} machines.",
             f"Which machines are of type {t}?",
             f"Show me every {t} in the plant."],
            f"SELECT name FROM machines WHERE machine_type = '{t}';")


@template("t1_count_employees_role", 1, ["count", "filter"])
def _(v, r):
    role = r.choice(v["role"])
    return ([f"How many {role}s do we employ?",
             f"Count the {role}s.",
             f"What is the total number of {role}s on staff?"],
            f"SELECT COUNT(*) FROM employees WHERE role = '{role}';")


@template("t1_products_in_category", 1, ["filter"])
def _(v, r):
    c = r.choice(v["category"])
    return ([f"Show the SKUs in the {c} category.",
             f"Which products are categorised as {c}?",
             f"List every {c} product with its SKU."],
            f"SELECT sku, name FROM products WHERE category = '{c}';")


# ---------- tier 2: one or two joins ----------

@template("t2_machines_at_plant", 2, ["join"])
def _(v, r):
    p = r.choice(v["plant"])
    return ([f"Which machines are installed at {p}?",
             f"List the machines at the {p} site.",
             f"Show all machines belonging to {p}."],
            "SELECT m.name FROM machines m "
            "JOIN production_lines l ON l.line_id = m.line_id "
            "JOIN plants p ON p.plant_id = l.plant_id "
            f"WHERE p.name = '{p}';")


@template("t2_open_orders_by_status", 2, ["join", "count"])
def _(v, r):
    s = r.choice(v["wo_status"])
    return ([f"How many work orders are {s.replace('_', ' ')}?",
             f"Count work orders with status {s}.",
             f"What's the number of {s.replace('_', ' ')} work orders?"],
            f"SELECT COUNT(*) FROM work_orders WHERE status = '{s}';")


@template("t2_defects_by_severity", 2, ["join", "count", "filter"])
def _(v, r):
    sev = r.choice(v["severity"])
    return ([f"How many {sev} severity defects were logged?",
             f"Count all defects graded {sev}.",
             f"What is the total quantity of {sev} defects?"],
            f"SELECT SUM(quantity) FROM defect_logs WHERE severity = '{sev}';")


# ---------- scope helpers ----------
# Aggregate templates take a scope parameter (whole dataset, one plant, a recent
# window, one shift). Without it each aggregate yields exactly one SQL string and
# collapses under de-duplication, leaving the most valuable tier the thinnest.
# Scopes vary join paths and predicates, not just wording.

PLANT_FROM_RUN = ("JOIN machines mm ON mm.machine_id = r.machine_id "
                  "JOIN production_lines ll ON ll.line_id = mm.line_id "
                  "JOIN plants pp ON pp.plant_id = ll.plant_id ")

DEFECT_TO_PLANT = ("JOIN inspections ii ON ii.inspection_id = d.inspection_id "
                   "JOIN production_runs rr ON rr.run_id = ii.run_id "
                   "JOIN machines mm ON mm.machine_id = rr.machine_id "
                   "JOIN production_lines ll ON ll.line_id = mm.line_id "
                   "JOIN plants pp ON pp.plant_id = ll.plant_id ")

MACHINE_TO_PLANT = ("JOIN production_lines ll ON ll.line_id = mm.line_id "
                    "JOIN plants pp ON pp.plant_id = ll.plant_id ")


def run_scope(v, r, allow=("all", "plant", "days", "shift")):
    """Scope for aggregates built on production_runs aliased r."""
    kind = r.choice(list(allow))
    if kind == "all":
        return "", "", ""
    if kind == "plant":
        pl = r.choice(v["plant"])
        return f" at {pl}", PLANT_FROM_RUN, f"pp.name = '{pl}'"
    if kind == "days":
        n = r.choice([30, 60, 90, 180])
        return (f" over the last {n} days", "",
                f"r.started_at >= date('{ASOF}', '-{n} days')")
    sh = r.choice(v["shift"])
    return (f" on the {sh} shift", "JOIN shifts ss ON ss.shift_id = r.shift_id ",
            f"ss.name = '{sh}'")


def defect_scope(v, r, allow=("all", "plant", "severity", "days")):
    """Scope for aggregates built on defect_logs aliased d."""
    kind = r.choice(list(allow))
    if kind == "all":
        return "", "", ""
    if kind == "plant":
        pl = r.choice(v["plant"])
        return f" at {pl}", DEFECT_TO_PLANT, f"pp.name = '{pl}'"
    if kind == "severity":
        sev = r.choice(v["severity"])
        return f" for {sev} severity defects", "", f"d.severity = '{sev}'"
    n = r.choice([30, 60, 90, 180])
    return (f" in the last {n} days", "",
            f"d.logged_at >= date('{ASOF}', '-{n} days')")


def maint_scope(v, r, allow=("all", "maint_type", "days")):
    """Scope for aggregates built on maintenance_orders aliased o."""
    kind = r.choice(list(allow))
    if kind == "all":
        return "", "", ""
    if kind == "maint_type":
        mt = r.choice(v["maint_type"])
        return f" for {mt} maintenance", "", f"o.maint_type = '{mt}'"
    n = r.choice([30, 60, 90, 180])
    return (f" in the last {n} days", "",
            f"o.opened_at >= date('{ASOF}', '-{n} days')")


def clause(cond, existing=False):
    if not cond:
        return ""
    return ("AND " if existing else "WHERE ") + cond + " "


# ---------- tier 3: multi-join aggregates ----------

@template("t3_scrap_by_category", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, joins, cond = run_scope(v, r)
    sql = ("SELECT p.category, "
           "ROUND(100.0 * SUM(r.units_scrapped) / SUM(r.units_produced), 2) AS scrap_pct "
           "FROM production_runs r "
           "JOIN work_orders w ON w.work_order_id = r.work_order_id "
           "JOIN products p ON p.product_id = w.product_id " + joins + clause(cond) +
           "GROUP BY p.category ORDER BY scrap_pct DESC;")
    return ([f"What is the scrap rate for each product category{sfx}?",
             f"Show scrap percentage broken down by product category{sfx}.",
             f"Compare scrap rates across product categories{sfx}."], sql)


@template("t3_output_by_shift", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, joins, cond = run_scope(v, r, allow=("all", "plant", "days"))
    sql = ("SELECT s.name, SUM(r.units_produced) AS units "
           "FROM production_runs r JOIN shifts s ON s.shift_id = r.shift_id " +
           joins + clause(cond) + "GROUP BY s.name ORDER BY units DESC;")
    return ([f"How many units did each shift produce{sfx}?",
             f"Show total production volume by shift{sfx}.",
             f"Break down units produced per shift{sfx}."], sql)


@template("t3_downtime_by_machine_type", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, joins, cond = maint_scope(v, r)
    sql = ("SELECT m.machine_type, SUM(o.downtime_minutes) AS mins "
           "FROM maintenance_orders o JOIN machines m ON m.machine_id = o.machine_id " +
           joins + "WHERE o.downtime_minutes IS NOT NULL " + clause(cond, True) +
           "GROUP BY m.machine_type ORDER BY mins DESC;")
    return ([f"Which machine types accumulate the most downtime{sfx}?",
             f"Total downtime minutes grouped by machine type{sfx}.",
             f"Show downtime by type of machine{sfx}."], sql)


@template("t3_defects_by_plant", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, _, cond = defect_scope(v, r, allow=("all", "severity", "days"))
    sql = ("SELECT pl.name, SUM(d.quantity) AS defects "
           "FROM defect_logs d "
           "JOIN inspections i ON i.inspection_id = d.inspection_id "
           "JOIN production_runs r ON r.run_id = i.run_id "
           "JOIN machines m ON m.machine_id = r.machine_id "
           "JOIN production_lines l ON l.line_id = m.line_id "
           "JOIN plants pl ON pl.plant_id = l.plant_id " + clause(cond) +
           "GROUP BY pl.name ORDER BY defects DESC;")
    return ([f"How many defects were logged at each plant{sfx}?",
             f"Break down defect quantity by plant{sfx}.",
             f"Show total defects per manufacturing site{sfx}."], sql)


# ---------- tier 4: date arithmetic, subqueries, superlatives ----------

@template("t4_overdue_maintenance", 4, ["date", "join"])
def _(v, r):
    kind = r.choice(["all", "plant", "task"])
    joins, cond, sfx = "", "", ""
    if kind == "plant":
        pl = r.choice(v["plant"])
        joins = ("JOIN production_lines ll ON ll.line_id = m.line_id "
                 "JOIN plants pp ON pp.plant_id = ll.plant_id ")
        cond, sfx = f"AND pp.name = '{pl}' ", f" at {pl}"
    elif kind == "task":
        t = r.choice(v["task_type"])
        cond, sfx = f"AND s.task_type = '{t}' ", f" for {t.lower()}"
    sql = ("SELECT m.name, s.task_type, s.last_completed_on "
           "FROM maintenance_schedules s "
           "JOIN machines m ON m.machine_id = s.machine_id " + joins +
           f"WHERE julianday('{ASOF}') - julianday(s.last_completed_on) > s.interval_days " +
           cond + "ORDER BY m.name;")
    return ([f"Which machines are overdue for maintenance{sfx}?",
             f"List machines past their maintenance interval{sfx}.",
             f"Show overdue maintenance schedules{sfx}."], sql)


@template("t4_worst_operator_by_scrap", 4, ["superlative", "join", "aggregate"])
def _(v, r):
    sh = r.choice(v["shift"])
    sql = ("SELECT e.name, "
           "ROUND(100.0 * SUM(r.units_scrapped) / SUM(r.units_produced), 2) AS scrap_pct "
           "FROM production_runs r "
           "JOIN employees e ON e.employee_id = r.operator_id "
           "JOIN shifts s ON s.shift_id = r.shift_id "
           f"WHERE s.name = '{sh}' "
           "GROUP BY e.name HAVING SUM(r.units_produced) > 0 "
           "ORDER BY scrap_pct DESC LIMIT 1;")
    return ([f"Which {sh} shift operator has the highest scrap rate?",
             f"Find the {sh} shift operator with the worst scrap percentage.",
             f"Who scraps the most units on the {sh} shift, proportionally?"], sql)


# ---------- tier 1 (extended) ----------

@template("t1_machines_by_manufacturer", 1, ["filter"])
def _(v, r):
    m = r.choice(v["manufacturer"])
    return ([f"Which machines were made by {m}?",
             f"List all {m} equipment.",
             f"Show machines from the manufacturer {m}."],
            f"SELECT name, machine_type FROM machines WHERE manufacturer = '{m}';")


@template("t1_slowest_products", 1, ["order", "limit"])
def _(v, r):
    n = r.choice([3, 5, 10])
    fast = r.choice([False, True])
    word = "shortest" if fast else "longest"
    direction = "ASC" if fast else "DESC"
    return ([f"Which {n} products have the {word} target cycle time?",
             f"Show the {n} products with the {word} cycle time target.",
             f"List the top {n} products ranked by {word} target cycle time."],
            "SELECT sku, name, target_cycle_time_sec FROM products "
            f"ORDER BY target_cycle_time_sec {direction} LIMIT {n};")


@template("t1_plants_list", 1, [])
def _(v, r):
    country = r.choice([None] + v["country"])
    if country is None:
        return (["List all our plants and where they are located.",
                 "Show every manufacturing site with its city and country.",
                 "What plants do we operate and where?"],
                "SELECT name, city, country FROM plants ORDER BY name;")
    return ([f"Which plants do we operate in {country}?",
             f"List our manufacturing sites located in {country}.",
             f"Show the {country} plants and their cities."],
            "SELECT name, city FROM plants "
            f"WHERE country = '{country}' ORDER BY name;")


@template("t1_defect_cats_by_severity", 1, ["filter"])
def _(v, r):
    sev = r.choice(v["severity"])
    return ([f"Which defect categories default to {sev} severity?",
             f"List defect types classified as {sev}.",
             f"Show all {sev} severity defect categories."],
            f"SELECT name FROM defect_categories WHERE default_severity = '{sev}';")


@template("t1_recent_hires", 1, ["date", "filter"])
def _(v, r):
    year = r.choice([2019, 2021, 2022, 2023, 2024])
    return ([f"Which employees were hired since the start of {year}?",
             f"List staff with a hire date on or after {year}-01-01.",
             f"Show our hires from {year} onward."],
            "SELECT name, role, hire_date FROM employees "
            f"WHERE hire_date >= '{year}-01-01' ORDER BY hire_date DESC;")


# ---------- tier 2 (extended) ----------

@template("t2_employees_at_plant_by_role", 2, ["join", "filter"])
def _(v, r):
    pl, role = r.choice(v["plant"]), r.choice(v["role"])
    return ([f"Who are the {role}s at {pl}?",
             f"List every {role} working at {pl}.",
             f"Show {pl} staff with the {role} role."],
            "SELECT e.name FROM employees e "
            "JOIN plants p ON p.plant_id = e.plant_id "
            f"WHERE p.name = '{pl}' AND e.role = '{role}';")


@template("t2_lines_at_plant", 2, ["join"])
def _(v, r):
    pl = r.choice(v["plant"])
    return ([f"What production lines does {pl} have?",
             f"List the lines at {pl}.",
             f"Show all production lines belonging to {pl}."],
            "SELECT l.name FROM production_lines l "
            "JOIN plants p ON p.plant_id = l.plant_id "
            f"WHERE p.name = '{pl}';")


@template("t2_workorders_by_product_category", 2, ["join", "count"])
def _(v, r):
    c = r.choice(v["category"])
    return ([f"How many work orders were raised for {c} products?",
             f"Count work orders in the {c} category.",
             f"What's the work order total for {c} items?"],
            "SELECT COUNT(*) FROM work_orders w "
            "JOIN products p ON p.product_id = w.product_id "
            f"WHERE p.category = '{c}';")


@template("t2_count_maintenance_by_type", 2, ["count", "filter"])
def _(v, r):
    mt = r.choice(v["maint_type"])
    return ([f"How many {mt} maintenance orders are there?",
             f"Count the {mt} maintenance jobs.",
             f"What is the number of {mt} maintenance orders logged?"],
            f"SELECT COUNT(*) FROM maintenance_orders WHERE maint_type = '{mt}';")


@template("t2_machines_on_line", 2, ["join", "filter"])
def _(v, r):
    ln = r.choice(v["line_name"])
    return ([f"Which machines are on {ln}?",
             f"List the equipment installed on {ln}.",
             f"Show every machine assigned to {ln}."],
            "SELECT m.name, m.machine_type FROM machines m "
            "JOIN production_lines l ON l.line_id = m.line_id "
            f"WHERE l.name = '{ln}';")


@template("t2_inspections_by_inspector", 2, ["join", "count"])
def _(v, r):
    ins = r.choice(v["inspector"])
    return ([f"How many inspections has {ins} carried out?",
             f"Count the inspections performed by {ins}.",
             f"What is {ins}'s total inspection count?"],
            "SELECT COUNT(*) FROM inspections i "
            "JOIN employees e ON e.employee_id = i.inspector_id "
            f"WHERE e.name = '{ins}';")


# ---------- tier 3 (extended) ----------

@template("t3_defect_qty_by_category_name", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, joins, cond = defect_scope(v, r, allow=("all", "plant", "days"))
    sql = ("SELECT c.name, SUM(d.quantity) AS qty FROM defect_logs d "
           "JOIN defect_categories c ON c.category_id = d.category_id " +
           joins + clause(cond) + "GROUP BY c.name ORDER BY qty DESC;")
    return ([f"What is the total defect quantity for each defect category{sfx}?",
             f"Break down logged defects by category{sfx}.",
             f"Show defect volume grouped by defect type{sfx}."], sql)


@template("t3_production_by_plant", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, _, cond = run_scope(v, r, allow=("all", "days", "shift"))
    joins = "JOIN shifts ss ON ss.shift_id = r.shift_id " if "shift" in sfx else ""
    sql = ("SELECT pl.name, SUM(r.units_produced) AS units "
           "FROM production_runs r "
           "JOIN machines m ON m.machine_id = r.machine_id "
           "JOIN production_lines l ON l.line_id = m.line_id "
           "JOIN plants pl ON pl.plant_id = l.plant_id " + joins + clause(cond) +
           "GROUP BY pl.name ORDER BY units DESC;")
    return ([f"How many units has each plant produced{sfx}?",
             f"Show total production volume per plant{sfx}.",
             f"Break down units produced by site{sfx}."], sql)


@template("t3_avg_downtime_by_maint_type", 3, ["aggregate", "group"])
def _(v, r):
    sfx, _, cond = maint_scope(v, r, allow=("all", "days"))
    sql = ("SELECT maint_type, ROUND(AVG(downtime_minutes), 1) AS avg_mins "
           "FROM maintenance_orders o WHERE downtime_minutes IS NOT NULL " +
           clause(cond, True) + "GROUP BY maint_type ORDER BY avg_mins DESC;")
    return ([f"What is the average downtime by maintenance type{sfx}?",
             f"Compare mean downtime minutes for preventive versus corrective work{sfx}.",
             f"Show average downtime grouped by maintenance type{sfx}."], sql)


@template("t3_scrap_by_machine_type", 3, ["join", "aggregate", "group"])
def _(v, r):
    sfx, joins, cond = run_scope(v, r, allow=("all", "plant", "shift", "days"))
    sql = ("SELECT m.machine_type, "
           "ROUND(100.0 * SUM(r.units_scrapped) / SUM(r.units_produced), 2) AS scrap_pct "
           "FROM production_runs r JOIN machines m ON m.machine_id = r.machine_id " +
           joins + clause(cond) +
           "GROUP BY m.machine_type ORDER BY scrap_pct DESC;")
    return ([f"Which machine types have the highest scrap rates{sfx}?",
             f"Show scrap percentage by type of machine{sfx}.",
             f"Compare scrap rate across machine types{sfx}."], sql)


@template("t3_fail_rate_by_plant", 3, ["join", "aggregate", "group"])
def _(v, r):
    kind = r.choice(["all", "days"])
    cond, sfx = "", ""
    if kind == "days":
        n = r.choice([30, 60, 90, 180])
        cond = f"WHERE i.inspected_at >= date('{ASOF}', '-{n} days') "
        sfx = f" in the last {n} days"
    sql = ("SELECT pl.name, "
           "ROUND(100.0 * SUM(i.units_failed) / SUM(i.sample_size), 2) AS fail_pct "
           "FROM inspections i "
           "JOIN production_runs r ON r.run_id = i.run_id "
           "JOIN machines m ON m.machine_id = r.machine_id "
           "JOIN production_lines l ON l.line_id = m.line_id "
           "JOIN plants pl ON pl.plant_id = l.plant_id " + cond +
           "GROUP BY pl.name ORDER BY fail_pct DESC;")
    return ([f"What is the inspection failure rate at each plant{sfx}?",
             f"Show the percentage of inspected units that failed, by plant{sfx}.",
             f"Compare inspection fail rates across sites{sfx}."], sql)


@template("t3_unplanned_reasons", 3, ["aggregate", "group", "filter"])
def _(v, r):
    kind = r.choice(["all", "plant", "days"])
    joins, cond, sfx = "", "", ""
    if kind == "plant":
        pl = r.choice(v["plant"])
        joins = "JOIN machines mm ON mm.machine_id = de.machine_id " + MACHINE_TO_PLANT
        cond, sfx = f"AND pp.name = '{pl}' ", f" at {pl}"
    elif kind == "days":
        n = r.choice([30, 60, 90, 180])
        cond = f"AND de.started_at >= date('{ASOF}', '-{n} days') "
        sfx = f" in the last {n} days"
    sql = ("SELECT de.reason, COUNT(*) AS events FROM downtime_events de " + joins +
           "WHERE de.category = 'unplanned' " + cond +
           "GROUP BY de.reason ORDER BY events DESC;")
    return ([f"What are the most common causes of unplanned downtime{sfx}?",
             f"Break down unplanned downtime events by reason{sfx}.",
             f"Show the frequency of each unplanned downtime cause{sfx}."], sql)


# ---------- tier 4 (extended) ----------

@template("t4_top_defect_category", 4, ["superlative", "join", "aggregate"])
def _(v, r):
    sfx, joins, cond = defect_scope(v, r, allow=("all", "plant", "days"))
    sql = ("SELECT c.name, SUM(d.quantity) AS qty FROM defect_logs d "
           "JOIN defect_categories c ON c.category_id = d.category_id " +
           joins + clause(cond) +
           "GROUP BY c.name ORDER BY qty DESC LIMIT 1;")
    return ([f"What is the single most common defect type by quantity{sfx}?",
             f"Which defect category accounts for the most defective units{sfx}?",
             f"Find the top defect category by total quantity{sfx}."], sql)


@template("t4_machines_above_avg_scrap", 4, ["subquery", "join", "aggregate"])
def _(v, r):
    # The comparison baseline is scope-consistent: "above average at plant X"
    # compares against plant X's own average, not the company-wide one. Using the
    # global average under a plant filter is both semantically wrong and can
    # return zero rows when every machine at that plant beats the global mean.
    kind = r.choice(["all", "plant"])
    if kind == "all":
        joins, cond, sfx = "", "", ""
        sub = ("SELECT 100.0 * SUM(units_scrapped) / SUM(units_produced) "
               "FROM production_runs")
    else:
        pl = r.choice(v["plant"])
        joins, cond, sfx = PLANT_FROM_RUN, f"pp.name = '{pl}'", f" at {pl}"
        sub = ("SELECT 100.0 * SUM(r2.units_scrapped) / SUM(r2.units_produced) "
               "FROM production_runs r2 "
               "JOIN machines m2 ON m2.machine_id = r2.machine_id "
               "JOIN production_lines l2 ON l2.line_id = m2.line_id "
               "JOIN plants p2 ON p2.plant_id = l2.plant_id "
               f"WHERE p2.name = '{pl}'")
    sql = ("SELECT m.name, "
           "ROUND(100.0 * SUM(r.units_scrapped) / SUM(r.units_produced), 2) AS scrap_pct "
           "FROM production_runs r JOIN machines m ON m.machine_id = r.machine_id " +
           joins + clause(cond) + "GROUP BY m.name "
           f"HAVING scrap_pct > ({sub}) ORDER BY scrap_pct DESC;")
    return ([f"Which machines have a scrap rate above the overall average{sfx}?",
             f"List machines scrapping more than the plant-wide average rate{sfx}.",
             f"Show machines performing worse than average on scrap{sfx}."], sql)


@template("t4_runs_last_n_days", 4, ["date", "count"])
def _(v, r):
    n = r.choice([7, 14, 30, 60, 90])
    kind = r.choice(["all", "plant", "shift"])
    joins, cond, sfx = "", "", ""
    if kind == "plant":
        pl = r.choice(v["plant"])
        joins = PLANT_FROM_RUN
        cond, sfx = f"AND pp.name = '{pl}' ", f" at {pl}"
    elif kind == "shift":
        sh = r.choice(v["shift"])
        joins = "JOIN shifts ss ON ss.shift_id = r.shift_id "
        cond, sfx = f"AND ss.name = '{sh}' ", f" on the {sh} shift"
    sql = ("SELECT COUNT(*) FROM production_runs r " + joins +
           f"WHERE r.started_at >= date('{ASOF}', '-{n} days') " + cond + ";")
    return ([f"How many production runs started in the last {n} days{sfx}?",
             f"Count runs begun within the past {n} days{sfx}.",
             f"What is the number of production runs in the previous {n} days{sfx}?"], sql)


@template("t4_longest_downtime_event", 4, ["superlative", "date", "join"])
def _(v, r):
    kind = r.choice(["all", "category", "plant"])
    joins, cond, sfx = "", "", ""
    if kind == "category":
        c = r.choice(["planned", "unplanned"])
        cond, sfx = f"AND d.category = '{c}' ", f" ({c})"
    elif kind == "plant":
        pl = r.choice(v["plant"])
        joins = MACHINE_TO_PLANT.replace("mm.line_id", "m.line_id")
        cond, sfx = f"AND pp.name = '{pl}' ", f" at {pl}"
    sql = ("SELECT m.name, d.reason, "
           "ROUND((julianday(d.ended_at) - julianday(d.started_at)) * 24 * 60) AS minutes "
           "FROM downtime_events d JOIN machines m ON m.machine_id = d.machine_id " +
           joins + "WHERE d.ended_at IS NOT NULL " + cond +
           "ORDER BY minutes DESC LIMIT 1;")
    return ([f"What was the longest single downtime event{sfx}, and on which machine?",
             f"Find the downtime event with the greatest duration{sfx}.",
             f"Show the machine that suffered the longest continuous downtime{sfx}."], sql)


@template("t4_hottest_machine_recent", 4, ["superlative", "date", "join", "aggregate"])
def _(v, r):
    n = r.choice([7, 14, 21])
    state = r.choice(["running", "idle"])
    sql = ("SELECT m.name, ROUND(AVG(sr.temperature_c), 2) AS avg_temp "
           "FROM sensor_readings sr JOIN machines m ON m.machine_id = sr.machine_id "
           f"WHERE sr.reading_ts >= date('{ASOF}', '-{n} days') "
           f"AND sr.machine_state = '{state}' "
           "GROUP BY m.name ORDER BY avg_temp DESC LIMIT 1;")
    return ([f"Which machine ran hottest on average while {state} over the last {n} days?",
             f"Find the machine with the highest average {state} temperature in the past {n} days.",
             f"Show the hottest machine while {state}, last {n} days."], sql)


@template("t4_machines_without_recent_maintenance", 4, ["subquery", "date"])
def _(v, r):
    n = r.choice([14, 20, 30])
    sql = ("SELECT m.name FROM machines m WHERE NOT EXISTS ("
           "SELECT 1 FROM maintenance_orders o WHERE o.machine_id = m.machine_id "
           f"AND o.opened_at >= date('{ASOF}', '-{n} days')) ORDER BY m.name;")
    return ([f"Which machines have had no maintenance in the last {n} days?",
             f"List equipment with no maintenance order opened in the past {n} days.",
             f"Show machines untouched by maintenance for {n} days."], sql)


def generate(n_per_template=1, seed=7):
    conn = sqlite3.connect(DB_PATH)
    vals = sample_values(conn)
    rng = random.Random(seed)
    out = []
    for t in TEMPLATES:
        for _ in range(n_per_template):
            questions, sql = t.build(vals, rng)
            out.append({"template_id": t.tid, "tier": t.tier, "tags": t.tags,
                        "questions": questions, "sql": sql})
    conn.close()
    return out


if __name__ == "__main__":
    from collections import Counter

    conn = sqlite3.connect(DB_PATH)
    seen, bad, empty = {}, [], []
    for seed in range(60):
        for p in generate(n_per_template=1, seed=seed):
            if p["sql"] in seen:
                continue
            seen[p["sql"]] = p["template_id"]
            try:
                if not conn.execute(p["sql"]).fetchall():
                    empty.append((p["template_id"], p["sql"]))
            except Exception as e:
                bad.append((p["template_id"], str(e), p["sql"]))

    per_tmpl = Counter(seen.values())
    tiers = Counter(t.tier for t in TEMPLATES)
    print(f"templates: {len(TEMPLATES)}   tiers {dict(sorted(tiers.items()))}")
    print(f"unique SQL variants: {len(seen)}")
    print(f"execution failures: {len(bad)}")
    for tid, err, sql in bad[:5]:
        print("   FAIL", tid, "->", err)
        print("        ", sql[:160])
    print(f"empty result sets: {len(empty)}")
    for tid, sql in empty[:5]:
        print("   EMPTY", tid)
        print("        ", sql[:160])
    thin = [(t, c) for t, c in sorted(per_tmpl.items()) if c < 3]
    print(f"templates with <3 variants: {len(thin)} {thin if thin else ''}")
    conn.close()
