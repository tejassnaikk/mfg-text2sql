"""Populate the manufacturing operations DB with deterministic synthetic data.

The database is anchored to a fixed as-of date so that evaluation queries return
stable result sets. Re-running this script reproduces an identical database.
"""

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 20260722
ASOF = date(2026, 6, 30)
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "db" / "mfg.db"

random.seed(SEED)

FIRST = ["Aisha", "Marco", "Priya", "Devin", "Lena", "Omar", "Sofia", "Tomas",
         "Nadia", "Bryce", "Hana", "Elias", "Rosa", "Kwame", "Ingrid", "Vikram",
         "Chloe", "Andre", "Mei", "Jonas", "Farah", "Luis", "Greta", "Noah",
         "Ivy", "Rafael", "Zara", "Miles", "Anya", "Dario"]
LAST = ["Ferreira", "Okonkwo", "Lindqvist", "Marchetti", "Nakamura", "Duarte",
        "Halvorsen", "Bhatt", "Novak", "Castellanos", "Adeyemi", "Riedel",
        "Sorensen", "Marchand", "Kowalski", "Iyer", "Petrov", "Almeida",
        "Vogel", "Strand"]

PLANTS = [
    ("Aurora Works", "Aurora", "USA", "2011-04-18"),
    ("Rivermill Plant", "Cincinnati", "USA", "2008-09-02"),
    ("Northgate Facility", "Hamilton", "Canada", "2015-01-26"),
]
LINES_PER_PLANT = [3, 3, 2]
MACHINES_PER_LINE = 5

SHIFTS = [("Day", 6, 14), ("Swing", 14, 22), ("Night", 22, 6)]

MACHINE_TYPES = ["CNC Mill", "CNC Lathe", "Injection Molder", "Stamping Press",
                 "Laser Cutter", "Assembly Robot", "Surface Grinder", "Welding Cell"]
MANUFACTURERS = ["Haas", "Okuma", "Engel", "Schuler", "Trumpf", "Fanuc",
                 "Okamoto", "Kuka"]

PRODUCT_CATEGORIES = ["Bracket", "Housing", "Gear", "Connector", "Panel", "Shaft"]

# Scrap propensity by product category and shift. Without these the aggregate
# scrap rate converges to the same value everywhere, so superlative queries
# ("which category scraps most?") get decided by rounding noise rather than by
# a real difference -- which would corrupt result-set-equivalence scoring.
CATEGORY_SCRAP_MULT = {"Bracket": 0.55, "Housing": 0.95, "Gear": 1.70,
                       "Connector": 0.75, "Panel": 1.30, "Shaft": 2.40}
SHIFT_SCRAP_MULT = {1: 0.85, 2: 1.00, 3: 1.45}

DEFECT_CATEGORIES = [
    ("Surface Scratch", "low"), ("Dimensional Deviation", "medium"),
    ("Porosity", "medium"), ("Crack", "critical"), ("Burr", "low"),
    ("Discoloration", "low"), ("Weld Void", "high"), ("Misalignment", "high"),
    ("Contamination", "medium"), ("Thread Damage", "high"),
]
CATEGORY_WEIGHTS = [18, 14, 11, 4, 16, 9, 5, 7, 10, 6]

MAINT_TASKS = ["Lubrication", "Belt Inspection", "Calibration",
               "Filter Replacement", "Spindle Service", "Coolant Flush"]

PLANNED_REASONS = ["Scheduled maintenance", "Tooling changeover", "Software update"]
UNPLANNED_REASONS = ["Motor failure", "Sensor fault", "Material jam",
                     "Power interruption", "Hydraulic leak", "Tool breakage"]

SEVERITIES = ["low", "medium", "high", "critical"]


def iso_dt(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def iso_d(d):
    return d.strftime("%Y-%m-%d")


def rand_day(max_back, min_back=0):
    return ASOF - timedelta(days=random.randint(min_back, max_back))


def rand_dt(max_back, min_back=0):
    d = rand_day(max_back, min_back)
    return datetime(d.year, d.month, d.day,
                    random.randint(0, 23), random.randint(0, 59), random.randint(0, 59))


def shift_for_hour(hour):
    if 6 <= hour < 14:
        return 1
    if 14 <= hour < 22:
        return 2
    return 3


def person():
    return f"{random.choice(FIRST)} {random.choice(LAST)}"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cur = conn.cursor()

    for table in ["sensor_readings", "downtime_events", "maintenance_orders",
                  "maintenance_schedules", "defect_logs", "inspections",
                  "production_runs", "work_orders", "products", "machines",
                  "production_lines", "employees", "shifts", "defect_categories",
                  "plants"]:
        cur.execute(f"DELETE FROM {table}")

    cur.executemany(
        "INSERT INTO plants (plant_id, name, city, country, commissioned_on) "
        "VALUES (?,?,?,?,?)",
        [(i + 1, *p) for i, p in enumerate(PLANTS)])
    plant_ids = list(range(1, len(PLANTS) + 1))

    cur.executemany(
        "INSERT INTO shifts (shift_id, name, start_hour, end_hour) VALUES (?,?,?,?)",
        [(i + 1, *s) for i, s in enumerate(SHIFTS)])

    cur.executemany(
        "INSERT INTO defect_categories (category_id, name, default_severity) "
        "VALUES (?,?,?)",
        [(i + 1, *d) for i, d in enumerate(DEFECT_CATEGORIES)])
    category_ids = list(range(1, len(DEFECT_CATEGORIES) + 1))

    employees = []
    by_plant_role = {p: {r: [] for r in ["operator", "inspector", "technician", "supervisor"]}
                     for p in plant_ids}
    eid = 0
    for p in plant_ids:
        for role, count in [("operator", 12), ("inspector", 5),
                            ("technician", 5), ("supervisor", 2)]:
            for _ in range(count):
                eid += 1
                employees.append((eid, person(), role, p,
                                  iso_d(rand_day(3500, 120))))
                by_plant_role[p][role].append(eid)
    cur.executemany(
        "INSERT INTO employees (employee_id, name, role, plant_id, hire_date) "
        "VALUES (?,?,?,?,?)", employees)

    lines, line_plant = [], {}
    lid = 0
    for p, n in zip(plant_ids, LINES_PER_PLANT):
        for k in range(n):
            lid += 1
            lines.append((lid, p, f"Line {chr(64 + k + 1)}-{p}"))
            line_plant[lid] = p
    cur.executemany(
        "INSERT INTO production_lines (line_id, plant_id, name) VALUES (?,?,?)", lines)
    line_ids = list(line_plant.keys())

    machines, machines_by_line = [], {l: [] for l in line_ids}
    mid = 0
    for l in line_ids:
        for k in range(MACHINES_PER_LINE):
            mid += 1
            mtype = random.choice(MACHINE_TYPES)
            machines.append((mid, l, f"M-{mid:03d}", mtype,
                             random.choice(MANUFACTURERS),
                             f"{random.choice('XZTQ')}{random.randint(100, 999)}",
                             iso_d(rand_day(4000, 200))))
            machines_by_line[l].append(mid)
    cur.executemany(
        "INSERT INTO machines (machine_id, line_id, name, machine_type, "
        "manufacturer, model, installed_on) VALUES (?,?,?,?,?,?,?)", machines)
    machine_ids = [m[0] for m in machines]
    degraded = set(random.sample(machine_ids, 6))

    products = []
    for i in range(24):
        cat = PRODUCT_CATEGORIES[i % len(PRODUCT_CATEGORIES)]
        products.append((i + 1, f"SKU-{1000 + i}", f"{cat} Type {i // 6 + 1}",
                         cat, round(random.uniform(8.0, 95.0), 1)))
    cur.executemany(
        "INSERT INTO products (product_id, sku, name, category, "
        "target_cycle_time_sec) VALUES (?,?,?,?,?)", products)
    product_ids = [p[0] for p in products]

    product_category = {p[0]: p[3] for p in products}
    work_orders, wo_line, wo_product = [], {}, {}
    for i in range(1, 401):
        line = random.choice(line_ids)
        created = rand_day(360, 5)
        wo_line[i] = line
        prod = random.choice(product_ids)
        wo_product[i] = prod
        work_orders.append((i, prod, line,
                            random.choice([250, 500, 750, 1000, 1500, 2000]),
                            random.choices(["completed", "in_progress", "open", "cancelled"],
                                           weights=[70, 15, 10, 5])[0],
                            iso_d(created),
                            iso_d(created + timedelta(days=random.randint(7, 45)))))
    cur.executemany(
        "INSERT INTO work_orders (work_order_id, product_id, line_id, "
        "quantity_ordered, status, created_at, due_date) VALUES (?,?,?,?,?,?,?)",
        work_orders)

    runs = []
    for rid in range(1, 4001):
        wo = random.randint(1, 400)
        line = wo_line[wo]
        machine = random.choice(machines_by_line[line])
        operator = random.choice(by_plant_role[line_plant[line]]["operator"])
        start = rand_dt(360, 1)
        produced = random.randint(40, 400)
        shift = shift_for_hour(start.hour)
        scrap_rate = random.uniform(0.005, 0.05)
        scrap_rate *= CATEGORY_SCRAP_MULT[product_category[wo_product[wo]]]
        scrap_rate *= SHIFT_SCRAP_MULT[shift]
        if machine in degraded:
            scrap_rate *= 2.2
        scrapped = min(produced, int(produced * scrap_rate))
        open_run = random.random() < 0.03
        end = None if open_run else iso_dt(start + timedelta(minutes=random.randint(45, 480)))
        runs.append((rid, wo, machine, operator, shift,
                     iso_dt(start), end, produced, scrapped))
    cur.executemany(
        "INSERT INTO production_runs (run_id, work_order_id, machine_id, "
        "operator_id, shift_id, started_at, ended_at, units_produced, "
        "units_scrapped) VALUES (?,?,?,?,?,?,?,?,?)", runs)

    run_line = {r[0]: wo_line[r[1]] for r in runs}
    inspections, insp_id = [], 0
    for r in runs:
        if random.random() > 0.6:
            continue
        insp_id += 1
        inspector = random.choice(by_plant_role[line_plant[run_line[r[0]]]]["inspector"])
        sample = random.choice([20, 30, 50, 80, 100])
        failed = min(sample, int(sample * random.uniform(0.0, 0.12)))
        inspected = datetime.strptime(r[5], "%Y-%m-%d %H:%M:%S") + timedelta(hours=random.randint(1, 12))
        inspections.append((insp_id, r[0], inspector, iso_dt(inspected), sample, failed))
    cur.executemany(
        "INSERT INTO inspections (inspection_id, run_id, inspector_id, "
        "inspected_at, sample_size, units_failed) VALUES (?,?,?,?,?,?)", inspections)

    defects, did = [], 0
    for insp in inspections:
        remaining = insp[5]
        while remaining > 0:
            did += 1
            qty = max(1, min(remaining, random.randint(1, 4)))
            remaining -= qty
            cat = random.choices(category_ids, weights=CATEGORY_WEIGHTS)[0]
            sev = DEFECT_CATEGORIES[cat - 1][1]
            if random.random() < 0.15:
                sev = random.choice(SEVERITIES)
            logged = datetime.strptime(insp[3], "%Y-%m-%d %H:%M:%S") + timedelta(minutes=random.randint(5, 90))
            defects.append((did, insp[0], cat, qty, sev, iso_dt(logged)))
    cur.executemany(
        "INSERT INTO defect_logs (defect_id, inspection_id, category_id, "
        "quantity, severity, logged_at) VALUES (?,?,?,?,?,?)", defects)

    schedules, sid = [], 0
    for m in machine_ids:
        for task in random.sample(MAINT_TASKS, 2):
            sid += 1
            interval = random.choice([30, 45, 60, 90, 120, 180])
            overdue = random.random() < 0.3
            back = random.randint(interval + 5, interval + 60) if overdue else random.randint(1, interval - 1)
            schedules.append((sid, m, task, interval, iso_d(ASOF - timedelta(days=back))))
    cur.executemany(
        "INSERT INTO maintenance_schedules (schedule_id, machine_id, task_type, "
        "interval_days, last_completed_on) VALUES (?,?,?,?,?)", schedules)

    orders = []
    for oid in range(1, 501):
        machine = random.choice(machine_ids)
        line = [l for l, ms in machines_by_line.items() if machine in ms][0]
        tech = random.choice(by_plant_role[line_plant[line]]["technician"])
        mtype = random.choices(["preventive", "corrective"], weights=[60, 40])[0]
        opened = rand_dt(360, 2)
        still_open = random.random() < 0.08
        mins = None if still_open else random.randint(20, 600)
        closed = None if still_open else iso_dt(opened + timedelta(minutes=mins))
        desc = ("Routine " + random.choice(MAINT_TASKS).lower() if mtype == "preventive"
                else random.choice(UNPLANNED_REASONS))
        orders.append((oid, machine, tech, mtype, iso_dt(opened), closed, mins, desc))
    cur.executemany(
        "INSERT INTO maintenance_orders (mo_id, machine_id, technician_id, "
        "maint_type, opened_at, closed_at, downtime_minutes, description) "
        "VALUES (?,?,?,?,?,?,?,?)", orders)

    downtime = []
    for eidx in range(1, 601):
        if random.random() < 0.55:
            src = random.choice(orders)
            machine, mo = src[1], src[0]
            cat = "planned" if src[3] == "preventive" else "unplanned"
        else:
            machine, mo = random.choice(machine_ids), None
            cat = random.choices(["planned", "unplanned"], weights=[35, 65])[0]
        reason = random.choice(PLANNED_REASONS if cat == "planned" else UNPLANNED_REASONS)
        start = rand_dt(360, 1)
        ended = None if random.random() < 0.05 else iso_dt(start + timedelta(minutes=random.randint(10, 480)))
        downtime.append((eidx, machine, mo, cat, reason, iso_dt(start), ended))
    cur.executemany(
        "INSERT INTO downtime_events (downtime_id, machine_id, mo_id, category, "
        "reason, started_at, ended_at) VALUES (?,?,?,?,?,?,?)", downtime)

    readings, rid = [], 0
    base = datetime(ASOF.year, ASOF.month, ASOF.day) - timedelta(days=30)
    for m in machine_ids:
        for h in range(30 * 24):
            ts = base + timedelta(hours=h)
            state = random.choices(["running", "idle", "down", "setup"],
                                   weights=[72, 15, 5, 8])[0]
            if state == "running":
                temp, vib = random.uniform(55, 85), random.uniform(1.5, 4.5)
                pres, pw, rpm = random.uniform(4, 8), random.uniform(12, 30), random.randint(800, 3000)
            elif state == "idle":
                temp, vib = random.uniform(30, 45), random.uniform(0.2, 0.8)
                pres, pw, rpm = random.uniform(1, 2), random.uniform(1, 3), 0
            elif state == "down":
                temp, vib = random.uniform(22, 30), random.uniform(0.0, 0.2)
                pres, pw, rpm = random.uniform(0, 0.5), random.uniform(0, 0.5), 0
            else:
                temp, vib = random.uniform(35, 50), random.uniform(0.5, 1.5)
                pres, pw, rpm = random.uniform(2, 4), random.uniform(3, 8), random.randint(0, 500)
            if m in degraded and state == "running":
                temp += random.uniform(6, 14)
                vib += random.uniform(1.5, 3.5)
            rid += 1
            readings.append((rid, m, iso_dt(ts), round(temp, 2), round(vib, 3),
                             round(pres, 2), round(pw, 2), rpm, state))
    cur.executemany(
        "INSERT INTO sensor_readings (reading_id, machine_id, reading_ts, "
        "temperature_c, vibration_mm_s, pressure_bar, power_kw, rpm, "
        "machine_state) VALUES (?,?,?,?,?,?,?,?,?)", readings)

    conn.commit()
    conn.close()
    print(f"seeded: {DB_PATH}")


if __name__ == "__main__":
    main()
