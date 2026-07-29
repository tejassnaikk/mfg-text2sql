"""Experiment-2 expansion templates (coverage arm).

Twelve templates covering query families absent from Exp-1 training:
ASC ordering, multi-column projection, under-covered join paths
(esp. the products path, which had zero training examples), and
ASC ('lowest'/'fewest') superlatives. Authored blind to the eight
test-template SQL bodies; validate_expansion.py enforces no collision.
All ids start with 'x_' so the validator/split pick them up.
"""

from templates import template


# ---------- ASC ordering + multi-column projection (tier 1) ----------

@template("x_machines_by_install_date", 1, ["order", "asc", "multicol"])
def _(v, r):
    sql = ("SELECT name, machine_type, installed_on FROM machines "
           "ORDER BY installed_on ASC;")
    return (["List machines from oldest to newest by install date.",
             "Show every machine ordered by installation date, earliest first.",
             "Which machines were installed longest ago? Order by install date ascending."], sql)


@template("x_plants_by_commission_date", 1, ["order", "asc", "multicol"])
def _(v, r):
    sql = ("SELECT name, city, commissioned_on FROM plants "
           "ORDER BY commissioned_on ASC;")
    return (["List plants from oldest to newest by commission date.",
             "Show plants ordered by when they were commissioned, earliest first.",
             "Which plants were commissioned earliest? Order ascending by commission date."], sql)


@template("x_employees_by_hire_date_asc", 1, ["order", "asc", "filter"])
def _(v, r):
    role = r.choice(["operator", "inspector", "technician"])
    sql = (f"SELECT name, role, hire_date FROM employees WHERE role = '{role}' "
           "ORDER BY hire_date ASC;")
    return ([f"List {role}s ordered by hire date, earliest first.",
             f"Show {role}s sorted by when they were hired, oldest first.",
             f"Which {role}s have been here longest? Order by hire date ascending."], sql)


# ---------- multi-column projection + under-covered join paths (tier 2) ----------

@template("x_workorders_detail_by_status", 2, ["join", "filter", "multicol"])
def _(v, r):
    st = r.choice(["open", "completed"])
    label = st.replace("_", " ")
    sql = ("SELECT w.work_order_id, p.name, w.quantity_ordered, w.status "
           "FROM work_orders w JOIN products p ON p.product_id = w.product_id "
           f"WHERE w.status = '{st}' ORDER BY w.work_order_id;")
    return ([f"Show {label} work orders with their product and quantity.",
             f"List {label} work orders including product name and quantity ordered.",
             f"For {label} work orders, give the product and quantity ordered."], sql)


@template("x_machine_inventory_by_type", 2, ["join", "filter", "multicol"])
def _(v, r):
    mt = r.choice(v["machine_type"])
    sql = ("SELECT m.name, pl.name, l.name "
           "FROM machines m JOIN production_lines l ON l.line_id = m.line_id "
           "JOIN plants pl ON pl.plant_id = l.plant_id "
           f"WHERE m.machine_type = '{mt}' ORDER BY m.name;")
    return ([f"List all {mt} machines with their line and plant.",
             f"Show {mt} machines and where they sit (line and plant).",
             f"Which {mt} machines exist, and on which line and plant?"], sql)


@template("x_maintenance_order_detail", 2, ["join", "filter", "multicol"])
def _(v, r):
    mt = r.choice(["preventive", "corrective"])
    sql = ("SELECT m.name, e.name, o.maint_type, o.downtime_minutes "
           "FROM maintenance_orders o JOIN machines m ON m.machine_id = o.machine_id "
           "JOIN employees e ON e.employee_id = o.technician_id "
           f"WHERE o.maint_type = '{mt}' ORDER BY o.mo_id;")
    return ([f"Show {mt} maintenance orders with machine, technician, and downtime.",
             f"List {mt} maintenance orders including the machine and technician.",
             f"For {mt} maintenance, give machine, technician, and downtime minutes."], sql)


# ---------- aggregates over under-covered join paths (tier 3) ----------

@template("x_output_by_product", 3, ["join", "aggregate", "group"])
def _(v, r):
    sql = ("SELECT p.name, SUM(r.units_produced) AS produced "
           "FROM production_runs r JOIN work_orders w ON w.work_order_id = r.work_order_id "
           "JOIN products p ON p.product_id = w.product_id "
           "GROUP BY p.name ORDER BY produced DESC;")
    return (["What is the total units produced for each product?",
             "Show output totals broken down by product.",
             "Sum units produced per product, highest first."], sql)


@template("x_output_by_line", 3, ["join", "aggregate", "group"])
def _(v, r):
    sql = ("SELECT l.name, SUM(r.units_produced) AS produced "
           "FROM production_runs r JOIN machines m ON m.machine_id = r.machine_id "
           "JOIN production_lines l ON l.line_id = m.line_id "
           "GROUP BY l.name ORDER BY produced DESC;")
    return (["What is the total units produced on each production line?",
             "Show output totals by production line.",
             "Sum units produced per line, highest first."], sql)


@template("x_planned_downtime_by_machine", 3, ["join", "aggregate", "group", "filter"])
def _(v, r):
    sql = ("SELECT m.name, COUNT(*) AS events "
           "FROM downtime_events de JOIN machines m ON m.machine_id = de.machine_id "
           "WHERE de.category = 'planned' "
           "GROUP BY m.name ORDER BY events DESC;")
    return (["How many planned downtime events did each machine have?",
             "Count planned downtime events per machine.",
             "Show the number of planned downtime events by machine."], sql)


# ---------- ASC superlatives (tier 4) ----------

@template("x_lowest_scrap_operator", 4, ["superlative", "join", "aggregate", "asc"])
def _(v, r):
    sh = r.choice(v["shift"])
    sql = ("SELECT e.name, "
           "ROUND(100.0 * SUM(r.units_scrapped) / SUM(r.units_produced), 2) AS scrap_pct "
           "FROM production_runs r "
           "JOIN employees e ON e.employee_id = r.operator_id "
           "JOIN shifts s ON s.shift_id = r.shift_id "
           f"WHERE s.name = '{sh}' "
           "GROUP BY e.name HAVING SUM(r.units_produced) > 0 "
           "ORDER BY scrap_pct ASC LIMIT 1;")
    return ([f"Which {sh} shift operator has the lowest scrap rate?",
             f"Find the {sh} shift operator with the best (lowest) scrap percentage.",
             f"Who scraps the fewest units on the {sh} shift, proportionally?"], sql)


@template("x_fewest_machines_line", 4, ["superlative", "join", "count", "asc"])
def _(v, r):
    sql = ("SELECT l.name, COUNT(*) AS n "
           "FROM machines m JOIN production_lines l ON l.line_id = m.line_id "
           "GROUP BY l.name ORDER BY n ASC LIMIT 1;")
    return (["Which production line has the fewest machines?",
             "Find the line with the smallest number of machines.",
             "What line has the least machines assigned to it?"], sql)


@template("x_lowest_output_plant", 4, ["superlative", "join", "aggregate", "asc"])
def _(v, r):
    sql = ("SELECT pl.name, SUM(r.units_produced) AS produced "
           "FROM production_runs r JOIN machines m ON m.machine_id = r.machine_id "
           "JOIN production_lines l ON l.line_id = m.line_id "
           "JOIN plants pl ON pl.plant_id = l.plant_id "
           "GROUP BY pl.name ORDER BY produced ASC LIMIT 1;")
    return (["Which plant produced the fewest units overall?",
             "Find the plant with the lowest total output.",
             "What manufacturing site has the smallest total units produced?"], sql)
