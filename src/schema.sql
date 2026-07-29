PRAGMA foreign_keys = ON;

-- ===== cross-cutting reference =====
CREATE TABLE plants (
    plant_id        INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    city            TEXT NOT NULL,
    country         TEXT NOT NULL,
    commissioned_on TEXT NOT NULL
);

CREATE TABLE shifts (
    shift_id   INTEGER PRIMARY KEY,
    name       TEXT NOT NULL CHECK (name IN ('Day','Swing','Night')),
    start_hour INTEGER NOT NULL CHECK (start_hour BETWEEN 0 AND 23),
    end_hour   INTEGER NOT NULL CHECK (end_hour BETWEEN 0 AND 23)
);

CREATE TABLE employees (
    employee_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('operator','inspector','technician','supervisor')),
    plant_id    INTEGER NOT NULL REFERENCES plants(plant_id),
    hire_date   TEXT NOT NULL
);

-- ===== production =====
CREATE TABLE production_lines (
    line_id  INTEGER PRIMARY KEY,
    plant_id INTEGER NOT NULL REFERENCES plants(plant_id),
    name     TEXT NOT NULL
);

CREATE TABLE machines (
    machine_id   INTEGER PRIMARY KEY,
    line_id      INTEGER NOT NULL REFERENCES production_lines(line_id),
    name         TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    manufacturer TEXT,
    model        TEXT,
    installed_on TEXT NOT NULL
);

CREATE TABLE products (
    product_id            INTEGER PRIMARY KEY,
    sku                   TEXT NOT NULL UNIQUE,
    name                  TEXT NOT NULL,
    category              TEXT NOT NULL,
    target_cycle_time_sec REAL NOT NULL
);

CREATE TABLE work_orders (
    work_order_id    INTEGER PRIMARY KEY,
    product_id       INTEGER NOT NULL REFERENCES products(product_id),
    line_id          INTEGER NOT NULL REFERENCES production_lines(line_id),
    quantity_ordered INTEGER NOT NULL CHECK (quantity_ordered > 0),
    status           TEXT NOT NULL CHECK (status IN ('open','in_progress','completed','cancelled')),
    created_at       TEXT NOT NULL,
    due_date         TEXT NOT NULL
);

CREATE TABLE production_runs (
    run_id         INTEGER PRIMARY KEY,
    work_order_id  INTEGER NOT NULL REFERENCES work_orders(work_order_id),
    machine_id     INTEGER NOT NULL REFERENCES machines(machine_id),
    operator_id    INTEGER NOT NULL REFERENCES employees(employee_id),
    shift_id       INTEGER NOT NULL REFERENCES shifts(shift_id),
    started_at     TEXT NOT NULL,
    ended_at       TEXT,
    units_produced INTEGER NOT NULL DEFAULT 0 CHECK (units_produced >= 0),
    units_scrapped INTEGER NOT NULL DEFAULT 0 CHECK (units_scrapped >= 0)
);

-- ===== quality =====
CREATE TABLE defect_categories (
    category_id      INTEGER PRIMARY KEY,
    name             TEXT NOT NULL UNIQUE,
    default_severity TEXT NOT NULL CHECK (default_severity IN ('low','medium','high','critical'))
);

CREATE TABLE inspections (
    inspection_id INTEGER PRIMARY KEY,
    run_id        INTEGER NOT NULL REFERENCES production_runs(run_id),
    inspector_id  INTEGER NOT NULL REFERENCES employees(employee_id),
    inspected_at  TEXT NOT NULL,
    sample_size   INTEGER NOT NULL CHECK (sample_size > 0),
    units_failed  INTEGER NOT NULL DEFAULT 0 CHECK (units_failed >= 0)
);

CREATE TABLE defect_logs (
    defect_id     INTEGER PRIMARY KEY,
    inspection_id INTEGER NOT NULL REFERENCES inspections(inspection_id),
    category_id   INTEGER NOT NULL REFERENCES defect_categories(category_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    severity      TEXT NOT NULL CHECK (severity IN ('low','medium','high','critical')),
    logged_at     TEXT NOT NULL
);

-- ===== maintenance =====
CREATE TABLE maintenance_schedules (
    schedule_id       INTEGER PRIMARY KEY,
    machine_id        INTEGER NOT NULL REFERENCES machines(machine_id),
    task_type         TEXT NOT NULL,
    interval_days     INTEGER NOT NULL CHECK (interval_days > 0),
    last_completed_on TEXT
);

CREATE TABLE maintenance_orders (
    mo_id            INTEGER PRIMARY KEY,
    machine_id       INTEGER NOT NULL REFERENCES machines(machine_id),
    technician_id    INTEGER NOT NULL REFERENCES employees(employee_id),
    maint_type       TEXT NOT NULL CHECK (maint_type IN ('preventive','corrective')),
    opened_at        TEXT NOT NULL,
    closed_at        TEXT,
    downtime_minutes INTEGER CHECK (downtime_minutes >= 0),
    description      TEXT
);

CREATE TABLE downtime_events (
    downtime_id INTEGER PRIMARY KEY,
    machine_id  INTEGER NOT NULL REFERENCES machines(machine_id),
    mo_id       INTEGER REFERENCES maintenance_orders(mo_id),
    category    TEXT NOT NULL CHECK (category IN ('planned','unplanned')),
    reason      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT
);

-- ===== sensors =====
CREATE TABLE sensor_readings (
    reading_id     INTEGER PRIMARY KEY,
    machine_id     INTEGER NOT NULL REFERENCES machines(machine_id),
    reading_ts     TEXT NOT NULL,
    temperature_c  REAL,
    vibration_mm_s REAL,
    pressure_bar   REAL,
    power_kw       REAL,
    rpm            INTEGER,
    machine_state  TEXT CHECK (machine_state IN ('running','idle','down','setup'))
);

-- ===== indexes =====
CREATE INDEX idx_lines_plant        ON production_lines(plant_id);
CREATE INDEX idx_machines_line      ON machines(line_id);
CREATE INDEX idx_runs_machine       ON production_runs(machine_id);
CREATE INDEX idx_runs_workorder     ON production_runs(work_order_id);
CREATE INDEX idx_runs_started       ON production_runs(started_at);
CREATE INDEX idx_inspections_run    ON inspections(run_id);
CREATE INDEX idx_defects_inspection ON defect_logs(inspection_id);
CREATE INDEX idx_defects_category   ON defect_logs(category_id);
CREATE INDEX idx_mo_machine         ON maintenance_orders(machine_id);
CREATE INDEX idx_downtime_machine   ON downtime_events(machine_id);
CREATE INDEX idx_sensor_machine_ts  ON sensor_readings(machine_id, reading_ts);
