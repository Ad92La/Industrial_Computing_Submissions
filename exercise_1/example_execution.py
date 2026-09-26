"""
End-to-end demosntration for the exercise.

Prequesition: The mcp server is running see --> README.

Execution-order:
    1. Dummy sqlite database with data is created
    2. The mcp tools are used
    3. Audit log shown at the end
"""
import asyncio
import sqlite3
from pathlib import Path

from exercise_1.agent import ReActAgent

BASE_PATH = Path(__file__).resolve().parent
DB_PATH = BASE_PATH / "demo.db"
LOG_PATH = BASE_PATH / "demo_audit.log"

CUSTOMERS = [
    (1, "Alice Müller",   "Germany", "2023-01-15", 1250.00),
    (2, "Bob Smith",      "USA",     "2023-02-20",  890.50),
    (3, "Clara Schmidt",  "Germany", "2023-03-05", 4200.75),
    (4, "Diego García",   "Spain",   "2023-04-11",  310.20),
    (5, "Erika Tanaka",   "Japan",   "2023-05-02", 2150.00),
    (6, "Farid Haddad",   "Germany", "2023-06-18",  780.40),
    (7, "Grace Chen",     "Singapore","2023-07-22", 3450.00),
    (8, "Hiroshi Sato",   "Japan",   "2023-08-09", 1560.90),
    (9, "Ivana Novak",    "Czechia", "2023-09-14",  675.00),
    (10, "Jonas Berg",    "Sweden",  "2023-10-30", 1980.60),
]

ORDERS = [
    (101,  1,  120.00, "paid",      "2024-01-02"),
    (102,  1,  250.00, "paid",      "2024-02-11"),
    (103,  2,   89.90, "pending",   "2024-02-15"),
    (104,  3,  500.00, "paid",      "2024-03-01"),
    (105,  3,  700.00, "paid",      "2024-03-22"),
    (106,  3, 1000.00, "refunded",  "2024-04-05"),
    (107,  4,   45.00, "paid",      "2024-04-18"),
    (108,  5,  320.00, "paid",      "2024-05-07"),
    (109,  6,   99.99, "pending",   "2024-05-20"),
    (110,  7,  410.00, "paid",      "2024-06-01"),
    (111,  7,  630.00, "paid",      "2024-06-15"),
    (112,  8,  220.00, "paid",      "2024-07-09"),
    (113,  9,   75.00, "cancelled", "2024-08-02"),
    (114, 10,  340.00, "paid",      "2024-09-19"),
]

def create_demo_db(path: Path) -> None:
    if path.exists():
        path.unlink()

    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE customers (
                id              INTEGER PRIMARY KEY,
                name            TEXT    NOT NULL,
                country         TEXT    NOT NULL,
                signup_date     TEXT    NOT NULL,
                lifetime_value  REAL    NOT NULL
            );

            CREATE TABLE orders (
                id           INTEGER PRIMARY KEY,
                customer_id  INTEGER NOT NULL REFERENCES customers(id),
                amount       REAL    NOT NULL,
                status       TEXT    NOT NULL,
                created_at   TEXT    NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO customers (id, name, country, signup_date, lifetime_value) "
            "VALUES (?, ?, ?, ?, ?)",
            CUSTOMERS,
        )
        conn.executemany(
            "INSERT INTO orders (id, customer_id, amount, status, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ORDERS,
        )

    print(
        f"[setup] demo.db created: {path} "
        f"({len(CUSTOMERS)} Customer, {len(ORDERS)} Orders)"
    )

def demo_tasks() -> list[tuple[str, str]]:
    return [
        (
            "SQLite-Lookup",
            "Use the tool `sqlite_lookup`, to receive all customers from Germany. "
            f"The database is located at: {DB_PATH}. "
            "Return the names and livetime-value of these customers.",
        ),
        (
            "Formula-Engine",
            "Calculate with the tool `formula_engine` the expression "
            "`a + b + c` where a=1250.0, b=4200.75, c=780.4, "
            "and subtract 12 % discount afterwards. "
            "Call the tool as often as necessary and state the final result.",
        ),
        (
            "Audit-Log",
            "Create with `log_audit_event` a new audit entry: "
            "event_type='demo_run', message='Presentation finished', "
            "details={'audience': 'students', 'room': 'A1'}, "
            f"log_file='{LOG_PATH}'.",
        ),
    ]

async def main() -> None:
    create_demo_db(DB_PATH)
    agent = ReActAgent()

    for title, task in demo_tasks():
        print("\n" + "=" * 72)
        print(f"DEMO · {title}")
        print("=" * 72)
        print(f"[user] {task}\n")
        answer = await agent.run(task)
        print(f"\n[final] {answer}")

    print("\n" + "=" * 72)
    print("AUDIT-LOG OUTPUT")
    print("=" * 72)
    if LOG_PATH.exists():
        print(LOG_PATH.read_text(encoding="utf-8"))
    else:
        print(f"(no logs found at location {LOG_PATH})")


if __name__ == "__main__":
    asyncio.run(main())