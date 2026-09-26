import ast
import datetime

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

# Initialize the FastMCP server instance
mcp = FastMCP("Mixed Tools")

def _safe_formula_eval(expression: str, variables: dict[str, float] | None = None) -> float:
    variables = variables or {}

    def eval_node(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable: {node.id}")
            value = variables[node.id]
            if not isinstance(value, (int, float)):
                raise ValueError(f"Variable '{node.id}' must be numeric.")
            return float(value)

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = eval_node(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value

        if isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return left % right

        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    parsed = ast.parse(expression, mode="eval")
    return float(eval_node(parsed.body))

@mcp.tool()
def sqlite_lookup(
    db_path: str,
    query: str,
    params: list[Any] | None = None,
    limit: int = 20,
) -> list[dict[Any, Any] | dict[str, Any] | dict[str, str] | dict[bytes, bytes]]:
    """Execute a read-only SQLite SELECT query and return rows as dictionaries."""
    sql = query.strip()
    if not sql.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    if limit <= 0:
        raise ValueError("Limit must be greater than 0.")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params or [])
        rows = cursor.fetchmany(limit)

    return [dict(row) for row in rows]


@mcp.tool()
def formula_engine(expression: str, variables: dict[str, float] | None = None) -> float:
    """Evaluate a safe arithmetic formula using named variables."""
    if not expression or not expression.strip():
        raise ValueError("Expression is required.")

    return _safe_formula_eval(expression, variables)


@mcp.tool()
def log_audit_event(
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
    log_file: str = "audit.log",
) -> str:
    """Append an audit event to a local text file/log resource."""
    log_path = Path(log_file).expanduser()
    if not log_path.is_absolute():
        log_path = Path(__file__).resolve().parent / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "message": message,
        "details": details or {},
    }

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return f"Event logged to {log_path}"

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)