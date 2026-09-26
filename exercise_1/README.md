# Mixed Tools MCP — ReAct Agent Demo

This is a minimal example project showing how a FastMCP server with three tools
(SQLite lookup, formula engine, audit log) is called from a ReAct agent via a
LiteLLM-compatible endpoint.

```
.
├── server.py              # FastMCP server ("Mixed Tools")
├── agent.py               # ReAct loop / MCP client
├── example_execution.py   # End-to-end demo for the exercise
├── README.md
└── .env                   # Local environment variables (created manually)
```

## 1. Requirements

- Python 3.13+
- A running LiteLLM proxy or any OpenAI-compatible endpoint
- Free port 8000 for the MCP server

## 2. Setup with uv only

This repository already includes a root `pyproject.toml` with the required Python dependencies.
Use uv for the environment and execution.

```bash
cd Industrial_Computing_Submissions
uv sync
```

Create a `.env` file in `exercise_1/` with the following values:

```env
LITELLM_BASE_URL=http://127.0.0.1:4000
LITELLM_API_KEY=sk-local
LITELLM_MODEL=gpt-4o-mini
MCP_URL=http://127.0.0.1:8000/mcp
```

You can place the file directly in the `exercise_1` directory so that `python-dotenv` loads it
when the scripts start.

## 3. Start the MCP server

Open Terminal A and run:

```bash
cd Industrial_Computing_Submissions/exercise_1
uv run python server.py
```

The server listens on `http://0.0.0.0:8000` by default. The MCP endpoint is:
`http://127.0.0.1:8000/mcp`.

## 4. Run the demo / probe

Open Terminal B and run:

```bash
cd Industrial_Computing_Submissions/exercise_1
uv run python example_execution.py
```

The script does the following:

1. Creates a fresh `demo.db` with 10 customers and 14 orders
2. Runs three ReAct tasks (SQLite, formula, audit log)
3. Prints the final contents of `demo_audit.log`

Typical output looks like this:

```text
--- Step 1 ---
[thought] I need to find all customers in Germany...
[action] sqlite_lookup({"db_path": "/.../demo.db",
                      "query": "SELECT name, lifetime_value FROM customers WHERE country='Germany'"})
[observation] [{"name": "Alice Müller", "lifetime_value": 1250.0}, ...]
...
[final] There are 3 customers in Germany: Alice Müller (1250.0), ...
```

## 5. Try your own task

```bash
cd Industrial_Computing_Submissions/exercise_1

# Single task from the CLI
uv run python agent.py "Use formula_engine to calculate (a*b) - c with a=7, b=6, c=11."

# Quiet mode (final answer only)
uv run python agent.py --quiet "What is the total revenue of all 'paid' orders?"

# In Python
uv run python - <<'PY'
import asyncio
from agent import ReActAgent

async def main():
   agent = ReActAgent(verbose=True)
   print(await agent.run("Log an audit event of type 'demo'."))

asyncio.run(main())
PY
```

## 6. Architecture overview

```text
 ┌─────────────┐    OpenAI API    ┌───────────────┐
 │  agent.py   │ ───────────────► │  LiteLLM      │
 │  ReAct loop │                  │  proxy        │
 └─────┬───────┘                  └───────────────┘
      │ MCP (HTTP)
      ▼
 ┌─────────────┐
 │  server.py  │ tools: sqlite_lookup, formula_engine, log_audit_event
 │  FastMCP    │
 └─────────────┘
```

- The agent fetches the tool schemas dynamically from the MCP server (`list_tools`) and maps them to OpenAI `tools=[...]`.
- The LLM decides itself when to call a tool (`tool_choice="auto"`).
- The loop ends as soon as the model no longer wants to call tools or `max_steps` is reached.

## 7. Troubleshooting

- `ConnectionError` to `http://127.0.0.1:8000/mcp`
 - The MCP server is not running; check Terminal A.

- `RuntimeError: LITELLM_MODEL is not set`
 - `.env` is missing or values are empty.

- `AuthenticationError` from the LLM
 - Check `LITELLM_API_KEY`; for a local LiteLLM proxy it is often a placeholder but should not be empty.

- Tool output is always a `str`
 - Update FastMCP to a recent version via `uv add fastmcp>=4.0.5` or re-run `uv sync`.

- `ValueError: Unknown variable: ...` in `formula_engine`
 - The model forgot to provide the variable mapping; use explicit values such as `a=..., b=..., c=...` in the task.
