import argparse
import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from openai import OpenAI

load_dotenv()


SYSTEM_PROMPT = """You are a ReAct agent.

You operate in loops of:
1. Thought   – brief reasoning about what is needed next.
2. Action    – calling the appropriate tool with suitable arguments.
3. Observation – evaluating the tool result.
4. Repeat until you can answer the task.

Rules:
- Never invent tool results.
- If you need data or side effects, call the relevant tool.
- If you have enough information, provide a brief, clear final answer.
- Respond in the user's language.
"""

class ReActAgent:
    def __init__(
        self,
        mcp_url: str | None = None,
        model: str | None = None,
        max_steps: int = 12,
        verbose: bool = True,
    ) -> None:
        self.mcp_url = mcp_url or os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")
        self.model = model or os.getenv("LITELLM_MODEL")
        if not self.model:
            raise RuntimeError("LITELLM_MODEL is not set (check .env).")

        self.max_steps = max_steps
        self.verbose = verbose

        self.llm = OpenAI(
            base_url=os.getenv("LITELLM_BASE_URL"),
            api_key=os.getenv("LITELLM_API_KEY"),
        )

    def _log(self, *args: Any) -> None:
        if self.verbose:
            print(*args)

    async def run(self, task: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        async with Client(self.mcp_url) as mcp:
            mcp_tools = await mcp.list_tools()
            openai_tools = [_to_openai_tool(t) for t in mcp_tools]

            self._log(
                f"[setup] MCP @ {self.mcp_url} — "
                f"{len(openai_tools)} Tool(s): "
                f"{[t['function']['name'] for t in openai_tools]}"
            )

            for step in range(1, self.max_steps + 1):
                self._log(f"\n--- Step {step} ---")

                completion = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=openai_tools or None,
                    tool_choice="auto" if openai_tools else None,
                    temperature=0.2,
                )
                message = completion.choices[0].message

                # Thought (optional)
                if message.content:
                    self._log(f"[thought] {message.content.strip()}")

                # Assistant-turn
                assistant_entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or "",
                }
                if message.tool_calls:
                    assistant_entry["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                messages.append(assistant_entry)

                # Finish
                if not message.tool_calls:
                    return (message.content or "").strip()

                # Execute actions
                for tc in message.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    self._log(f"[action] {name}({json.dumps(args, ensure_ascii=False)})")

                    try:
                        raw_result = await mcp.call_tool(name, args)
                        payload = _normalise_result(raw_result)
                    except Exception as exc:  # Return tool errors to the LLM
                        payload = {"error": str(exc)}

                    self._log(
                        "[observation] "
                        + json.dumps(payload, ensure_ascii=False, default=str)[:600]
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(payload, ensure_ascii=False, default=str),
                        }
                    )

            return "Max steps reached without a final answer."

def _to_openai_tool(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None) or {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }

def _normalise_result(result: Any) -> Any:
    """FastMCP CallToolResult -> plain JSON-serialisable data."""
    for attr in ("structured_content", "data"):
        value = getattr(result, attr, None)
        if value is not None and not callable(value):
            return value

    content = getattr(result, "content", None)
    if content is None:
        return str(result)

    chunks: list[Any] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is None:
            chunks.append(str(block))
            continue
        try:
            chunks.append(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            chunks.append(text)

    return chunks[0] if len(chunks) == 1 else chunks

async def run() -> None:
    parser = argparse.ArgumentParser(description="ReAct agent against an MCP server")
    parser.add_argument("task", nargs="?", help="Task for the agent")
    parser.add_argument(
        "--quiet", action="store_true", help="Only print the final answer"
    )
    args = parser.parse_args()

    agent = ReActAgent(verbose=not args.quiet)
    task = args.task or "This is a quick functionality check run."
    answer = await agent.run(task)
    print(f"\n[final] {answer}")


if __name__ == "__main__":
    asyncio.run(run())