from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if isinstance(value, str) and value else default


LITELLM_BASE_URL: str = (_env_str("LITELLM_BASE_URL", _env_str("OPENAI_BASE_URL", "http://localhost:4000/v1"))).rstrip("/")
LITELLM_URL: str = LITELLM_BASE_URL[:-3] if LITELLM_BASE_URL.endswith("/v1") else LITELLM_BASE_URL
API_KEY: str = _env_str("LITELLM_API_KEY", _env_str("OPENAI_API_KEY", ""))
ROUTER_MODEL: str = _env_str("ROUTER_MODEL", _env_str("LITELLM_MODEL", "gpt-4o-mini"))

PAPER_AGENT_NAME: str = _env_str("PAPER_AGENT_NAME", "paper_analysis_agent")
LITERATURE_AGENT_NAME: str = _env_str("LITERATURE_AGENT_NAME", "literature_synthesis_agent")
WRITING_AGENT_NAME: str = _env_str("WRITING_AGENT_NAME", "thesis_writing_agent")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "x-litellm-api-key": API_KEY,
    "accept": "application/json",
    "content-type": "application/json",
}


async def fetch_available_agents(client: httpx.AsyncClient) -> list[dict[str, Any]]:
    response = await client.get(f"{LITELLM_URL}/v1/agents", headers=HEADERS)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else payload.get("agents", [])


def _extract_text_from_a2a_result(data: dict[str, Any]) -> str:
    result = data.get("result") or {}

    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected A2A result type: {type(result).__name__}"
        )

    candidates: list[dict[str, Any]] = []

    # Direct result/message
    message = result.get("message")
    if isinstance(message, dict):
        parts = message.get("parts") or []
        if isinstance(parts, list):
            candidates.extend(
                part for part in parts if isinstance(part, dict)
            )

    # Task
    task = result.get("task")
    if isinstance(task, dict):
        # Task status message
        status = task.get("status") or {}
        if isinstance(status, dict):
            status_message = status.get("message")
            if isinstance(status_message, dict):
                parts = status_message.get("parts") or []
                if isinstance(parts, list):
                    candidates.extend(
                        part for part in parts if isinstance(part, dict)
                    )

        # Task artifacts
        artifacts = task.get("artifacts") or []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue

                parts = artifact.get("parts") or []
                if isinstance(parts, list):
                    candidates.extend(
                        part for part in parts if isinstance(part, dict)
                    )

    # Newest part first
    for part in reversed(candidates):
        kind = part.get("kind")

        # Plain text
        if kind == "text" and part.get("text") is not None:
            text = str(part["text"]).strip()
            if text:
                return text

        # Structured output from Pydantic AI
        if kind == "data" and part.get("data") is not None:
            return json.dumps(
                part["data"],
                ensure_ascii=False,
            )

        # Some implementations may expose content directly
        if part.get("content") is not None:
            content = part["content"]

            if isinstance(content, str):
                content = content.strip()
                if content:
                    return content

            if content:
                return json.dumps(
                    content,
                    ensure_ascii=False,
                )

        # Fallback: data without an explicit kind
        if part.get("data") is not None:
            return json.dumps(
                part["data"],
                ensure_ascii=False,
            )

    raise RuntimeError(
        "No usable content found in A2A response:\n"
        + json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )[:8000]
    )

async def call_agent_via_a2a(
    client: httpx.AsyncClient,
    agent_id: str,
    prompt: str,
) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": str(uuid4()),
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": prompt,
                    }
                ],
            },
            "configuration": {
                "blocking": True,
                "acceptedOutputModes": ["text/plain"],
            },
        },
    }

    response = await client.post(
        f"{LITELLM_URL}/a2a/{agent_id}",
        headers=HEADERS,
        json=payload,
    )

    body = response.json()

    if response.is_error:
        raise RuntimeError(f"A2A call failed: {body}")

    response.raise_for_status()

    print("\n=== FULL A2A RESPONSE ===")
    print(json.dumps(body, ensure_ascii=False, indent=2))
    print("=========================\n")

    return _extract_text_from_a2a_result(body)

async def route_query_to_agent(
    client: httpx.AsyncClient,
    query: str,
    agents: list[dict[str, Any]],
) -> str:
    manifest = [
        {
            "agent_id": a.get("agent_id"),
            "name": a.get("agent_name") or a.get("name"),
            "description": a.get("agent_card_params", {}).get(
                "description", a.get("description", "No description")
            ),
        }
        for a in agents
    ]
    response = await client.post(
        f"{LITELLM_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": ROUTER_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an intent router. Return JSON with selected_agent_id, "
                        "choosing exactly one agent from this list:\n"
                        f"{json.dumps(manifest, ensure_ascii=False)}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "agent_selection",
                    "schema": {
                        "type": "object",
                        "properties": {"selected_agent_id": {"type": "string"}},
                        "required": ["selected_agent_id"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        },
    )
    response.raise_for_status()
    selected = json.loads(response.json()["choices"][0]["message"]["content"])["selected_agent_id"]
    ids = {a.get("agent_id") for a in agents}
    if selected not in ids:
        raise RuntimeError(f"Router returned unknown agent_id: {selected}")
    return selected


def resolve_agent_ids(agents: list[dict[str, Any]]) -> dict[str, str]:
    required: dict[str, str] = {
        "paper": PAPER_AGENT_NAME,
        "literature": LITERATURE_AGENT_NAME,
        "writing": WRITING_AGENT_NAME,
    }
    resolved: dict[str, str] = {}
    for role, target_name in required.items():
        matched_id: str | None = None
        for agent in agents:
            name = agent.get("agent_name") or agent.get("name")
            agent_id = agent.get("agent_id")
            if isinstance(name, str) and isinstance(agent_id, str) and name == target_name:
                matched_id = agent_id
                break
        if matched_id is None:
            raise RuntimeError(f"Agent named '{target_name}' not found in LiteLLM registry.")
        resolved[role] = matched_id
    return resolved

async def main() -> None:
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
    research_focus = (
        "How can generative AI agents support literature-review quality in a master thesis "
        "without reducing source traceability?"
    )
    paper_texts = [
        """
        Smith, J. and Lee, A. (2023). Human-AI Collaboration in Knowledge Work.
        We study how AI support changes revision behavior in knowledge tasks.
        Mixed methods with 42 participants: activity logs plus interviews.
        Faster first drafts were observed; final quality depended on verification discipline.
        Limitation: small convenience sample.
        """,
        """
        Kumar, R. et al. (2024). Generative AI and Academic Writing Practices.
        A cross-sectional survey with 186 students assessed AI usage in literature work.
        Frequent use for summarization and restructuring was reported.
        Major concern: missing citation verification and weak source traceability.
        """,
    ]

    async with httpx.AsyncClient(timeout=timeout) as client:
        agents = await fetch_available_agents(client)
        ids = resolve_agent_ids(agents)

        selected = await route_query_to_agent(
            client, "Analyze this paper and extract structured findings.", agents
        )
        print(f"Router selected for paper-analysis intent: {selected}")

        paper_analyses: list[dict[str, Any]] = []
        for idx, paper_text in enumerate(paper_texts, start=1):
            prompt = (
                f"Analyze paper #{idx}. Return only the structured object.\n\n"
                f"RAW PAPER TEXT:\n{paper_text}"
            )
            raw = await call_agent_via_a2a(client, ids["paper"], prompt)
            print("\n=== RAW PAPER AGENT RESULT ===")
            print(repr(raw))
            print("================================")
            paper_analyses.append(json.loads(raw))

        synthesis_prompt = (
            "Create a literature synthesis from these analyses.\n\n"
            f"RESEARCH FOCUS:\n{research_focus}\n\n"
            f"PAPER_ANALYSES_JSON:\n{json.dumps(paper_analyses, ensure_ascii=False)}"
        )
        synthesis_raw = await call_agent_via_a2a(client, ids["literature"], synthesis_prompt)
        synthesis = json.loads(synthesis_raw)

        writing_prompt = (
            "Draft a concise literature-review section for a thesis from this synthesis.\n\n"
            f"SYNTHESIS_JSON:\n{json.dumps(synthesis, ensure_ascii=False)}"
        )
        writing_raw = await call_agent_via_a2a(client, ids["writing"], writing_prompt)
        writing = json.loads(writing_raw)

    print("\n=== Final Writing Support ===")
    print(json.dumps(writing, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
