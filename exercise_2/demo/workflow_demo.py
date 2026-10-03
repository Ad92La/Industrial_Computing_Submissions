from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import httpx
from dotenv import load_dotenv


# ============================================================
# Environment
# ============================================================

load_dotenv()


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if isinstance(value, str) and value else default


LITELLM_BASE_URL = _env_str(
    "LITELLM_BASE_URL",
    _env_str(
        "OPENAI_BASE_URL",
        "http://localhost:4000/v1",
    ),
).rstrip("/")


LITELLM_URL = (
    LITELLM_BASE_URL[:-3]
    if LITELLM_BASE_URL.endswith("/v1")
    else LITELLM_BASE_URL
)


API_KEY = _env_str(
    "LITELLM_API_KEY",
    _env_str("OPENAI_API_KEY", ""),
)


PAPER_AGENT_NAME = _env_str(
    "PAPER_AGENT_NAME",
    "paper_analysis_agent",
)


LITERATURE_AGENT_NAME = _env_str(
    "LITERATURE_AGENT_NAME",
    "literature_synthesis_agent",
)


WRITING_AGENT_NAME = _env_str(
    "WRITING_AGENT_NAME",
    "thesis_writing_agent",
)


HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "x-litellm-api-key": API_KEY,
    "accept": "application/json",
    "content-type": "application/json",
}


# ============================================================
# Direct A2A configuration
# ============================================================
#
# LiteLLM läuft im Docker-Container.
#
# LiteLLM -> Agents:
#   http://host.docker.internal:8021
#   http://host.docker.internal:8022
#   http://host.docker.internal:8023
#
# workflow_demo.py läuft direkt auf Windows.
#
# workflow -> Agents:
#   http://localhost:8021
#   http://localhost:8022
#   http://localhost:8023
#
# Die URLs in den Agent Cards werden NICHT verändert.
# ============================================================

DIRECT_AGENT_URLS = {
    PAPER_AGENT_NAME: "http://localhost:8021",
    LITERATURE_AGENT_NAME: "http://localhost:8022",
    WRITING_AGENT_NAME: "http://localhost:8023",
}


# ============================================================
# LiteLLM: Agent discovery
# ============================================================

async def fetch_available_agents(
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """
    Liest die registrierten Agents aus LiteLLM.
    """

    response = await client.get(
        f"{LITELLM_URL}/v1/agents",
        headers=HEADERS,
        timeout=30.0,
    )

    response.raise_for_status()

    payload = response.json()

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        agents = payload.get("agents")

        if isinstance(agents, list):
            return agents

    raise RuntimeError(
        "Unerwartetes Format von /v1/agents:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


# ============================================================
# Agent resolution
# ============================================================

def resolve_agents(
    agents: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """
    Ordnet die drei benötigten Rollen ihren LiteLLM-Agent-IDs
    und ihren direkten Windows-A2A-URLs zu.
    """

    by_name: dict[str, dict[str, Any]] = {}

    for agent in agents:
        name = agent.get("agent_name")

        if isinstance(name, str) and name:
            by_name[name] = agent

    required = {
        "paper": PAPER_AGENT_NAME,
        "literature": LITERATURE_AGENT_NAME,
        "writing": WRITING_AGENT_NAME,
    }

    resolved: dict[str, dict[str, str]] = {}

    for role, name in required.items():

        agent = by_name.get(name)

        if agent is None:
            available = ", ".join(sorted(by_name.keys()))

            raise RuntimeError(
                f"Agent '{name}' wurde in LiteLLM nicht gefunden.\n"
                f"Verfügbare Agents: {available}"
            )

        agent_id = agent.get("agent_id")

        if not isinstance(agent_id, str) or not agent_id:
            raise RuntimeError(
                f"Agent '{name}' besitzt keine gültige agent_id."
            )

        direct_url = DIRECT_AGENT_URLS.get(name)

        if direct_url is None:
            raise RuntimeError(
                f"Keine direkte A2A-URL für Agent '{name}' konfiguriert."
            )

        resolved[role] = {
            "agent_id": agent_id,
            "name": name,
            "url": direct_url,
        }

    return resolved


# ============================================================
# A2A ID helpers
# ============================================================

def _new_id() -> str:
    return str(uuid.uuid4())


# ============================================================
# A2A response helpers
# ============================================================

def _extract_task_id(
    data: dict[str, Any],
) -> str | None:
    """
    Extrahiert die Task-ID aus verschiedenen möglichen
    A2A message/send Antwortformaten.
    """

    result = data.get("result")

    if not isinstance(result, dict):
        return None

    task = result.get("task")

    if isinstance(task, dict):
        task_id = task.get("id")

        if isinstance(task_id, str) and task_id:
            return task_id

    direct_id = result.get("id")

    if isinstance(direct_id, str) and direct_id:
        return direct_id

    return None


def _extract_task_state(
    data: dict[str, Any],
) -> str | None:
    """
    Liest den Status einer A2A-Task.
    """

    result = data.get("result")

    if not isinstance(result, dict):
        return None

    status = result.get("status")

    if not isinstance(status, dict):
        return None

    state = status.get("state")

    if isinstance(state, str):
        return state

    return None


def _extract_text(
    data: dict[str, Any],
) -> str | None:
    """
    Extrahiert normalen Text aus einer A2A Message.
    """

    result = data.get("result")

    if not isinstance(result, dict):
        return None

    parts = result.get("parts")

    if not isinstance(parts, list):
        return None

    texts: list[str] = []

    for part in parts:

        if not isinstance(part, dict):
            continue

        if part.get("kind") != "text":
            continue

        text = part.get("text")

        if isinstance(text, str) and text.strip():
            texts.append(text.strip())

    if texts:
        return "\n".join(texts)

    return None


def _extract_artifact_result(
    data: dict[str, Any],
) -> Any:
    result = data.get("result")

    if not isinstance(result, dict):
        return None

    artifacts = result.get("artifacts")

    if not isinstance(artifacts, list):
        return None

    for artifact in artifacts:

        if not isinstance(artifact, dict):
            continue

        parts = artifact.get("parts")

        if not isinstance(parts, list):
            continue

        for part in parts:

            if not isinstance(part, dict):
                continue

            # Structured output from Pydantic AI
            data_part = part.get("data")

            if isinstance(data_part, dict):
                if "result" in data_part:
                    return data_part["result"]

            # Plain-text output from agents
            text_part = part.get("text")

            if isinstance(text_part, str) and text_part.strip():
                return text_part.strip()

    return None

# ============================================================
# Direct A2A call
# ============================================================

async def call_agent_via_a2a(
    client: httpx.AsyncClient,
    agent_url: str,
    message: str,
    *,
    timeout_seconds: float = 300.0,
    poll_interval_seconds: float = 1.0,
) -> Any:
    """
    Sendet eine Nachricht direkt an einen FastA2A-Agenten.

    Ablauf:

        message/send
             |
             v
          Task-ID
             |
             v
          tasks/get
             |
             v
        completed
             |
             v
        Artifact/result

    Der LiteLLM-A2A-Proxy wird hier absichtlich nicht verwendet.
    """

    message_id = _new_id()
    request_id = _new_id()

    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "message/send",
        "params": {
            "message": {
                "kind": "message",
                "messageId": message_id,
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": message,
                    }
                ],
            },
            "configuration": {
                "blocking": True,
                "acceptedOutputModes": [
                    "text/plain",
                ],
            },
        },
    }

    print()
    print("A2A message/send")
    print(f"  URL: {agent_url}/")

    response = await client.post(
        f"{agent_url}/",
        json=payload,
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    data = response.json()

    # --------------------------------------------------------
    # JSON-RPC Fehler
    # --------------------------------------------------------

    if "error" in data:
        raise RuntimeError(
            "A2A message/send lieferte einen Fehler:\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    # --------------------------------------------------------
    # Fall 1:
    # Agent liefert direkt eine Message
    # --------------------------------------------------------

    task_id = _extract_task_id(data)

    if task_id is None:

        text = _extract_text(data)

        if text is not None:
            return text

        artifact_result = _extract_artifact_result(data)

        if artifact_result is not None:
            return artifact_result

        raise RuntimeError(
            "A2A message/send lieferte weder eine Task-ID "
            "noch ein verwertbares Ergebnis.\n"
            f"{json.dumps(data, ensure_ascii=False, indent=2)}"
        )

    print(f"  Task-ID: {task_id}")

    # --------------------------------------------------------
    # Fall 2:
    # Task pollen
    # --------------------------------------------------------

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:

        await asyncio.sleep(poll_interval_seconds)

        get_payload = {
            "jsonrpc": "2.0",
            "id": _new_id(),
            "method": "tasks/get",
            "params": {
                "id": task_id,
            },
        }

        task_response = await client.post(
            f"{agent_url}/",
            json=get_payload,
            timeout=timeout_seconds,
        )

        task_response.raise_for_status()

        task_data = task_response.json()

        if "error" in task_data:
            raise RuntimeError(
                "A2A tasks/get lieferte einen Fehler:\n"
                f"{json.dumps(task_data, ensure_ascii=False, indent=2)}"
            )

        state = _extract_task_state(task_data)

        # print(f"  Task state: {state}")

        # ----------------------------------------------------
        # Task fertig
        # ----------------------------------------------------

        if state == "completed":

            # Structured Pydantic-AI result
            artifact_result = _extract_artifact_result(
                task_data
            )

            if artifact_result is not None:
                return artifact_result

            # Normaler Text
            text = _extract_text(task_data)

            if text is not None:
                return text

            raise RuntimeError(
                "A2A Task wurde erfolgreich abgeschlossen, "
                "enthielt aber kein verwertbares Ergebnis.\n"
                f"{json.dumps(task_data, ensure_ascii=False, indent=2)}"
            )

        # ----------------------------------------------------
        # Task fehlgeschlagen
        # ----------------------------------------------------

        if state in {
            "failed",
            "canceled",
            "rejected",
        }:
            print()
            print("A2A Task fehlgeschlagen.")
            print(
                json.dumps(
                    task_data,
                    ensure_ascii=False,
                    indent=2,
                )
            )

            # Sometimes FastA2A exposes the actual error in the history.
            result = task_data.get("result")

            if isinstance(result, dict):
                history = result.get("history")

                if isinstance(history, list):
                    for entry in history:
                        if not isinstance(entry, dict):
                            continue

                        parts = entry.get("parts")

                        if not isinstance(parts, list):
                            continue

                        for part in parts:
                            if not isinstance(part, dict):
                                continue

                            text = part.get("text")

                            if isinstance(text, str) and text.strip():
                                print()
                                print("Agent message:")
                                print(text)

            raise RuntimeError(
                f"A2A Task {task_id} endete mit Status '{state}'."
            )

    raise TimeoutError(
        f"A2A Task {task_id} wurde innerhalb von "
        f"{timeout_seconds:.0f} Sekunden nicht abgeschlossen."
    )


# ============================================================
# Main workflow
# ============================================================

async def main() -> None:

    # --------------------------------------------------------
    # Research focus
    # --------------------------------------------------------

    research_focus = (
        "How can generative AI agents support literature-review "
        "quality in a master thesis without reducing source traceability?"
    )

    # --------------------------------------------------------
    # Demo papers
    # --------------------------------------------------------

    paper_texts = [
        """
        Smith, J. and Lee, A. (2023). Human-AI Collaboration in Knowledge Work.

        We study how AI support changes revision behavior in knowledge tasks.

        Mixed methods with 42 participants: activity logs plus interviews.

        Faster first drafts were observed; final quality depended on
        verification discipline.

        Limitation: small convenience sample.
        """,

        """
        Kumar, R. et al. (2024). Generative AI and Academic Writing Practices.

        A cross-sectional survey with 186 students assessed AI usage
        in literature work.

        Frequent use for summarization and restructuring was reported.

        Major concern: missing citation verification and weak source
        traceability.
        """,
    ]

    # --------------------------------------------------------
    # HTTP client
    # --------------------------------------------------------

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=300.0,
            write=30.0,
            pool=30.0,
        )
    ) as client:

        # ====================================================
        # 1. Agent discovery
        # ====================================================

        print()
        print("=" * 60)
        print("1. AGENT DISCOVERY")
        print("=" * 60)

        agents = await fetch_available_agents(client)

        resolved = resolve_agents(agents)

        for role, info in resolved.items():
            print(
                f"{role:10s} "
                f"{info['name']:30s} "
                f"{info['agent_id']} "
                f"-> {info['url']}"
            )

        # ====================================================
        # 2. Deterministic pipeline
        # ====================================================

        print()
        print("=" * 60)
        print("2. PIPELINE")
        print("=" * 60)

        print("1. Papers -> Paper Analysis Agent")
        print("2. Analyses -> Literature Synthesis Agent")
        print("3. Synthesis -> Thesis Writing Agent")

        # ====================================================
        # 3. Paper analysis
        # ====================================================

        print()
        print("=" * 60)
        print("3. PAPER ANALYSIS")
        print("=" * 60)

        paper_analyses: list[Any] = []

        for index, paper_text in enumerate(
            paper_texts,
            start=1,
        ):

            print()
            print(f"--- Paper {index} ---")

            paper_prompt = f"""
Research focus:
{research_focus}

Analyze ONLY the research paper text provided below.

Do not use external knowledge.
Do not invent information.

Return the structured paper analysis according to
your output schema.

Paper:
{paper_text}
"""

            result = await call_agent_via_a2a(
                client,
                resolved["paper"]["url"],
                paper_prompt,
            )

            paper_analyses.append(result)

            print()
            print("Paper analysis result:")

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        # ====================================================
        # 4. Literature synthesis
        # ====================================================

        print()
        print("=" * 60)
        print("4. LITERATURE SYNTHESIS")
        print("=" * 60)

        synthesis_prompt = f"""
Research focus:

{research_focus}

The following structured paper analyses were produced
by the paper analysis agent:

{json.dumps(
    paper_analyses,
    ensure_ascii=False,
    indent=2,
)}

Synthesize these findings into a literature review.

Identify:

- major themes
- converging findings
- divergent findings
- research gaps
- candidate thesis claims
- uncertainty flags

Use ONLY the information contained in the provided
paper analyses.

Do not invent studies, citations, findings, or claims.
"""

        synthesis_result = await call_agent_via_a2a(
            client,
            resolved["literature"]["url"],
            synthesis_prompt,
        )

        print()
        print("Literature synthesis:")

        print(
            json.dumps(
                synthesis_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        # ====================================================
        # 5. Thesis writing
        # ====================================================

        print()
        print("=" * 60)
        print("5. THESIS WRITING")
        print("=" * 60)

        writing_prompt = f"""
Research focus:

{research_focus}

Use the following literature synthesis to draft
a thesis section.

Literature synthesis:

{json.dumps(
    synthesis_result,
    ensure_ascii=False,
    indent=2,
)}

Requirements:

- academic writing style
- clearly distinguish findings from interpretation
- do not invent citations
- use explicit citation placeholders where necessary
- preserve uncertainty
- stay strictly within the provided evidence
"""

        writing_result = await call_agent_via_a2a(
            client,
            resolved["writing"]["url"],
            writing_prompt,
        )

        print()
        print("Thesis writing result:")

        print(
            json.dumps(
                writing_result,
                ensure_ascii=False,
                indent=2,
            )
        )

        # ====================================================
        # 6. Final result
        # ====================================================

        final_result = {
            "research_focus": research_focus,
            "paper_analyses": paper_analyses,
            "literature_synthesis": synthesis_result,
            "thesis_draft": writing_result,
        }

        print()
        print("=" * 60)
        print("FINAL WORKFLOW RESULT")
        print("=" * 60)

        print(
            json.dumps(
                final_result,
                ensure_ascii=False,
                indent=2,
            )
        )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())

