from __future__ import annotations

import json

from fasta2a.pydantic_ai import agent_to_a2a
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.common import build_model

model = build_model()


class WritingSupport(BaseModel):
    section_purpose: str
    suggested_structure: list[str] = Field(default_factory=list)
    draft_text: str
    citation_placeholders: list[str] = Field(default_factory=list)
    claims_requiring_verification: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)


agent = Agent(
    model=model,
    name="thesis_writing_agent",
    description=(
        "Turns a literature synthesis into a draft thesis section "
        "with explicit citation placeholders and uncertainty markers."
    ),
    output_type=str,
    system_prompt=(
        "You are the thesis writing step in a literature-review pipeline.\n\n"
        "You receive a literature synthesis as JSON. "
        "Use ONLY the information contained in that synthesis. "
        "Do not introduce external evidence, citations, statistics, "
        "authors, or claims.\n\n"
        "Return exactly one valid JSON object with these fields:\n"
        "- section_purpose: string\n"
        "- suggested_structure: array of strings\n"
        "- draft_text: string\n"
        "- citation_placeholders: array of strings\n"
        "- claims_requiring_verification: array of strings\n"
        "- missing_evidence: array of strings\n\n"
        "Use placeholders such as [CITATION: AuthorYear] when a citation "
        "is required but the available input does not provide enough "
        "information to construct one.\n\n"
        "Do not fabricate references or bibliography entries.\n"
        "Do not use Markdown code fences around the JSON.\n"
        "Do not add explanations before or after the JSON object."
    ),
)


@agent.tool_plain
def validate_synthesis(
    synthesis_json: str,
) -> dict[str, object]:
    """Validate the required fields of a literature synthesis."""
    try:
        data = json.loads(synthesis_json)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "error": f"Invalid JSON: {exc}",
        }

    if not isinstance(data, dict):
        return {
            "valid": False,
            "error": "Expected a JSON object.",
        }

    required = {
        "research_focus",
        "themes",
        "converging_findings",
        "research_gaps",
    }

    missing = sorted(required - set(data.keys()))

    return {
        "valid": not missing,
        "missing_keys": missing,
    }


app = agent_to_a2a(
    agent,
    name="thesis_writing_agent",
    url="http://host.docker.internal:8023",
    version="1.0.0",
    description=(
        "Turns a literature synthesis into a draft thesis section."
    ),
)


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8023")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )