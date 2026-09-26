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
        "Turns a literature synthesis into a draft thesis section with explicit citation placeholders."
    ),
    output_type=str,
    system_prompt=(
        "You draft a literature review section from a provided synthesis only. "
        "Never fabricate citations, numbers, or evidence. "
        "Use placeholders like [CITATION: AuthorYear] where needed."
    ),
)


@agent.tool_plain
def validate_synthesis(synthesis_json: str) -> dict[str, object]:
    try:
        data = json.loads(synthesis_json)
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": f"Invalid JSON: {exc}"}
    required = {"themes", "converging_findings", "research_gaps"}
    if not isinstance(data, dict):
        return {"valid": False, "error": "Expected a JSON object."}
    missing = sorted(required - set(data.keys()))
    return {"valid": not missing, "missing_keys": missing}


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
    import os
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8023")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
