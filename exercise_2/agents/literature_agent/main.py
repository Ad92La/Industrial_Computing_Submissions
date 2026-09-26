from __future__ import annotations

import json

from fasta2a.pydantic_ai import agent_to_a2a
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.common import build_model

model = build_model()


class Theme(BaseModel):
    name: str
    summary: str
    supporting_papers: list[str] = Field(default_factory=list)


class LiteratureSynthesis(BaseModel):
    research_focus: str
    themes: list[Theme] = Field(default_factory=list)
    converging_findings: list[str] = Field(default_factory=list)
    divergent_findings: list[str] = Field(default_factory=list)
    research_gaps: list[str] = Field(default_factory=list)
    candidate_thesis_claims: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)


agent = Agent(
    model=model,
    name="literature_synthesis_agent",
    description=(
        "Aggregates several paper analyses into themes, common findings, differences, and research gaps."
    ),
    output_type=str,
    system_prompt=(
        "You synthesize a bundle of already-analyzed papers. "
        "Do not add external claims. "
        "If papers disagree, report both sides and keep uncertainty explicit."
    ),
)


@agent.tool_plain
def validate_paper_bundle(bundle_json: str) -> dict[str, object]:
    try:
        data = json.loads(bundle_json)
    except json.JSONDecodeError as exc:
        return {"valid": False, "error": f"Invalid JSON: {exc}"}
    if not isinstance(data, list):
        return {"valid": False, "error": "Expected a JSON array."}
    return {"valid": True, "paper_count": len(data)}


app = agent_to_a2a(
    agent,
    name="literature_synthesis_agent",
    url="http://host.docker.internal:8022",
    version="1.0.0",
    description=(
        "Synthesizes structured paper analyses into literature-review findings."
    ),
)


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8022")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
