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
        "Aggregates several structured paper analyses into themes, "
        "common findings, differences, research gaps, and candidate claims."
    ),
    output_type=str,
    system_prompt=(
        "You are the literature synthesis step in a thesis pipeline.\n\n"
        "You receive a JSON array containing analyses of research papers. "
        "Use ONLY the information contained in those paper analyses. "
        "Do not add external knowledge or unsupported claims.\n\n"
        "Return exactly one valid JSON object with these fields:\n"
        "- research_focus: string\n"
        "- themes: array of objects with name, summary, supporting_papers\n"
        "- converging_findings: array of strings\n"
        "- divergent_findings: array of strings\n"
        "- research_gaps: array of strings\n"
        "- candidate_thesis_claims: array of strings\n"
        "- uncertainty_flags: array of strings\n\n"
        "If information is unavailable, use an empty string or empty array.\n"
        "supporting_papers MUST always be an array of strings.\n"
        "Do not invent citations, authors, findings, or research gaps.\n"
        "If papers disagree, represent the disagreement explicitly.\n"
        "Do not use Markdown code fences.\n"
        "Do not add explanations before or after the JSON object."
    ),
)


@agent.tool_plain
def validate_paper_bundle(
    bundle_json: str,
) -> dict[str, object]:
    """Validate that the input is a JSON array of paper analyses."""
    try:
        data = json.loads(bundle_json)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "error": f"Invalid JSON: {exc}",
        }

    if not isinstance(data, list):
        return {
            "valid": False,
            "error": "Expected a JSON array.",
        }

    return {
        "valid": True,
        "paper_count": len(data),
    }


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
    import uvicorn
    import os

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8022")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )