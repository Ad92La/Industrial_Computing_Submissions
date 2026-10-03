from __future__ import annotations

import re
from fasta2a.pydantic_ai import agent_to_a2a
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from agents.common import build_model

model = build_model()

class PaperAnalysis(BaseModel):
    title: str
    authors: list[str]
    research_question: str
    methodology: str
    sample: str
    findings: list[str]
    limitations: list[str]
    relevance_to_research_focus: str


agent = Agent(
    model=model,
    name="paper_analysis_agent",
    description=(
        "Extracts a structured analysis from one research paper excerpt, "
        "including metadata, research question, methodology, findings, "
        "limitations, and relevance."
    ),
    output_type=PaperAnalysis,
    system_prompt=(
        "/no_think\n\n"
        "You are the paper analysis step in a thesis literature-review pipeline.\n\n"
        "Analyze ONLY the paper text provided by the user. "
        "Do not use external knowledge and do not invent information.\n\n"
        "Return exactly one valid JSON object with these fields:\n"
        "- title: string\n"
        "- authors: array of strings\n"
        "- research_question: string\n"
        "- methodology: string\n"
        "- sample: string\n"
        "- findings: array of strings\n"
        "- limitations: array of strings\n"
        "- relevance_to_research_focus: string\n\n"
        "If information is unavailable, use an empty string or empty array.\n"
        "The fields findings and limitations MUST always be JSON arrays of strings.\n"
        "authors MUST always be a JSON array of strings.\n"
        "Do not use Markdown code fences.\n"
        "Do not add explanations before or after the JSON object."
    ),
)


@agent.tool_plain
def normalize_paper_text(text: str) -> str:
    """Normalize whitespace and remove null characters from paper text."""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


app = agent_to_a2a(
    agent,
    name="paper_analysis_agent",
    url="http://host.docker.internal:8021",
    version="1.0.0",
    description=(
        "Analyzes research papers and extracts structured findings."
    ),
)


if __name__ == "__main__":
    import uvicorn
    import os

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8021")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )