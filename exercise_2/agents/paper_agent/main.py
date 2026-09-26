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
        "Extracts a clean, structured analysis from one paper excerpt: metadata, method, findings, and limitations."
    ),
    output_type=str,
    system_prompt=(
        "You are the paper analysis step in a thesis pipeline. "
    "Use only information available in the provided paper text. "
    "Do not invent bibliography fields or findings. "
    "Return exactly one valid JSON object with these fields: "
    "title, authors, research_question, methodology, sample, findings, "
    "limitations, relevance_to_research_focus. "
    "Use an empty string or empty list when information is unavailable. "
    "Do not use Markdown code fences. "
    "Do not add any text before or after the JSON."
    ),
)


@agent.tool_plain
def normalize_paper_text(text: str) -> str:
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
    import asyncio
    import os
    import uvicorn

    async def debug_agent():
        result = await agent.run(
            """
            Analyze this paper:

            Smith, J. and Lee, A. (2023). Human-AI Collaboration in Knowledge Work.
            We study how AI support changes revision behavior in knowledge tasks.
            Mixed methods with 42 participants: activity logs plus interviews.
            Faster first drafts were observed; final quality depended on verification discipline.
            Limitation: small convenience sample.
            """
        )

        print("=== DIRECT PYDANTIC AI RESULT ===")
        print(repr(result.output))
        print("==================================")

    asyncio.run(debug_agent())

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8021")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
