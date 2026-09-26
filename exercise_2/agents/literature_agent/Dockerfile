FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN uv pip install --system --no-cache \
    "fasta2a[pydantic-ai]>=0.6.1" \
    "httpx>=0.28.1" \
    "pydantic-ai-slim[openai]>=2.18.0" \
    "python-dotenv>=1.2.2" \
    "uvicorn>=0.51.0"

COPY exercise_2/agents ./agents

EXPOSE 8022
CMD ["uvicorn", "agents.literature_agent.main:app", "--host", "0.0.0.0", "--port", "8022"]
