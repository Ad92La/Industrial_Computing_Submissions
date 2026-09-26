# Exercise 2: Distributed Remote A2A System (Dynamic Registry Lookup)

This folder contains a simple distributed multi-agent setup with **3 separate paper-focused agents**:

1. `paper_analysis_agent` on port `8021`
2. `literature_synthesis_agent` on port `8022`
3. `thesis_writing_agent` on port `8023`

All agents use **LiteLLM only** and run as separate Docker containers. Each agent has its own Dockerfile in its own folder, like the sample network.

## What is included

- `agents/paper_agent/main.py`: extracts structured information from raw paper text
- `agents/literature_agent/main.py`: synthesizes multiple paper analyses
- `agents/writing_agent/main.py`: drafts a thesis section using the synthesis
- `demo/workflow_demo.py`: dynamic registry lookup + end-to-end pipeline call
- `agents/*/Dockerfile`: one Dockerfile per agent
- `docker-compose.yml`: starts each agent in its own container and port

## Setup (single repo-level `.env`)

1. Create one `.env` in the repository root from your existing root `.env.example`.

2. Ensure these LiteLLM keys are set there:
   - `LITELLM_BASE_URL`
   - `LITELLM_API_KEY`
   - `LITELLM_MODEL`

3. Register the three agents in LiteLLM using these exact names so the demo can resolve them from `GET /v1/agents`:
   - `paper_analysis_agent`
   - `literature_synthesis_agent`
   - `thesis_writing_agent`

   If LiteLLM restricts agent callback URLs, add the Docker host/IP to `user_url_allowed_hosts` in the LiteLLM config.

4. Start all 3 agents:

   ```bash
   docker compose -f exercise_2/docker-compose.yml up -d --build
   ```

5. For local non-Docker runs, use `uv`:

   ```bash
   uv sync --project exercise_2
   uv run --project exercise_2 python -m agents.paper_agent.main
   ```

## Dynamic registry lookup demo

Run the demo script (with LiteLLM reachable and agents registered there):

```bash
uv run --project exercise_2 python exercise_2/demo/workflow_demo.py
```

The demo:

- fetches available agents from `GET /v1/agents`
- uses an LLM router to select the best agent for a paper-analysis intent
- resolves all 3 required agents dynamically by registry name
- executes the full pipeline: **Paper -> Literature -> Writing**
