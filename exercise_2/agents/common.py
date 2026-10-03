from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_model() -> OpenAIChatModel:
    base_url = (
        os.getenv("AGENT_LITELLM_BASE_URL")
        or os.getenv("LITELLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )

    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model_name = (
        os.getenv("AGENT_MODEL")
        or os.getenv("LITELLM_MODEL")
        or "gpt-4o-mini"
    )

    print("=== BUILD MODEL ===")
    print(f"base_url  = {base_url}")
    print(f"model     = {model_name}")
    print(f"api_key   = {'***' if api_key else '(leer)'}")
    print("===================")
    
    return OpenAIChatModel(
        model_name,
        provider=OpenAIProvider(
            base_url=base_url,
            api_key=api_key,
        ),
    )