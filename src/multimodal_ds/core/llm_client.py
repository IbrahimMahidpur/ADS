"""
Unified LLM client supporting multiple backends.
Supports OpenCode Zen and Ollama based on model prefix.
"""
import logging
import os
import re

import httpx

from multimodal_ds.config import LLM_TIMEOUT

logger = logging.getLogger(__name__)

OPENCODE_BASE_URL = "https://opencode.ai/zen/v1"
OPENCODE_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

RETRY_DELAYS = [2, 4, 8]
MAX_RETRIES = 3


def _strip_think_blocks(text: str) -> str:
    """Remove <think>... blocks from response using regex with DOTALL."""
    return re.sub(r'<think>.*?', '', text, flags=re.DOTALL).strip()


def _call_opencode_zen(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float
) -> str:
    """Call OpenCode Zen (OpenAI-compatible API) with retry logic."""
    headers = {
        "Authorization": f"Bearer {OPENCODE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model.replace("opencode/", ""),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.post(
                f"{OPENCODE_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=httpx.Timeout(LLM_TIMEOUT)
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if response.status_code in (429, 500):
                last_exception = Exception(f"HTTP {response.status_code}: {response.text}")
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"[LLM Client] OpenCode Zen returned {response.status_code}, "
                        f"retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                    )
                    import time
                    time.sleep(delay)
                    continue
            else:
                logger.error(f"[LLM Client] OpenCode Zen HTTP {response.status_code}: {response.text}")
                response.raise_for_status()

        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code in (429, 500) and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    f"[LLM Client] OpenCode Zen HTTP error {e.response.status_code}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                import time
                time.sleep(delay)
                continue
            logger.error(f"[LLM Client] OpenCode Zen HTTP error: {e}")
            raise
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                logger.warning(
                    f"[LLM Client] OpenCode Zen request failed: {e}, "
                    f"retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                import time
                time.sleep(delay)
                continue
            logger.error(f"[LLM Client] OpenCode Zen request failed: {e}")
            raise

    raise last_exception


def _call_ollama(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float
) -> str:
    """Call local Ollama API."""
    payload = {
        "model": model.replace("ollama/", ""),
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=httpx.Timeout(LLM_TIMEOUT)
        )

        if response.status_code == 200:
            data = response.json()
            return data.get("message", {}).get("content", "")
        else:
            logger.error(f"[LLM Client] Ollama HTTP {response.status_code}: {response.text}")
            response.raise_for_status()

    except Exception as e:
        logger.error(f"[LLM Client] Ollama request failed: {e}")
        raise


def chat(
    model: str,
    messages: list[dict],
    max_tokens: int = 1000,
    temperature: float = 0.1
) -> str:
    """
    Unified chat function that routes to the appropriate backend based on model prefix.

    Args:
        model: Model name with prefix (e.g., "opencode/..." or "ollama/...")
        messages: List of message dicts with "role" and "content" keys
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text with <think>... blocks stripped

    Raises:
        Exception: On API errors (propagated for fallback handling)
    """
    if model.startswith("opencode/"):
        response = _call_opencode_zen(model, messages, max_tokens, temperature)
    elif model.startswith("ollama/"):
        response = _call_ollama(model, messages, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown model prefix in: {model}. Expected 'opencode/' or 'ollama/'")

    return _strip_think_blocks(response)


def chat_with_fallback(
    primary_model: str,
    fallback_model: str,
    messages: list[dict],
    max_tokens: int = 1000,
    temperature: float = 0.1
) -> str:
    """
    Try primary_model first, fall back to fallback_model on any exception.

    Args:
        primary_model: Model to try first
        fallback_model: Model to use if primary fails
        messages: List of message dicts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    try:
        return chat(primary_model, messages, max_tokens, temperature)
    except Exception as e:
        logger.warning(
            f"[LLM Client] Primary model {primary_model} failed: {e}. "
            f"Falling back to {fallback_model}."
        )
        return chat(fallback_model, messages, max_tokens, temperature)