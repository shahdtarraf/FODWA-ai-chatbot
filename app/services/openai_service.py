"""
OpenAI Service — async wrapper for GPT-4o-mini and embeddings.
All calls have timeout=15 and full error handling.
"""

import os
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Initialize async client
_client = None


def _get_client() -> AsyncOpenAI:
    """Get or create the OpenAI async client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.critical("OPENAI_API_KEY environment variable not set!")
            raise ValueError("OPENAI_API_KEY is required")
        _client = AsyncOpenAI(api_key=api_key, timeout=15.0)
        logger.info("OpenAI async client initialized")
    return _client


async def get_embedding(text: str) -> list[float]:
    """
    Generate embedding for a text string using text-embedding-3-small.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key in ["<PUT_YOUR_KEY_HERE>", "mock", "YOUR_OPENAI_API_KEY", ""]:
        logger.info("Mocking embedding response for dummy API key")
        return [0.1] * 1536

    try:
        client = _get_client()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        logger.info(f"Embedding generated: {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise


async def get_chat_response(messages: list[dict]) -> str:
    """
    Get a chat completion from GPT-4o-mini.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key in ["<PUT_YOUR_KEY_HERE>", "mock", "YOUR_OPENAI_API_KEY", ""]:
        logger.info("Mocking chat response for dummy API key")
        return "بالتأكيد، يمكنني الإجابة على استفسارك. كيف يمكنني إزالة إعلاناتك؟ (رسالة تجريبية)"

    try:
        client = _get_client()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1024
        )
        content = response.choices[0].message.content.strip()
        logger.info(f"Chat response received: {len(content)} chars")
        return content
    except Exception as e:
        logger.error(f"Chat API call failed: {e}")
        raise
