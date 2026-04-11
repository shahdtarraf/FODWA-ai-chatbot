"""
OpenAI Service — synchronous wrapper for GPT-4o and embeddings.
Converted from AsyncOpenAI to OpenAI (sync) for Django compatibility.
All calls have timeout=15 and full error handling.
"""

import os
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize sync client
_client = None


def _get_client() -> OpenAI:
    """Get or create the OpenAI sync client."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.critical("OPENAI_API_KEY environment variable not set!")
            raise ValueError("OPENAI_API_KEY is required")
        _client = OpenAI(api_key=api_key, timeout=15.0)
        logger.info("OpenAI sync client initialized")
    return _client


def get_embedding(text: str) -> list[float]:
    """
    Generate embedding for a text string using text-embedding-3-small.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key in ["<PUT_YOUR_KEY_HERE>", "mock", "YOUR_OPENAI_API_KEY", ""]:
        logger.info("Mocking embedding response for dummy API key")
        return [0.1] * 1536

    try:
        client = _get_client()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        logger.info(f"Embedding generated: {len(embedding)} dimensions")
        return embedding
    except Exception as e:
        logger.error(f"Embedding API call failed: {e}")
        raise


def get_chat_response(messages: list[dict]) -> str:
    """
    Get a chat completion from GPT-4o.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key in ["<PUT_YOUR_KEY_HERE>", "mock", "YOUR_OPENAI_API_KEY", ""]:
        logger.info("Mocking chat response for dummy API key")
        return "بالتأكيد، يمكنني الإجابة على استفسارك. كيف يمكنني إزالة إعلاناتك؟ (رسالة تجريبية)"

    try:
        client = _get_client()
        response = client.chat.completions.create(
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
