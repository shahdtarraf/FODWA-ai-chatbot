"""
JWT handler — decodes tokens without strict verification.
Extracts user_id safely; returns 'anonymous' on any failure.
"""

import jwt
import logging

logger = logging.getLogger(__name__)


def decode_token(token: str) -> str:
    """
    Decode a JWT token and extract user_id.
    No signature verification — just payload extraction.

    Args:
        token: JWT token string.

    Returns:
        user_id string, or 'anonymous' if decoding fails.
    """
    if not token:
        return "anonymous"

    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False},
            algorithms=["HS256", "RS256"]
        )
        user_id = payload.get("user_id") or payload.get("sub") or "anonymous"
        logger.info(f"Decoded JWT for user: {user_id}")
        return str(user_id)
    except Exception as e:
        logger.warning(f"JWT decode failed: {e}")
        return "anonymous"
