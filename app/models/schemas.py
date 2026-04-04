"""
Pydantic models for request/response schemas.
"""

from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    """Chat endpoint request body."""
    message: str
    token: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat endpoint response body."""
    response: str
    user_id: Optional[str] = None
