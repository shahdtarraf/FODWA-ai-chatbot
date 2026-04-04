"""
Chat route — POST /chat endpoint.
"""

import logging
from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.auth.jwt_handler import decode_token
from app.services.chat_service import process_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint — accepts a message and optional JWT token.
    Returns an Arabic response from the RAG pipeline.
    """
    try:
        # Extract user_id from JWT token
        user_id = decode_token(request.token) if request.token else "anonymous"

        # Process message through RAG pipeline
        response_text = await process_chat(
            message=request.message,
            user_id=user_id
        )

        return ChatResponse(
            response=response_text,
            user_id=user_id
        )

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response="أعتذر، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.",
            user_id="anonymous"
        )
