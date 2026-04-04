"""
Chat Service — orchestrates the RAG pipeline.
1. Embed query → 2. Search FAISS → 3. Build prompt → 4. Call GPT-4o-mini
Manages per-user conversation history (max 5 messages).
"""

import logging
from app.services.faiss_service import faiss_service
from app.services import openai_service

logger = logging.getLogger(__name__)

# In-memory conversation history (user_id -> list of messages)
_conversation_history: dict[str, list[dict]] = {}

# Maximum messages to keep per user
MAX_HISTORY = 5

# Arabic fallback response
FALLBACK_RESPONSE = "أعتذر، لا أملك معلومات كافية للإجابة على هذا السؤال حالياً."

# System prompt — Arabic support agent persona
SYSTEM_PROMPT = """أنت مساعد ذكي محترف لموقع shahd.ai، اسمك "فودة".

القواعد:
1. أجب دائماً باللغة العربية فقط، حتى لو كان السؤال باللغة الإنجليزية أو لغة مختلطة.
2. استخدم فقط المعلومات المقدمة في السياق أدناه للإجابة.
3. إذا لم تجد إجابة في السياق، قل بالحرف: "أعتذر، لا أملك معلومات كافية للإجابة على هذا السؤال حالياً."
4. كن مهذباً ومحترفاً في إجاباتك.
5. حاول فهم نية المستخدم الصحيحة حتى مع وجود أخطاء إملائية أو لغات مختلطة.
6. لا تكشف عن طريقة تفكيرك أو المصادر الداخلية.
7. أجب بشكل مباشر ومختصر ومفيد.

السياق:
{context}
"""


def _get_history(user_id: str) -> list[dict]:
    """Get conversation history for a user, creating if needed."""
    if user_id not in _conversation_history:
        _conversation_history[user_id] = []
    return _conversation_history[user_id]


def _add_to_history(user_id: str, role: str, content: str):
    """Add a message to user's history, enforcing max limit."""
    history = _get_history(user_id)
    history.append({"role": role, "content": content})

    # Keep only the last MAX_HISTORY messages
    if len(history) > MAX_HISTORY * 2:
        _conversation_history[user_id] = history[-(MAX_HISTORY * 2):]


async def process_chat(message: str, user_id: str = "anonymous") -> str:
    """
    Process a user chat message through the RAG pipeline.

    Args:
        message: User's question/message.
        user_id: User identifier from JWT.

    Returns:
        Arabic response string.
    """
    try:
        # Step 1: Get query embedding
        try:
            query_embedding = await openai_service.get_embedding(message)
        except Exception as e:
            logger.error(f"Failed to get embedding for query: {e}")
            return FALLBACK_RESPONSE

        # Step 2: Search FAISS for relevant chunks
        relevant_chunks = faiss_service.search(query_embedding, top_k=3)

        if not relevant_chunks:
            logger.warning("No relevant chunks found — using fallback")
            return FALLBACK_RESPONSE

        # Step 3: Build context from retrieved chunks
        context = "\n\n---\n\n".join(relevant_chunks)

        # Step 4: Build messages list
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=context)
        }

        # Get user's conversation history
        history = _get_history(user_id)

        messages = [system_message] + history + [
            {"role": "user", "content": message}
        ]

        # Step 5: Call GPT-4o-mini
        try:
            response = await openai_service.get_chat_response(messages)
        except Exception as e:
            logger.error(f"Failed to get chat response: {e}")
            return FALLBACK_RESPONSE

        # Step 6: Update conversation history
        _add_to_history(user_id, "user", message)
        _add_to_history(user_id, "assistant", response)

        logger.info(f"Chat processed for user '{user_id}': {len(response)} chars")
        return response

    except Exception as e:
        logger.error(f"Unexpected error in chat processing: {e}")
        return FALLBACK_RESPONSE
