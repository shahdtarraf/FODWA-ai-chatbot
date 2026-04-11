"""
Chat Service — orchestrates the RAG pipeline.
1. Embed query → 2. Search FAISS → 3. Build prompt → 4. Call GPT-4o
Manages per-user conversation history (max 5 messages).

Converted from async to sync for Django compatibility.
All business logic is IDENTICAL to the FastAPI version.
"""

import logging
from chatbot.services.faiss_service import faiss_service
from chatbot.services import openai_service

logger = logging.getLogger(__name__)

# In-memory conversation history (user_id -> list of messages)
_conversation_history: dict[str, list[dict]] = {}

# Maximum messages to keep per user
MAX_HISTORY = 5

# Arabic fallback response
FALLBACK_RESPONSE = "لا أملك معلومات كافية حالياً، يرجى التواصل مع فريق الدعم"

# System prompt — Arabic support agent persona
SYSTEM_PROMPT = """أنت مساعد ذكي ومحترف، خبير في تحليل المستندات المتخصصة والردود الفنية. 
مهمتك الأساسية هي الإجابة على أسئلة المستخدم بذكاء. تعامل بلهجة ودية واحترافية.

التعليمات الأساسية:
• الإجابة باللغة العربية فقط: لا تستخدم أي لغة أخرى.
• طريقة العرض والتنسيق: ابدأ برد قصير وموجز. تجنب الفقرات الطويلة، واستخدم النقاط (•) بدلاً من الأرقام، وضع مسافات بين الأسطر لتسهيل القراءة. قدم إجابات مهيكلة ومنظمة بشكل جميل.
• مثال على الرد النموذجي:
📌 يمكنك القيام بذلك عبر الخطوات التالية:

• ادخل إلى حسابك
• اختر الإعلان
• اضغط على "إلغاء"

🎯 ملاحظة:
تأكد من مراجعة الشروط قبل الإلغاء
• الفهم الذكي والاستنتاج: افهم نية المستخدم الحقيقية، حتى مع الأخطاء الإملائية أو اللهجات العامة.
• التعامل مع مشاكل التنقيب: فك تشفير الكلمات المعكوسة الناتجة عن استخراج النصوص.
• مكافحة الهلوسة: اعتمد بشكل مشدد على السياق المرفق. يُمنع تأليف المعلومات من خارج المستندات.
• غياب المعلومات: إذا كان السؤال خارج السياق ولا تملك معلومات كافية، التزم بالرد التالي:
"لا أملك معلومات كافية حالياً، يرجى التواصل مع فريق الدعم"
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


def process_chat(message: str, user_id: str = "anonymous") -> str:
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
            logger.info(f"[{user_id}] Generating embedding for user query...")
            query_embedding = openai_service.get_embedding(message)
        except Exception as e:
            logger.error(f"[{user_id}] Failed to get embedding for query: {e}")
            return FALLBACK_RESPONSE

        # Step 2: Search FAISS for relevant chunks
        logger.info(f"[{user_id}] Searching FAISS for relevant chunks (top_k=10)...")
        relevant_chunks = faiss_service.search(query_embedding, top_k=10)
        
        # We always create context even if empty. The LLM can handle it intelligently.
        if relevant_chunks:
            context = "\n\n---\n\n".join(relevant_chunks)
            logger.info(f"[{user_id}] Retrieved {len(relevant_chunks)} chunks. Context length: {len(context)} chars")
            # Log a small preview of the context for debugging
            logger.debug(f"[{user_id}] Context preview: {context[:200]}...")
        else:
            logger.warning(f"[{user_id}] No relevant chunks found in FAISS.")
            context = "لا تتوفر أي نصوص أو سياق إضافي للإجابة على هذا السؤال."

        # Step 3: Build user message incorporating the retrieved context
        user_prompt_content = f"""بناءً على هذا السياق المستخرج من المستندات الأساسية:
{context}

---
سؤال المستخدم:
{message}"""

        # Step 4: Build messages list (System -> History -> User)
        system_message = {
            "role": "system",
            "content": SYSTEM_PROMPT
        }

        history = _get_history(user_id)

        messages = [system_message] + history + [
            {"role": "user", "content": user_prompt_content}
        ]

        # Step 5: Call GPT-4o
        try:
            logger.info(f"[{user_id}] Sending payload to GPT-4o...")
            response = openai_service.get_chat_response(messages)
        except Exception as e:
            logger.error(f"[{user_id}] Failed to get chat response: {e}")
            return FALLBACK_RESPONSE

        # Step 6: Update conversation history
        # Note: We only add the strict user 'message' to history to prevent giant context from inflating token usage across turns
        _add_to_history(user_id, "user", message)
        _add_to_history(user_id, "assistant", response)

        logger.info(f"[{user_id}] Chat processed successfully: {len(response)} chars returned")
        return response

    except Exception as e:
        logger.error(f"[{user_id}] Unexpected error in chat processing: {e}")
        return FALLBACK_RESPONSE
