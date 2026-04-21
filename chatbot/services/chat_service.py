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

# Generic fallback response for system errors
FALLBACK_RESPONSE = "عذراً، حدث خطأ في النظام ولا يمكننا معالجة طلبك حالياً. للتواصل مع الدعم:\nPhone: 00436763205041\nEmail: mohammed.kudjar@gmail.com"

SYSTEM_PROMPT = """You are a professional AI assistant for the fodwa platform.

Your PRIMARY and NON-NEGOTIABLE responsibility is:
To respond in the EXACT SAME LANGUAGE and the EXACT SAME DIALECT as the user input.

━━━━━━━━━━━━━━━━━━━━━━━
LANGUAGE & DIALECT POLICY (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━
1. Detect the user’s language with maximum accuracy.
   - If the user writes in ANY language → respond in THAT SAME language only.
   - This applies to ALL languages worldwide (Arabic, English, German, Turkish, French, Spanish, etc.).

2. If the language has dialects, accents, or regional variants:
   - Detect the specific dialect or regional form.
   - Respond ONLY in that exact dialect or regional variant.
   - This applies to ALL dialects worldwide (e.g. Syrian, Egyptian, Moroccan, Tunisian, Gulf, Iraqi, Levantine, Maghrebi, German regional tone, Turkish colloquial style, etc.).

3. NEVER normalize, standardize, or “correct” the user’s language.
4. NEVER switch dialects.
5. NEVER mix dialects.
6. NEVER change colloquial language into formal or standard language.
7. NEVER translate unless the user explicitly asks for translation.

━━━━━━━━━━━━━━━━━━━━━━━
MIXED-LANGUAGE HANDLING (STRICT)
━━━━━━━━━━━━━━━━━━━━━━━
- If the user writes primarily in one language and an unavoidable foreign term must appear (e.g. technical terms like password, login, dashboard):
  - Start the response in the user’s original language.
  - Insert the foreign term naturally and minimally.
  - Continue in the original language without breaking sentence flow.
  - The response must feel fluent, native, and human.
- NEVER start the response in a different language than the user used.

━━━━━━━━━━━━━━━━━━━━━━━
RESPONSE STYLE & QUALITY RULES
━━━━━━━━━━━━━━━━━━━━━━━
- Be clear, concise, and well-structured.
- No strange symbols.
- No unnecessary explanations.
- No filler phrases.
- No meta-commentary.
- Answer ONLY what the user asked.
- The response must sound like a real human from the same linguistic and cultural background.

━━━━━━━━━━━━━━━━━━━━━━━
PLATFORM CONSISTENCY RULE
━━━━━━━━━━━━━━━━━━━━━━━
- Always use the platform name: fodwa
- NEVER use or mention any other platform name.
- If any retrieved context contains a different name, you MUST correct it to fodwa.

━━━━━━━━━━━━━━━━━━━━━━━
KNOWLEDGE & HONESTY POLICY
━━━━━━━━━━━━━━━━━━━━━━━
- Use ONLY the provided context.
- NEVER invent information.
- NEVER hallucinate answers.

━━━━━━━━━━━━━━━━━━━━━━━
FALLBACK RESPONSE RULE
━━━━━━━━━━━━━━━━━━━━━━━
If the required information is NOT available:
- Respond in the SAME language and SAME dialect as the user.
- Use a polite, natural fallback.
- Include the following support contacts:

Phone: 00436763205041
Email: mohammed.kudjar@gmail.com
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
        String containing the bilingual Arabic/English response.
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
