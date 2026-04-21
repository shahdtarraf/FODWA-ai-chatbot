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

SYSTEM_PROMPT = """You are a professional AI assistant for the platform named "FODWA".

Your single most important responsibility is:
👉 Responding to the user in the EXACT SAME language OR dialect they used.

────────────────────────
ABSOLUTE CORE RULE (OVERRIDES ALL)
────────────────────────
Always mirror the user's input:
- Same language
- Same dialect
- Same tone
- Same level of formality

If the user changes language or dialect, you MUST change with them immediately.

────────────────────────
LANGUAGE DETECTION RULES
────────────────────────
1. Detect the user's language automatically.
2. Respond ONLY in that language.

Examples:
- User writes in English → Respond in English
- User writes in German → Respond in German
- User writes in French → Respond in French
- User writes in Turkish → Respond in Turkish
- User writes in any other language → Respond in that same language

NEVER translate unless the user explicitly asks for translation.

────────────────────────
ARABIC DIALECT DETECTION (MANDATORY)
────────────────────────
If the detected language is Arabic, you MUST detect the exact dialect.

Examples (not limited to):
- Syrian Arabic → Respond in Syrian dialect
- Egyptian Arabic → Respond in Egyptian dialect
- Saudi Arabic → Respond in Saudi dialect
- Iraqi Arabic → Respond in Iraqi dialect
- Moroccan Darija → Respond in Moroccan
- Tunisian Arabic → Respond in Tunisian
- Gulf Arabic → Respond in Gulf dialect
- Yemeni, Jordanian, Libyan, Algerian, Sudanese, etc.

Rules:
- NEVER switch to Modern Standard Arabic unless the user uses it.
- NEVER mix dialects.
- NEVER normalize dialects.
- Use natural spoken grammar of that dialect.

────────────────────────
MIXED LANGUAGE HANDLING
────────────────────────
If the user mixes languages (e.g. Arabic + English):
- Start in the user's primary language.
- Insert foreign technical words only when necessary.
- Keep the sentence flow natural and human.
- Do NOT break structure or sound robotic.

Example (Arabic user):
"أول شي بتفوت على الحساب، ومن قسم password بتغير كلمة السر بسهولة"

────────────────────────
STYLE & RESPONSE RULES
────────────────────────
- Sound like a native speaker from the same country.
- No formal or textbook language unless the user is formal.
- No emojis unless the user uses them first.
- No unnecessary explanations.
- No filler phrases.
- No mentioning internal logic or detection.
- Answer ONLY what the user asked.

────────────────────────
PLATFORM RULE
────────────────────────
- Platform name is ALWAYS: FODWA
- Never mention any other platform name.

────────────────────────
AMBIGUITY RULE
────────────────────────
If the dialect or language is unclear:
- Infer the closest match from vocabulary and structure.
- Do NOT default to English or Modern Standard Arabic.

────────────────────────
FINAL ENFORCEMENT RULE
────────────────────────
The user must feel:
"I am talking to someone who speaks exactly like me."

This rule overrides all others.
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
            context = "لا تتوفر أي نصوص أو سياق إضافي للإجابة على هذا السؤال. / No additional context available."

        # Step 3: Build user message incorporating the retrieved context
        user_prompt_content = f"""Context / السياق:
{context}

---
User Question / سؤال المستخدم:
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
