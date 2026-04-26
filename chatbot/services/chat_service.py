"""
Chat Service — orchestrates the RAG pipeline.
1. Embed query → 2. Search FAISS → 3. Build prompt → 4. Call GPT-4o
Manages per-user conversation history (max 5 messages).

Converted from async to sync for Django compatibility.
All business logic is IDENTICAL to the FastAPI version.

NLP Layer (added — non-breaking):
  - Pre-processing via nlp_utils.preprocess() detects language, dialect, intent.
  - Router: rephrase_request intent → bypass FAISS, use last assistant message.
  - Dynamic instructions appended to BASE_SYSTEM_PROMPT per request.
  - Post-processing via nlp_utils.post_process_response() fixes RTL + truncation.
"""

import logging
from chatbot.services.faiss_service import faiss_service
from chatbot.services import openai_service
from chatbot.utils.nlp_utils import preprocess, build_dynamic_system_prompt, post_process_response

logger = logging.getLogger(__name__)

# In-memory conversation history (user_id -> list of messages)
_conversation_history: dict[str, list[dict]] = {}

# Maximum messages to keep per user
MAX_HISTORY = 5

# Generic fallback response for system errors
FALLBACK_RESPONSE = "عذراً، حدث خطأ في النظام ولا يمكننا معالجة طلبك حالياً. للتواصل مع الدعم:\nPhone: 00436763205041\nEmail: mohammed.kudjar@gmail.com"

# BASE_SYSTEM_PROMPT — never modified at runtime.
# Dynamic per-request instructions are APPENDED by build_dynamic_system_prompt().
BASE_SYSTEM_PROMPT = """You are a professional AI assistant for the platform named "FODWA".

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
        String containing the response in the user's language/dialect.
    """
    try:
        history = _get_history(user_id)

        # ── NLP Pre-processing ───────────────────────────────────────────────
        # Runs BEFORE any FAISS or LLM call.
        # Returns: NLPMetadata(language, dialect, intent)
        nlp = preprocess(message, history)
        logger.info(
            f"[{user_id}] NLP | lang={nlp.language!r} "
            f"dialect={nlp.dialect!r} intent={nlp.intent!r}"
        )

        # ── Router: Rephrase path ────────────────────────────────────────────
        if nlp.intent == "rephrase_request":
            logger.info(f"[{user_id}] Router → rephrase_request: bypassing FAISS entirely")

            # Find the last assistant message from history
            last_assistant_msg = None
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg["content"]
                    break

            if not last_assistant_msg:
                # No prior assistant message — treat as new question instead
                logger.warning(
                    f"[{user_id}] rephrase_request but no prior assistant message. "
                    "Falling back to new_question flow."
                )
                nlp.intent = "new_question"
            else:
                # Build rephrase-specific user content (no context, no FAISS)
                user_prompt_content = (
                    f"Previous answer to rephrase:\n{last_assistant_msg}\n\n"
                    f"User request: {message}"
                )

                # Build dynamic system prompt (base + rephrase instructions)
                dynamic_instructions = build_dynamic_system_prompt(
                    lang=nlp.language,
                    dialect=nlp.dialect,
                    intent="rephrase_request",
                )
                full_system_content = BASE_SYSTEM_PROMPT + "\n\n" + dynamic_instructions

                messages = [
                    {"role": "system", "content": full_system_content},
                    # No history injected for rephrase — keeps context minimal
                    {"role": "user", "content": user_prompt_content},
                ]

                try:
                    logger.info(f"[{user_id}] Sending rephrase payload to GPT-4o...")
                    raw_response = openai_service.get_chat_response(messages)
                except Exception as e:
                    logger.error(f"[{user_id}] GPT-4o rephrase call failed: {e}")
                    return FALLBACK_RESPONSE

                # Post-processing
                response = post_process_response(raw_response)

                # Update history
                _add_to_history(user_id, "user", message)
                _add_to_history(user_id, "assistant", response)

                logger.info(
                    f"[{user_id}] Rephrase processed successfully: {len(response)} chars returned"
                )
                return response

        # ── Router: New question path (default) ─────────────────────────────
        logger.info(f"[{user_id}] Router → new_question: executing FAISS search")

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
            logger.info(
                f"[{user_id}] Retrieved {len(relevant_chunks)} chunks. "
                f"Context length: {len(context)} chars"
            )
            logger.debug(f"[{user_id}] Context preview: {context[:200]}...")
        else:
            logger.warning(f"[{user_id}] No relevant chunks found in FAISS.")
            context = (
                "لا تتوفر أي نصوص أو سياق إضافي للإجابة على هذا السؤال. "
                "/ No additional context available."
            )

        # Step 3: Build user message incorporating the retrieved context
        user_prompt_content = (
            f"Context / السياق:\n{context}\n\n"
            f"---\nUser Question / سؤال المستخدم:\n{message}"
        )

        # Step 4: Build dynamic system prompt (base + new_question instructions)
        dynamic_instructions = build_dynamic_system_prompt(
            lang=nlp.language,
            dialect=nlp.dialect,
            intent="new_question",
        )
        full_system_content = BASE_SYSTEM_PROMPT + "\n\n" + dynamic_instructions

        # Step 5: Build messages list (System -> History -> User)
        messages = [
            {"role": "system", "content": full_system_content}
        ] + history + [
            {"role": "user", "content": user_prompt_content}
        ]

        # Step 6: Call GPT-4o
        try:
            logger.info(f"[{user_id}] Sending payload to GPT-4o...")
            raw_response = openai_service.get_chat_response(messages)
        except Exception as e:
            logger.error(f"[{user_id}] Failed to get chat response: {e}")
            return FALLBACK_RESPONSE

        # Post-processing
        response = post_process_response(raw_response)

        # Step 7: Update conversation history
        # Note: store only the raw user message (not the full context-injected prompt)
        # to prevent token inflation across turns.
        _add_to_history(user_id, "user", message)
        _add_to_history(user_id, "assistant", response)

        logger.info(f"[{user_id}] Chat processed successfully: {len(response)} chars returned")
        return response

    except Exception as e:
        logger.error(f"[{user_id}] Unexpected error in chat processing: {e}")
        return FALLBACK_RESPONSE
