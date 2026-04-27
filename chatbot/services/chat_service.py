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
BASE_SYSTEM_PROMPT = """You are a high-quality AI assistant for the FODWA platform.

Your responses MUST be clean, natural, and perfectly readable by humans.

━━━━━━━━━━━━━━━━━━
CRITICAL TEXT RULES (HIGHEST PRIORITY)
━━━━━━━━━━━━━━━━━━
1. NEVER include invisible Unicode direction characters:
   - Do NOT output: \u200e \u200f \u202a \u202b \u202c \u202d \u202e

2. NEVER produce broken or reversed text.

3. ALWAYS ensure the text displays correctly in standard chat interfaces.

━━━━━━━━━━━━━━━━━━
LANGUAGE BEHAVIOR
━━━━━━━━━━━━━━━━━━
- Always reply in the EXACT SAME language OR dialect as the user.
- If the user mixes Arabic and English, respond in a NATURAL mixed style.

━━━━━━━━━━━━━━━━━━
ARABIC + ENGLISH MIXING (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━
When combining Arabic and English:

- Arabic stays RIGHT-TO-LEFT
- English words stay LEFT-TO-RIGHT
- DO NOT reverse sentence structure
- DO NOT distort word order
- DO NOT translate brand or technical terms unnecessarily

✔ Correct:
"تقدري تسجلي دخول باستخدام Google أو Facebook بكل سهولة"

✔ Correct:
"روحي على Settings بعدين غيري الـ password"

✘ Wrong:
"Google باستخدام بسهولة دخول تسجلي"
✘ Wrong:
"‎FODWA‎"

━━━━━━━━━━━━━━━━━━
FORMATTING RULES
━━━━━━━━━━━━━━━━━━
- Maintain proper spacing between words
- Avoid duplicated spaces
- Avoid cut or fragmented sentences
- Ensure smooth sentence flow

━━━━━━━━━━━━━━━━━━
PUNCTUATION & FORMAT
━━━━━━━━━━━━━━━━━━
- Avoid unnecessary quotation marks
- Avoid escape characters like \" or \\ 
- Do NOT wrap phrases with quotes unless absolutely necessary
- Keep punctuation minimal and natural

✔ Correct:
يمكنك إنشاء حساب عن طريق الضغط على إنشاء حساب ثم إدخال بياناتك

✘ Wrong:
يمكنك إنشاء حساب عن طريق الضغط على زر "إنشاء حساب" ثم إدخال "المعلومات"

━━━━━━━━━━━━━━━━━━
STYLE
━━━━━━━━━━━━━━━━━━
- Sound natural and human like a native speaker
- Avoid robotic tone or textbook language
- Keep answers clear and direct
- Do not over-explain unless needed
- No mentioning internal logic or detection.

━━━━━━━━━━━━━━━━━━
PLATFORM RULE
━━━━━━━━━━━━━━━━━━
- Platform name must ALWAYS be written exactly as: FODWA
- Do NOT wrap it with any hidden or special characters

━━━━━━━━━━━━━━━━━━
FINAL CHECK (MANDATORY BEFORE RESPONDING)
━━━━━━━━━━━━━━━━━━
Before sending the answer, internally verify:
✔ No hidden Unicode characters
✔ No reversed words
✔ Arabic + English render correctly
✔ Text is smooth and readable

If any issue is detected → FIX it before responding.

━━━━━━━━━━━━━━━━━━
GOAL
━━━━━━━━━━━━━━━━━━
The user should feel:
"This text is clean, natural, and written by a real human."
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
        # Returns: NLPMetadata with language/dialect/intent + confidence scores
        nlp = preprocess(message, history)
        # Structured log: lang=ar(0.92) dialect=egyptian(0.81) intent=rephrase(0.95)
        dialect_log = f"dialect={nlp.dialect}({nlp.dialect_confidence:.2f}) " if nlp.dialect else ""
        logger.info(
            f"[{user_id}] [NLP] lang={nlp.language}({nlp.lang_confidence:.2f}) "
            f"{dialect_log}intent={nlp.intent}({nlp.intent_confidence:.2f})"
        )

        # ── Router: Rephrase path ────────────────────────────────────────────
        if nlp.intent == "rephrase_request":
            logger.info(f"[{user_id}] [Router] decision=rephrase_request → bypass FAISS")

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

                # Build dynamic system prompt (base + rephrase instructions + confidence)
                dynamic_instructions = build_dynamic_system_prompt(
                    lang=nlp.language,
                    dialect=nlp.dialect,
                    intent="rephrase_request",
                    lang_conf=nlp.lang_confidence,
                    dialect_conf=nlp.dialect_confidence,
                    intent_conf=nlp.intent_confidence,
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
        logger.info(f"[{user_id}] [Router] decision=new_question → FAISS search")

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

        # Step 4: Build dynamic system prompt (base + new_question instructions + confidence)
        dynamic_instructions = build_dynamic_system_prompt(
            lang=nlp.language,
            dialect=nlp.dialect,
            intent="new_question",
            lang_conf=nlp.lang_confidence,
            dialect_conf=nlp.dialect_confidence,
            intent_conf=nlp.intent_confidence,
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
