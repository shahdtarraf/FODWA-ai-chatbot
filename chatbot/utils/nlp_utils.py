"""
NLP Utilities — Pre/Post-processing layer for FODWA Chatbot.

Responsibilities:
  1. detect_language(text)          → ISO language code ('ar', 'en', 'de', ...)
  2. detect_arabic_dialect(text)    → Dialect label ('egyptian', 'syrian', ...)
  3. detect_intent(text, history)   → Intent label ('new_question', 'rephrase_request')
  4. build_dynamic_system_prompt()  → Dynamic instructions appended to base SYSTEM_PROMPT
  5. post_process_response(text)    → Fix FODWA RTL bug + truncate if needed

Design rules:
  - ZERO dependencies on chat_service, faiss_service, or openai_service.
  - All functions are pure and stateless (except logging).
  - Never raises — always returns a safe default.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

# Left-to-Right Mark — forces "FODWA" to render correctly in RTL contexts
_LRM = "\u200E"

# Maximum lines allowed in a response before hard truncation
_MAX_LINES = 4

# Maximum characters as a secondary safety net
_MAX_CHARS = 600


# ─────────────────────────────────────────────────────────────
# 1. Language Detection
# ─────────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Returns:
        ISO 639-1 language code string, e.g. 'ar', 'en', 'de', 'fr', 'tr'.
        Falls back to 'en' on any error.
    """
    if not text or not text.strip():
        return "en"

    # Fast Arabic check via Unicode range (avoids langdetect overhead for common case)
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars / max(len(text.strip()), 1) > 0.3:
        logger.debug(f"[NLP] Language detected via Unicode heuristic: ar")
        return "ar"

    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        logger.debug(f"[NLP] Language detected via langdetect: {lang}")
        return lang
    except Exception as e:
        logger.warning(f"[NLP] langdetect failed: {e} — falling back to 'en'")
        return "en"


# ─────────────────────────────────────────────────────────────
# 2. Arabic Dialect Detection
# ─────────────────────────────────────────────────────────────

# Rule-based keyword map. Order matters: more specific dialects first.
_DIALECT_KEYWORDS: list[tuple[str, list[str]]] = [
    ("egyptian",   ["ازيك", "ازيكم", "عامل ايه", "عامله ايه", "ايه ده", "ده", "دي",
                    "ايه", "مش عارف", "مش عارفة", "بقى", "بقا", "اللي", "كده",
                    "عايز", "عايزة", "فين", "امتى", "ليه", "هو ايه"]),

    ("syrian",     ["شو", "هلق", "كيفك", "شلونك", "منيح", "هيك", "عنجد", "ما في",
                    "في شي", "ليش", "هون", "هناك", "شو بدك", "شو بدي"]),

    ("iraqi",      ["شگول", "شلونك", "شنو", "هواية", "گاعد", "چي", "بعد شوية",
                    "أكو", "ماكو", "شبيك", "يمه", "أبه", "گلتلك"]),

    ("gulf",       ["وش", "ليش", "ايش", "كيف حالك", "إيش", "زين", "وايد",
                    "شلون", "حيل", "عيل", "يبه", "يمه", "ما ادري", "ترى"]),

    ("moroccan",   ["واش", "كيداير", "كيف داير", "بزاف", "ماشي", "دابا", "هاد",
                    "بغيت", "مزيان", "لا باس"]),

    ("tunisian",   ["شنية", "برشة", "ياسر", "فما", "نحب", "نعرف", "كيفاش",
                    "شنو", "علاش"]),

    ("algerian",   ["واش راك", "راك", "نتا", "نتي", "كيراك", "رانا", "هكذا",
                    "دروك", "تاع"]),

    ("libyan",     ["شحالك", "شحالكم", "كيفاش", "أشحال", "شكون"]),

    ("sudanese",   ["زين", "كيفك", "ما علينا", "يا زول", "زول", "ما قادر"]),

    ("yemeni",     ["كيف حالك", "ايش", "وش", "شو", "ما اعرف", "اليوم"]),

    ("jordanian",  ["شو بدك", "والله", "هلأ", "هلق", "بدي", "شو صار",
                    "يعني إيش"]),

    ("levantine",  ["كيفك", "شو", "هلأ", "عم", "رح", "بدي", "ما بدي"]),

    ("msa",        ["هل يمكنني", "أود أن", "أرجو", "يرجى", "من فضلك"]),
]


def detect_arabic_dialect(text: str) -> str:
    """
    Detect the Arabic dialect using rule-based keyword heuristics.

    Args:
        text: Arabic input text.

    Returns:
        Dialect label string. Possible values:
        'egyptian', 'syrian', 'iraqi', 'gulf', 'moroccan', 'tunisian',
        'algerian', 'libyan', 'sudanese', 'yemeni', 'jordanian', 'levantine',
        'msa', 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"

    text_lower = text.lower().strip()
    scores: dict[str, int] = {}

    for dialect, keywords in _DIALECT_KEYWORDS:
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[dialect] = count

    if not scores:
        logger.debug("[NLP] Arabic dialect: unknown (no keywords matched)")
        return "unknown"

    best = max(scores, key=lambda d: scores[d])
    logger.debug(f"[NLP] Arabic dialect detected: {best} (score={scores[best]})")
    return best


# ─────────────────────────────────────────────────────────────
# 3. Intent Detection
# ─────────────────────────────────────────────────────────────

# Patterns that indicate the user wants to rephrase/translate the PREVIOUS answer.
# Grouped into clear intent categories.
_REPHRASE_PATTERNS: list[re.Pattern] = [
    # ── Language switch requests ────────────────────────────
    re.compile(r"\b(in\s+english|translate\s+(to|into)\s+english)\b", re.IGNORECASE),
    re.compile(r"\b(auf\s+deutsch|translate\s+(to|into)\s+german)\b", re.IGNORECASE),
    re.compile(r"\b(en\s+français|translate\s+(to|into)\s+french)\b", re.IGNORECASE),
    re.compile(r"\b(translate|ترجم|ترجمة|ترجملي|حولها)\b", re.IGNORECASE),
    re.compile(r"(اكتب(ها|ه|لي)?\s*(بالانجليزي|بالعربي|بالألماني|بالفرنسي|بالتركي))", re.IGNORECASE),
    re.compile(r"(قلها|قوله|قولها)\s*(بالانجليزي|بالعربي|بالألماني|بالفرنسي)", re.IGNORECASE),
    re.compile(r"(بالإنجليزي|بالانجليزي|بالإنكليزي|باللغة الإنجليزية)", re.IGNORECASE),
    re.compile(r"(بالعربي|باللغة العربية|بالعربية)", re.IGNORECASE),
    re.compile(r"(بالألماني|باللغة الألمانية|بالالماني)", re.IGNORECASE),
    re.compile(r"(بالفرنسي|باللغة الفرنسية)", re.IGNORECASE),
    re.compile(r"(بالتركي|باللغة التركية)", re.IGNORECASE),

    # ── Dialect switch requests ──────────────────────────────
    re.compile(r"(حولها|حوله|اكتبها|قولها|قله)\s*(بالمصري|بالسوري|بالعراقي|بالخليجي|بالمغربي|بالتونسي|بالجزائري|بالليبي|بالأردني|بالشامي|بالفصحى|بالعامية)", re.IGNORECASE),
    re.compile(r"(بالمصري|بالمصرية|باللهجة المصرية)", re.IGNORECASE),
    re.compile(r"(بالسوري|باللهجة السورية|بالشامي)", re.IGNORECASE),
    re.compile(r"(بالعراقي|باللهجة العراقية)", re.IGNORECASE),
    re.compile(r"(بالخليجي|باللهجة الخليجية|بالسعودي|بالإماراتي|بالكويتي)", re.IGNORECASE),
    re.compile(r"(بالمغربي|باللهجة المغربية|بالدارجة)", re.IGNORECASE),
    re.compile(r"(بالأردني|باللهجة الأردنية)", re.IGNORECASE),
    re.compile(r"(بالفصحى|بالعربية الفصحى|بالمعياري)", re.IGNORECASE),
    re.compile(r"(بالعامية)", re.IGNORECASE),

    # ── Rephrase without translation ────────────────────────
    re.compile(r"(أعد(ها|ه)?\s*(بكلمات|بأسلوب|بطريقة)\s*أبسط)", re.IGNORECASE),
    re.compile(r"(اشرح(ها|ه)?\s*(بطريقة|بأسلوب)\s*أبسط)", re.IGNORECASE),
    re.compile(r"(بشكل\s*أبسط|بكلمات\s*أبسط|بطريقة\s*أسهل)", re.IGNORECASE),
    re.compile(r"(simplify|rephrase|rewrite|say\s+it\s+(again|differently))", re.IGNORECASE),
]


def detect_intent(text: str, history: list[dict] | None = None) -> str:
    """
    Detect user intent from the message text and conversation history.

    Logic:
      - If the message matches any rephrase/translation pattern AND
        there is at least one previous assistant message → 'rephrase_request'
      - Otherwise → 'new_question'

    Args:
        text:    The user's current message.
        history: Conversation history list (role/content dicts).

    Returns:
        'rephrase_request' | 'new_question'
    """
    if not text or not text.strip():
        return "new_question"

    text_stripped = text.strip()

    # Only consider rephrase if the message is short (not a new detailed question)
    # and there's previous history to rephrase
    has_prior_assistant = any(
        m.get("role") == "assistant" for m in (history or [])
    )

    if has_prior_assistant and len(text_stripped) < 120:
        for pattern in _REPHRASE_PATTERNS:
            if pattern.search(text_stripped):
                logger.debug(f"[NLP] Intent detected: rephrase_request (pattern={pattern.pattern[:40]})")
                return "rephrase_request"

    logger.debug("[NLP] Intent detected: new_question")
    return "new_question"


# ─────────────────────────────────────────────────────────────
# 4. Dynamic System Prompt Builder
# ─────────────────────────────────────────────────────────────

_DIALECT_LABELS: dict[str, str] = {
    "egyptian":   "Egyptian Arabic (عامية مصرية)",
    "syrian":     "Syrian Arabic (لهجة سورية شامية)",
    "iraqi":      "Iraqi Arabic (لهجة عراقية)",
    "gulf":       "Gulf Arabic (لهجة خليجية)",
    "moroccan":   "Moroccan Darija (الدارجة المغربية)",
    "tunisian":   "Tunisian Arabic (اللهجة التونسية)",
    "algerian":   "Algerian Arabic (اللهجة الجزائرية)",
    "libyan":     "Libyan Arabic (اللهجة الليبية)",
    "sudanese":   "Sudanese Arabic (اللهجة السودانية)",
    "yemeni":     "Yemeni Arabic (اللهجة اليمنية)",
    "jordanian":  "Jordanian Arabic (اللهجة الأردنية)",
    "levantine":  "Levantine Arabic (اللهجة الشامية)",
    "msa":        "Modern Standard Arabic (العربية الفصحى)",
    "unknown":    "Arabic (detect dialect from context)",
}

_LANG_LABELS: dict[str, str] = {
    "ar": "Arabic",
    "en": "English",
    "de": "German",
    "fr": "French",
    "tr": "Turkish",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "fa": "Persian",
    "ur": "Urdu",
}


def build_dynamic_system_prompt(
    lang: str,
    dialect: str | None = None,
    intent: str = "new_question",
    rephrase_target: str | None = None,
) -> str:
    """
    Build the dynamic instruction block to be appended to the base SYSTEM_PROMPT.

    This does NOT replace the base prompt — it extends it with
    request-specific instructions so the base rules are always preserved.

    Args:
        lang:            Detected language code (e.g. 'ar', 'en', 'de').
        dialect:         Detected Arabic dialect (only used if lang == 'ar').
        intent:          'new_question' or 'rephrase_request'.
        rephrase_target: Optional explicit target language/dialect if user specified.

    Returns:
        Dynamic instruction string to append to the base SYSTEM_PROMPT.
    """
    lang_label = _LANG_LABELS.get(lang, lang.upper())
    instructions: list[str] = []

    instructions.append("────────────────────────")
    instructions.append("DYNAMIC REQUEST INSTRUCTIONS (THIS REQUEST ONLY)")
    instructions.append("────────────────────────")

    # ── Language instruction ──
    if lang == "ar":
        dialect_label = _DIALECT_LABELS.get(dialect or "unknown", "Arabic")
        instructions.append(f"✅ Detected language: Arabic — Dialect: {dialect_label}")
        instructions.append(f"👉 You MUST respond in: {dialect_label}")
        instructions.append("   Use natural spoken vocabulary of this dialect. Do NOT switch to MSA or another dialect.")
    else:
        instructions.append(f"✅ Detected language: {lang_label}")
        instructions.append(f"👉 You MUST respond entirely in: {lang_label}")

    # ── Intent instruction ──
    if intent == "rephrase_request":
        target = rephrase_target or (
            _DIALECT_LABELS.get(dialect or "unknown") if lang == "ar" else lang_label
        )
        instructions.append("")
        instructions.append("🔄 REPHRASE MODE — CRITICAL:")
        instructions.append(f"   Rewrite the PREVIOUS assistant answer in: {target}")
        instructions.append("   ⛔ Do NOT add any new information.")
        instructions.append("   ⛔ Do NOT search for new content.")
        instructions.append("   ⛔ Do NOT explain or comment on the rephrasing.")
        instructions.append("   ✅ Output ONLY the rephrased version of the previous answer.")
    else:
        instructions.append("")
        instructions.append("🔍 NEW QUESTION MODE:")
        instructions.append("   Answer based on the provided context only.")
        instructions.append("   If the context does not contain the answer, say so briefly.")

    # ── Length constraint (always applied) ──
    instructions.append("")
    instructions.append("📏 RESPONSE LENGTH — STRICT RULE:")
    instructions.append("   ⛔ Maximum 3 to 4 lines ONLY.")
    instructions.append("   ⛔ No essays, no bullet lists with 5+ items, no long explanations.")
    instructions.append("   ✅ Be direct. Be concise. Answer the question only.")
    instructions.append("   ✅ If you need a list, maximum 3 items.")
    instructions.append("────────────────────────")

    return "\n".join(instructions)


# ─────────────────────────────────────────────────────────────
# 5. Post-processing
# ─────────────────────────────────────────────────────────────

# Matches "FODWA" in any casing, possibly already wrapped in LRM markers
_FODWA_PATTERN = re.compile(
    r"(?<!\u200E)\b(FODWA|Fodwa|fodwa|AWDOF|awdof)\b(?!\u200E)",
    re.IGNORECASE,
)


def post_process_response(text: str) -> str:
    """
    Apply post-processing to LLM output before returning to the user.

    Operations (in order):
      1. Fix FODWA RTL rendering bug by wrapping with LRM markers.
      2. Hard-truncate response if it exceeds allowed line/char limits.

    Args:
        text: Raw LLM response string.

    Returns:
        Cleaned, safe-to-display response string.
    """
    if not text:
        return text

    # ── Step 1: Fix FODWA RTL bug ──────────────────────────
    # Replace any variant (FODWA / Fodwa / AWDOF) with LRM-wrapped canonical form
    fixed = _FODWA_PATTERN.sub(f"{_LRM}FODWA{_LRM}", text)

    if fixed != text:
        logger.debug("[NLP] post_process: FODWA RTL fix applied")

    # ── Step 2: Hard truncation (safety net) ────────────────
    lines = fixed.splitlines()

    # Filter empty-only trailing lines
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) > _MAX_LINES:
        logger.warning(
            f"[NLP] post_process: Response exceeded {_MAX_LINES} lines "
            f"({len(lines)} lines). Truncating."
        )
        truncated = "\n".join(lines[:_MAX_LINES])
        # Append ellipsis only if the cut line wasn't already ending with punctuation
        last_char = truncated.rstrip()[-1] if truncated.rstrip() else ""
        if last_char not in (".", "!", "?", "…", "،", "؟", "؛"):
            truncated += "…"
        fixed = truncated

    # Secondary character limit check
    if len(fixed) > _MAX_CHARS:
        logger.warning(
            f"[NLP] post_process: Response exceeded {_MAX_CHARS} chars. Truncating."
        )
        fixed = fixed[:_MAX_CHARS].rstrip() + "…"

    return fixed


# ─────────────────────────────────────────────────────────────
# 6. Convenience: Full Pre-processing Pipeline
# ─────────────────────────────────────────────────────────────

class NLPMetadata:
    """Structured result from the NLP pre-processing pipeline."""

    __slots__ = ("language", "dialect", "intent")

    def __init__(self, language: str, dialect: str | None, intent: str):
        self.language = language
        self.dialect = dialect
        self.intent = intent

    def __repr__(self) -> str:
        return (
            f"NLPMetadata(language={self.language!r}, "
            f"dialect={self.dialect!r}, intent={self.intent!r})"
        )


def preprocess(text: str, history: list[dict] | None = None) -> NLPMetadata:
    """
    Run the full NLP pre-processing pipeline on a user message.

    Args:
        text:    User's raw message.
        history: Current conversation history list.

    Returns:
        NLPMetadata with language, dialect, and intent.
    """
    lang = detect_language(text)
    dialect = detect_arabic_dialect(text) if lang == "ar" else None
    intent = detect_intent(text, history)

    # ── Structured debug log ──
    logger.info(
        f"[NLP] Pre-processing | language={lang!r} | dialect={dialect!r} | intent={intent!r}"
    )

    return NLPMetadata(language=lang, dialect=dialect, intent=intent)
