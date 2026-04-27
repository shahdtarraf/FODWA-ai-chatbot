"""
NLP Utilities — Production-grade Pre/Post-processing for FODWA Chatbot.

v2 improvements over v1:
  - detect_language_with_confidence() → (lang, confidence)
    * Short text (<15 chars) uses heuristic only
    * Low confidence flagged for GPT fallback
  - detect_arabic_dialect() → (dialect, confidence)
    * Expanded keyword coverage + weighted scoring
    * Confidence = top_score / total_score
  - detect_intent() → (intent, confidence)
    * Distinguishes "translation request" vs "language mention"
    * Context-aware: requires both pattern match AND prior history
    * Confidence based on pattern strength
  - post_process_response() — sentence-aware truncation (not line-based)
  - NLPMetadata — now carries confidence scores for all three fields
  - Structured logging: lang=ar(0.92) dialect=egyptian(0.81) intent=rephrase(0.95)

Design rules (unchanged):
  - ZERO dependencies on chat_service / faiss_service / openai_service
  - Never raises — always returns safe defaults
"""

import re
import logging

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

_MAX_SENTENCES = 3       # max sentences in truncated response
_MAX_CHARS = 600         # hard character cap (safety net)

# Confidence thresholds
_LANG_CONFIDENCE_MIN = 0.70    # below → treat as uncertain
_DIALECT_CONFIDENCE_MIN = 0.40
_INTENT_CONFIDENCE_MIN = 0.60

# Short text threshold — below this, langdetect is unreliable
_SHORT_TEXT_CHARS = 15


# ──────────────────────────────────────────────
# 1. Language Detection  (with confidence)
# ──────────────────────────────────────────────

def _arabic_ratio(text: str) -> float:
    """Fraction of Arabic Unicode characters in text."""
    if not text:
        return 0.0
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return ar / len(text)


def _latin_ratio(text: str) -> float:
    """Fraction of Latin characters in text."""
    if not text:
        return 0.0
    lat = sum(1 for c in text if "A" <= c.upper() <= "Z")
    return lat / len(text)


def detect_language_with_confidence(text: str) -> tuple[str, float]:
    """
    Detect language and return (lang_code, confidence 0-1).

    Strategy:
      1. Empty → ('en', 1.0)
      2. Arabic ratio > 0.30 → ('ar', ratio clamped to 1.0)  [fast heuristic]
      3. Short text (<15 chars):
           - Latin ratio > 0.5 → ('en', 0.65)  [heuristic]
           - else → ('en', 0.5)
      4. langdetect.detect_langs() for full confidence
      5. Fallback → ('en', 0.5)
    """
    stripped = (text or "").strip()
    if not stripped:
        return "en", 1.0

    ar_ratio = _arabic_ratio(stripped)
    if ar_ratio > 0.30:
        conf = min(ar_ratio * 1.2, 1.0)
        logger.debug(f"[NLP] lang=ar via heuristic (conf={conf:.2f})")
        return "ar", round(conf, 2)

    if len(stripped) < _SHORT_TEXT_CHARS:
        if _latin_ratio(stripped) > 0.5:
            return "en", 0.65
        return "en", 0.50

    try:
        from langdetect import detect_langs
        results = detect_langs(stripped)
        if results:
            top = results[0]
            logger.debug(f"[NLP] langdetect → {top.lang}({top.prob:.2f})")
            return top.lang, round(top.prob, 2)
    except Exception as exc:
        logger.warning(f"[NLP] langdetect failed: {exc}")

    return "en", 0.50


def detect_language(text: str) -> str:
    """Backwards-compatible wrapper — returns lang code only."""
    lang, _ = detect_language_with_confidence(text)
    return lang


# ──────────────────────────────────────────────
# 2. Arabic Dialect Detection  (with confidence)
# ──────────────────────────────────────────────

# Each entry: (dialect, [(keyword, weight), ...])
# Weight 2 = strong marker, 1 = common word
_DIALECT_KEYWORDS: list[tuple[str, list[tuple[str, int]]]] = [
    ("egyptian", [
        ("ايه ده", 2), ("عامل ايه", 2), ("ازيك", 2), ("امتى", 2),
        ("عايز", 2), ("عايزة", 2), ("مش عارف", 2), ("بقى", 2),
        ("كده", 2), ("فين", 1), ("ليه", 1), ("ده", 1), ("دي", 1),
        ("ايه", 1), ("بقا", 1), ("اللي", 1), ("هو ايه", 2),
        ("مش عارفة", 2), ("ازيكم", 2), ("عامله ايه", 2),
    ]),
    ("syrian", [
        ("شو", 2), ("هلق", 2), ("عنجد", 2), ("منيح", 2), ("هيك", 2),
        ("ما في", 2), ("في شي", 2), ("ليش", 2), ("هون", 2),
        ("شو بدك", 2), ("شو بدي", 2), ("شلونك", 1), ("كيفك", 1),
        ("هناك", 1), ("شو صار", 2), ("رح", 1),
    ]),
    ("iraqi", [
        ("شگول", 2), ("شنو", 2), ("هواية", 2), ("گاعد", 2), ("چي", 2),
        ("أكو", 2), ("ماكو", 2), ("شبيك", 2), ("يمه", 2), ("أبه", 2),
        ("گلتلك", 2), ("شلونك", 1), ("بعد شوية", 2),
    ]),
    ("gulf", [
        ("وش", 2), ("ايش", 2), ("إيش", 2), ("وايد", 2), ("زين", 2),
        ("شلون", 2), ("حيل", 2), ("عيل", 2), ("يبه", 2), ("ما ادري", 2),
        ("ترى", 2), ("يمه", 1), ("ليش", 1), ("كيف حالك", 1),
        ("إن شاء الله", 1), ("والله", 1), ("ما قلت", 2),
    ]),
    ("moroccan", [
        ("واش", 2), ("كيداير", 2), ("كيف داير", 2), ("بزاف", 2),
        ("ماشي", 2), ("دابا", 2), ("هاد", 2), ("بغيت", 2),
        ("مزيان", 2), ("لا باس", 2), ("فين غادي", 2), ("شنو", 1),
    ]),
    ("tunisian", [
        ("شنية", 2), ("برشة", 2), ("ياسر", 2), ("فما", 2), ("نحب", 2),
        ("نعرف", 2), ("كيفاش", 2), ("علاش", 2), ("شنو", 1),
        ("باهي", 2), ("نحبش", 2),
    ]),
    ("algerian", [
        ("واش راك", 2), ("راك", 2), ("نتا", 2), ("نتي", 2),
        ("كيراك", 2), ("رانا", 2), ("دروك", 2), ("تاع", 2),
        ("بلاك", 2), ("يصح", 2),
    ]),
    ("libyan", [
        ("شحالك", 2), ("شحالكم", 2), ("أشحال", 2), ("شكون", 2),
        ("كيفاش", 1), ("بهي", 2), ("مرحبا", 1),
    ]),
    ("sudanese", [
        ("يا زول", 2), ("زول", 2), ("ما علينا", 2), ("ما قادر", 2),
        ("زين", 1), ("كيفك", 1), ("طيب", 1),
    ]),
    ("yemeni", [
        ("شو", 1), ("ايش", 1), ("وش", 1), ("ما اعرف", 1),
        ("هيه", 2), ("تعال", 1), ("عشان", 1),
    ]),
    ("jordanian", [
        ("هلأ", 2), ("يعني إيش", 2), ("شو صار", 2), ("والله", 1),
        ("شو بدك", 1), ("بدي", 1), ("هيك", 1),
    ]),
    ("levantine", [
        ("عم", 2), ("رح", 2), ("ما بدي", 2), ("بدي", 1),
        ("هلأ", 1), ("شو", 1), ("كيفك", 1),
    ]),
    ("msa", [
        ("هل يمكنني", 2), ("أود أن", 2), ("أرجو", 2), ("يرجى", 2),
        ("من فضلك", 1), ("تفضل", 1), ("يمكن", 1),
    ]),
]


def detect_arabic_dialect(text: str) -> tuple[str, float]:
    """
    Detect Arabic dialect.

    Returns:
        (dialect_label, confidence) where confidence in [0, 1].
        dialect_label one of: egyptian, syrian, iraqi, gulf, moroccan,
        tunisian, algerian, libyan, sudanese, yemeni, jordanian,
        levantine, msa, unknown.
    """
    if not text or not text.strip():
        return "unknown", 0.0

    t = text.lower().strip()
    scores: dict[str, int] = {}
    total_weight = 0

    for dialect, kw_list in _DIALECT_KEYWORDS:
        score = sum(w for kw, w in kw_list if kw in t)
        if score > 0:
            scores[dialect] = score
            total_weight += score

    if not scores:
        logger.debug("[NLP] dialect=unknown (no keywords matched)")
        return "unknown", 0.0

    best = max(scores, key=lambda d: scores[d])
    confidence = round(scores[best] / max(total_weight, 1), 2)
    logger.debug(f"[NLP] dialect={best}({confidence:.2f}) scores={scores}")
    return best, confidence


# ──────────────────────────────────────────────
# 3. Intent Detection  (with confidence)
# ──────────────────────────────────────────────

# (pattern, weight) — higher weight = stronger rephrase signal
_REPHRASE_PATTERNS: list[tuple[re.Pattern, float]] = [
    # Strong: explicit translate/rephrase verb + target
    (re.compile(r"(اكتب(ها|ه|لي)?\s*(بالانجليزي|بالعربي|بالألماني|بالفرنسي|بالتركي))", re.I), 1.0),
    (re.compile(r"(حولها|حوله|اكتبها|قولها|قله)\s*(بالمصري|بالسوري|بالعراقي|بالخليجي|بالمغربي|بالتونسي|بالجزائري|بالليبي|بالأردني|بالشامي|بالفصحى|بالعامية)", re.I), 1.0),
    (re.compile(r"(حولها|حوله|اكتبها)\s*ل(لمصري|لسوري|لعراقي|لخليجي|لمغربي|لتونسي|لجزائري|لليبي|لأردني|لشامي|للفصحى|للعامية)", re.I), 1.0),
    (re.compile(r"(translate\s+(to|into)\s+\w+|ترجملي|ترجم\s+ل)", re.I), 1.0),
    (re.compile(r"(قلها|قوله|قولها)\s*(بالانجليزي|بالعربي|بالألماني|بالفرنسي)", re.I), 1.0),
    (re.compile(r"\b(auf\s+deutsch|en\s+français|in\s+english)\b", re.I), 0.9),
    # Medium: just the language/dialect name (can be ambiguous)
    (re.compile(r"^(بالإنجليزي|بالانجليزي|بالإنكليزي)$", re.I), 0.9),
    (re.compile(r"^(بالعربي|بالعربية)$", re.I), 0.85),
    (re.compile(r"^(بالألماني|بالفرنسي|بالتركي)$", re.I), 0.85),
    (re.compile(r"^(بالمصري|بالسوري|بالعراقي|بالخليجي|بالمغربي)$", re.I), 0.85),
    (re.compile(r"^(بالفصحى|بالعامية|بالشامي|بالأردني)$", re.I), 0.85),
    # Medium: rephrase without translation
    (re.compile(r"(أعد(ها|ه)?\s*(بكلمات|بأسلوب|بطريقة)\s*أبسط)", re.I), 0.9),
    (re.compile(r"(بشكل\s*أبسط|بكلمات\s*أبسط|بطريقة\s*أسهل)", re.I), 0.85),
    (re.compile(r"\b(simplify|rephrase|rewrite|say\s+it\s+(again|differently))\b", re.I), 0.9),
    # Weaker: bare language mention inside a longer sentence
    (re.compile(r"(بالإنجليزي|بالانجليزي|بالعربي|بالألماني|بالفرنسي|بالتركي)", re.I), 0.65),
    (re.compile(r"(بالمصري|بالمصرية|بالسوري|بالعراقي|بالخليجي|بالمغربي)", re.I), 0.65),
    (re.compile(r"(بالفصحى|بالعربية الفصحى|بالعامية)", re.I), 0.65),
    (re.compile(r"\b(translate|ترجم|ترجمة)\b", re.I), 0.70),
]


def detect_intent(text: str, history: list[dict] | None = None) -> tuple[str, float]:
    """
    Detect intent and return (intent_label, confidence).

    Rules:
      - rephrase_request only if: pattern matches + prior assistant msg exists
        + message is short (<= 120 chars)
      - new_question otherwise
    """
    if not text or not text.strip():
        return "new_question", 1.0

    stripped = text.strip()

    has_prior = any(m.get("role") == "assistant" for m in (history or []))

    if has_prior and len(stripped) <= 120:
        best_conf = 0.0
        for pattern, weight in _REPHRASE_PATTERNS:
            if pattern.search(stripped):
                if weight > best_conf:
                    best_conf = weight
        if best_conf >= _INTENT_CONFIDENCE_MIN:
            logger.debug(f"[NLP] intent=rephrase_request({best_conf:.2f})")
            return "rephrase_request", round(best_conf, 2)

    return "new_question", 1.0


# ──────────────────────────────────────────────
# 4. Dynamic System Prompt Builder
# ──────────────────────────────────────────────

_DIALECT_LABELS: dict[str, str] = {
    "egyptian":  "Egyptian Arabic (عامية مصرية)",
    "syrian":    "Syrian Arabic (لهجة سورية شامية)",
    "iraqi":     "Iraqi Arabic (لهجة عراقية)",
    "gulf":      "Gulf Arabic (لهجة خليجية)",
    "moroccan":  "Moroccan Darija (الدارجة المغربية)",
    "tunisian":  "Tunisian Arabic (اللهجة التونسية)",
    "algerian":  "Algerian Arabic (اللهجة الجزائرية)",
    "libyan":    "Libyan Arabic (اللهجة الليبية)",
    "sudanese":  "Sudanese Arabic (اللهجة السودانية)",
    "yemeni":    "Yemeni Arabic (اللهجة اليمنية)",
    "jordanian": "Jordanian Arabic (اللهجة الأردنية)",
    "levantine": "Levantine Arabic (اللهجة الشامية)",
    "msa":       "Modern Standard Arabic (العربية الفصحى)",
    "unknown":   "Arabic (detect dialect from context)",
}

_LANG_LABELS: dict[str, str] = {
    "ar": "Arabic", "en": "English", "de": "German", "fr": "French",
    "tr": "Turkish", "es": "Spanish", "it": "Italian", "nl": "Dutch",
    "pl": "Polish", "ru": "Russian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "fa": "Persian", "ur": "Urdu",
}


def build_dynamic_system_prompt(
    lang: str,
    dialect: str | None = None,
    intent: str = "new_question",
    lang_conf: float = 1.0,
    dialect_conf: float = 1.0,
    intent_conf: float = 1.0,
    rephrase_target: str | None = None,
) -> str:
    """
    Build the dynamic instruction block to append to BASE_SYSTEM_PROMPT.

    Low-confidence signals are communicated to GPT so it can compensate.
    """
    lang_label = _LANG_LABELS.get(lang, lang.upper())
    lines: list[str] = [
        "────────────────────────",
        "DYNAMIC REQUEST INSTRUCTIONS (THIS REQUEST ONLY)",
        "────────────────────────",
    ]

    # ── Language ──
    if lang == "ar":
        dlabel = _DIALECT_LABELS.get(dialect or "unknown", "Arabic")
        lines.append(f"✅ Detected: Arabic — Dialect: {dlabel} (conf={lang_conf:.0%}/{dialect_conf:.0%})")
        if dialect_conf < _DIALECT_CONFIDENCE_MIN:
            lines.append("⚠️  Dialect confidence LOW — infer from message vocabulary.")
        lines.append(f"👉 Respond in: {dlabel}")
        lines.append("   Natural spoken vocab only. Do NOT switch to MSA or another dialect.")
    else:
        lines.append(f"✅ Detected: {lang_label} (conf={lang_conf:.0%})")
        if lang_conf < _LANG_CONFIDENCE_MIN:
            lines.append("⚠️  Language confidence LOW — infer from message vocabulary.")
        lines.append(f"👉 Respond entirely in: {lang_label}")

    # ── Intent ──
    if intent == "rephrase_request":
        target = rephrase_target or (
            _DIALECT_LABELS.get(dialect or "unknown") if lang == "ar" else lang_label
        )
        lines += [
            "", "🔄 REPHRASE MODE — CRITICAL:",
            f"   Rewrite the PREVIOUS assistant answer in: {target}",
            "   ⛔ Do NOT add new information.",
            "   ⛔ Do NOT search for new content.",
            "   ✅ Output ONLY the rephrased version.",
        ]
    else:
        lines += [
            "", "🔍 NEW QUESTION MODE:",
            "   Answer based on provided context only.",
            "   If context lacks the answer, say so briefly.",
        ]

    # ── Length (always) ──
    lines += [
        "", "📏 RESPONSE LENGTH — STRICT:",
        "   ⛔ Max 3–4 lines. No long explanations.",
        "   ✅ Direct. Concise. Answer only.",
        "   ✅ Lists: max 3 items.",
        "────────────────────────",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 5. Post-processing  (sentence-aware truncation)
# ──────────────────────────────────────────────



# Sentence boundary: ends with . ! ? … ؟ ؛ ، followed by space/newline/end
_SENT_END = re.compile(r"[.!?…؟؛،]\s*")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving delimiters."""
    parts = _SENT_END.split(text)
    delims = _SENT_END.findall(text)
    sentences = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            delim = delims[i].strip() if i < len(delims) else ""
            sentences.append(part + (delim if delim else ""))
    return [s for s in sentences if s.strip()]


def clean_text(text: str) -> str:
    """Strip hidden Unicode direction characters and normalize whitespace."""
    if not text:
        return text
    text = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def post_process_response(text: str) -> str:
    """
    Post-process LLM output:
      1. Strip all hidden unicode directional markers + normalize spaces.
      2. Sentence-aware truncation to _MAX_SENTENCES.
      3. Hard character cap as safety net.
    """
    if not text:
        return text

    # Step 1: Clean hidden chars and spaces
    cleaned = clean_text(text)

    # Step 2: Sentence-based truncation
    sentences = _split_sentences(cleaned)
    if len(sentences) > _MAX_SENTENCES:
        logger.warning(
            f"[NLP] post_process: {len(sentences)} sentences → truncating to {_MAX_SENTENCES}"
        )
        kept = sentences[:_MAX_SENTENCES]
        cleaned = " ".join(kept)
        last = cleaned.rstrip()[-1] if cleaned.rstrip() else ""
        if last not in (".", "!", "?", "…", "،", "؟", "؛"):
            cleaned = cleaned.rstrip() + "…"

    # Step 3: Hard char cap
    if len(cleaned) > _MAX_CHARS:
        logger.warning(f"[NLP] post_process: {len(cleaned)} chars → hard cap {_MAX_CHARS}")
        cleaned = cleaned[:_MAX_CHARS].rstrip() + "…"

    return cleaned


# ──────────────────────────────────────────────
# 6. NLPMetadata + preprocess() pipeline
# ──────────────────────────────────────────────

class NLPMetadata:
    """Structured result from the NLP pre-processing pipeline (v2)."""

    __slots__ = (
        "language", "lang_confidence",
        "dialect", "dialect_confidence",
        "intent", "intent_confidence",
    )

    def __init__(
        self,
        language: str, lang_confidence: float,
        dialect: str | None, dialect_confidence: float,
        intent: str, intent_confidence: float,
    ):
        self.language = language
        self.lang_confidence = lang_confidence
        self.dialect = dialect
        self.dialect_confidence = dialect_confidence
        self.intent = intent
        self.intent_confidence = intent_confidence

    @property
    def lang_uncertain(self) -> bool:
        return self.lang_confidence < _LANG_CONFIDENCE_MIN

    @property
    def dialect_uncertain(self) -> bool:
        return self.dialect_confidence < _DIALECT_CONFIDENCE_MIN

    @property
    def intent_uncertain(self) -> bool:
        return self.intent_confidence < _INTENT_CONFIDENCE_MIN

    def __repr__(self) -> str:
        return (
            f"NLPMetadata("
            f"lang={self.language!r}({self.lang_confidence:.2f}), "
            f"dialect={self.dialect!r}({self.dialect_confidence:.2f}), "
            f"intent={self.intent!r}({self.intent_confidence:.2f}))"
        )


def preprocess(text: str, history: list[dict] | None = None) -> NLPMetadata:
    """
    Run the full NLP pre-processing pipeline.

    Returns NLPMetadata with language, dialect, intent + confidence scores.
    Structured log format: lang=ar(0.92) dialect=egyptian(0.81) intent=rephrase(0.95)
    """
    lang, lang_conf = detect_language_with_confidence(text)

    if lang == "ar":
        dialect, dialect_conf = detect_arabic_dialect(text)
    else:
        dialect, dialect_conf = None, 0.0

    intent, intent_conf = detect_intent(text, history)

    # Structured production log
    dialect_part = f"dialect={dialect}({dialect_conf:.2f}) " if dialect else ""
    logger.info(
        f"[NLP] lang={lang}({lang_conf:.2f}) "
        f"{dialect_part}"
        f"intent={intent}({intent_conf:.2f})"
    )

    return NLPMetadata(
        language=lang, lang_confidence=lang_conf,
        dialect=dialect, dialect_confidence=dialect_conf,
        intent=intent, intent_confidence=intent_conf,
    )
