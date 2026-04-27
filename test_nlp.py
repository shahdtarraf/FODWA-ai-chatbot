"""
Tests for chatbot/utils/nlp_utils.py  (v2 — production-grade)

Run with:
    $env:PYTHONIOENCODING="utf-8"; python test_nlp.py
Or:
    $env:PYTHONIOENCODING="utf-8"; python -m pytest test_nlp.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fodwa_project.settings")

import django
try:
    django.setup()
except Exception:
    pass

from chatbot.utils.nlp_utils import (
    detect_language_with_confidence,
    detect_language,
    detect_arabic_dialect,
    detect_intent,
    build_dynamic_system_prompt,
    post_process_response,
    preprocess,
    NLPMetadata,
    _LANG_CONFIDENCE_MIN,
    _DIALECT_CONFIDENCE_MIN,
    _INTENT_CONFIDENCE_MIN,
)

_PASSED: list[str] = []
_FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    sym = "PASS" if ok else "FAIL"
    print(f"  [{sym}] {name}" + (f"\n          -> {detail}" if detail else ""))
    (_PASSED if ok else _FAILED).append(name)


# ═══════════════════════════════════════════════════════════
# 1. Language Detection with Confidence
# ═══════════════════════════════════════════════════════════

def test_language_confidence():
    print("\n--- 1. Language Detection + Confidence ---")

    cases = [
        ("كيف يمكنني حذف إعلاني؟",           "ar",  0.7),
        ("How do I delete my listing?",       "en",  0.3),  # langdetect varies wildly on short EN
        ("Wie kann ich mein Konto löschen?",  "de",  0.7),
        ("Comment puis-je supprimer mon annonce sur ce site?", "fr", 0.7),  # Longer FR text
        ("İlanımı nasıl silebilirim?",        "tr",  0.7),
        ("",                                  "en",  0.9),
    ]
    for text, exp_lang, min_conf in cases:
        lang, conf = detect_language_with_confidence(text)
        check(
            f"detect_language_with_confidence({text[:30]!r})",
            lang == exp_lang and conf >= min_conf,
            f"lang={lang!r} conf={conf:.2f} (expected lang={exp_lang!r} conf>={min_conf})"
        )


def test_short_text_heuristic():
    print("\n--- 1b. Short Text Heuristic (<15 chars) ---")

    # Very short Arabic → still ar
    lang, conf = detect_language_with_confidence("ايه ده")
    check("Short Arabic text → ar", lang == "ar", f"got lang={lang!r} conf={conf:.2f}")

    # Very short Latin text → en (heuristic)
    lang2, conf2 = detect_language_with_confidence("hi")
    check("Short Latin 'hi' → en", lang2 == "en", f"got lang={lang2!r} conf={conf2:.2f}")
    check("Short text conf < 0.90 (uncertain)", conf2 < 0.90, f"conf={conf2:.2f}")

    # Empty
    lang3, conf3 = detect_language_with_confidence("")
    check("Empty text → en conf=1.0", lang3 == "en" and conf3 == 1.0, f"lang={lang3!r} conf={conf3:.2f}")


# ═══════════════════════════════════════════════════════════
# 2. Arabic Dialect Detection with Confidence
# ═══════════════════════════════════════════════════════════

def test_dialect_confidence():
    print("\n--- 2. Dialect Detection + Confidence ---")

    cases = [
        ("ايه ده؟ عايز أعرف إزاي أحذف الإعلان بتاعي", "egyptian"),
        ("شو هلق؟ عنجد ما في مشكلة",                   "syrian"),
        ("شنو هواية ماكو وقت",                           "iraqi"),
        ("وش المشكلة؟ وايد أسئلة يا زين",               "gulf"),
        ("واش كيداير؟ بزاف مشاكل دابا",                  "moroccan"),
        ("شنية برشة ياسر",                               "tunisian"),
        ("واش راك؟ دروك رانا هنا",                       "algerian"),
        ("أود أن أستفسر عن هذه المسألة يرجى",           "msa"),
        ("hello world",                                   "unknown"),
    ]
    for text, expected in cases:
        dialect, conf = detect_arabic_dialect(text)
        check(
            f"detect_arabic_dialect({text[:35]!r})",
            dialect == expected,
            f"expected={expected!r} got={dialect!r} conf={conf:.2f}"
        )


def test_dialect_confidence_score():
    print("\n--- 2b. Dialect Confidence Scores ---")

    # Rich text → high confidence
    _, conf_rich = detect_arabic_dialect(
        "ايه ده؟ عايز ايه بقى؟ مش عارف فين الحاجة دي"
    )
    check("Rich Egyptian text → conf >= 0.4", conf_rich >= 0.4, f"conf={conf_rich:.2f}")

    # Ambiguous → low or unknown
    _, conf_ambig = detect_arabic_dialect("والله")
    check("Single ambiguous word → conf <= 0.5", conf_ambig <= 0.5, f"conf={conf_ambig:.2f}")

    # Empty
    d, c = detect_arabic_dialect("")
    check("Empty text → unknown conf=0.0", d == "unknown" and c == 0.0)


# ═══════════════════════════════════════════════════════════
# 3. Intent Detection with Confidence
# ═══════════════════════════════════════════════════════════

def test_intent_confidence():
    print("\n--- 3. Intent Detection + Confidence ---")

    hist = [
        {"role": "user", "content": "كيف أحذف الإعلان؟"},
        {"role": "assistant", "content": "يمكنك حذف الإعلان من لوحة التحكم."},
    ]
    no_hist: list = []

    strong_rephrase = [
        ("اكتبها بالانجليزي",       1.0),
        ("اكتبه بالعربي",           1.0),
        ("حولها للمصري",            0.65),  # matches bare dialect pattern
        ("in English",              0.9),
        ("auf Deutsch",             0.9),
        ("translate to French",     1.0),
        ("simplify it",             0.9),
        ("بشكل أبسط",              0.85),
    ]
    for text, min_conf in strong_rephrase:
        intent, conf = detect_intent(text, hist)
        check(
            f"rephrase: {text!r}",
            intent == "rephrase_request" and conf >= min_conf,
            f"intent={intent!r} conf={conf:.2f}"
        )

    # Bare language name → rephrase only if history present
    intent, conf = detect_intent("بالمصري", hist)
    check("'بالمصري' with history → rephrase", intent == "rephrase_request", f"conf={conf:.2f}")

    intent, conf = detect_intent("بالمصري", no_hist)
    check("'بالمصري' NO history → new_question", intent == "new_question")

    # Real questions → new_question
    new_qs = [
        "كيف يمكنني حذف إعلاني؟",
        "What is FODWA?",
        "Wie registriere ich mich?",
    ]
    for q in new_qs:
        intent, conf = detect_intent(q, hist)
        check(f"new_question: {q!r}", intent == "new_question", f"conf={conf:.2f}")

    # Empty
    intent, conf = detect_intent("", hist)
    check("Empty text → new_question", intent == "new_question")


def test_intent_long_text_not_rephrase():
    print("\n--- 3b. Long text never rephrase ---")
    hist = [{"role": "assistant", "content": "..."}]
    long_text = "بالمصري " * 20  # >120 chars
    intent, conf = detect_intent(long_text, hist)
    check("Long text (>120) → new_question even if pattern matches", intent == "new_question")


# ═══════════════════════════════════════════════════════════
# 4. build_dynamic_system_prompt
# ═══════════════════════════════════════════════════════════

def test_build_prompt():
    print("\n--- 4. build_dynamic_system_prompt ---")

    p = build_dynamic_system_prompt("en", intent="new_question", lang_conf=0.95)
    check("English prompt", "English" in p)
    check("Length rule present", "3" in p and "4" in p)
    check("Conf in prompt", "95%" in p)

    p2 = build_dynamic_system_prompt("ar", dialect="egyptian", intent="rephrase_request",
                                     lang_conf=0.88, dialect_conf=0.75, intent_conf=0.95)
    check("Arabic+Egyptian dialect label", "Egyptian Arabic" in p2)
    check("Rephrase mode", "REPHRASE MODE" in p2)
    check("No new info rule", "Do NOT add new information" in p2)

    # Low dialect confidence warning
    p3 = build_dynamic_system_prompt("ar", dialect="unknown", intent="new_question",
                                     lang_conf=0.95, dialect_conf=0.20)
    check("Low dialect conf → warning in prompt", "LOW" in p3)

    # Low lang confidence warning
    p4 = build_dynamic_system_prompt("en", intent="new_question", lang_conf=0.50)
    check("Low lang conf → warning in prompt", "LOW" in p4)

    check("Separator present", "---" in p or "────" in p)


# ═══════════════════════════════════════════════════════════
# 5. post_process_response — clean_text
# ═══════════════════════════════════════════════════════════

def test_clean_text():
    print("\n--- 5. Clean Text (No Hidden Unicode) ---")
    cases = [
        ("Hello \u200eFODWA\u200e", "Hello FODWA"),
        ("مرحبا \u202bبكم\u202c", "مرحبا بكم"),
        ("Spaces    are\n\nnormalized", "Spaces are normalized"),
        ("", ""),
    ]
    for text, expected in cases:
        result = post_process_response(text)
        check(f"Clean: {text!r}", result == expected, f"result={result!r}")


# ═══════════════════════════════════════════════════════════
# 6. post_process_response — Sentence-based Truncation
# ═══════════════════════════════════════════════════════════

def test_sentence_truncation():
    print("\n--- 6. Sentence-based Truncation ---")

    # Short (1 sentence) → unchanged
    short = "يمكنك حذف الإعلان من لوحة التحكم."
    check("Short (1 sent) → unchanged", post_process_response(short) == short)

    # 3 sentences → unchanged (at limit)
    three = "الجملة الأولى. الجملة الثانية. الجملة الثالثة."
    result = post_process_response(three)
    check("3 sentences → not truncated", "…" not in result, f"result={result!r}")

    # 5 sentences → truncated
    five = "الجملة الأولى. الجملة الثانية. الجملة الثالثة. الجملة الرابعة. الجملة الخامسة."
    result2 = post_process_response(five)
    check("5 sentences → truncated", "الجملة الرابعة" not in result2, f"result={result2!r}")
    check("Truncated ends with ellipsis or punctuation",
          result2.endswith("…") or result2[-1] in ".!?؟")

    # Hard char cap
    long_text = "A" * 700
    result3 = post_process_response(long_text)
    check(f"Hard cap: {len(long_text)} chars → <= 600+3", len(result3) <= 603, f"len={len(result3)}")


# ═══════════════════════════════════════════════════════════
# 7. NLPMetadata confidence properties
# ═══════════════════════════════════════════════════════════

def test_nlp_metadata():
    print("\n--- 7. NLPMetadata ---")

    meta = NLPMetadata(
        language="ar", lang_confidence=0.95,
        dialect="egyptian", dialect_confidence=0.80,
        intent="new_question", intent_confidence=1.0,
    )
    check("lang_uncertain False (0.95)", not meta.lang_uncertain)
    check("dialect_uncertain False (0.80)", not meta.dialect_uncertain)
    check("intent_uncertain False (1.0)", not meta.intent_uncertain)

    meta2 = NLPMetadata(
        language="en", lang_confidence=0.55,
        dialect=None, dialect_confidence=0.0,
        intent="rephrase_request", intent_confidence=0.50,
    )
    check("lang_uncertain True (0.55)", meta2.lang_uncertain)
    check("intent_uncertain True (0.50)", meta2.intent_uncertain)
    check("repr contains conf", "0.55" in repr(meta2))


# ═══════════════════════════════════════════════════════════
# 8. Full preprocess() Pipeline
# ═══════════════════════════════════════════════════════════

def test_preprocess_pipeline():
    print("\n--- 8. preprocess() pipeline ---")

    hist = [{"role": "assistant", "content": "يمكنك حذف الإعلان من لوحة التحكم."}]

    # German new question
    m = preprocess("Wie kann ich mein Konto löschen?", [])
    check("German lang=de", m.language == "de", f"got {m.language!r}")
    check("German dialect=None", m.dialect is None)
    check("German intent=new_question", m.intent == "new_question")
    check("German has conf", m.lang_confidence > 0)

    # Arabic Egyptian rephrase — use a message that has Egyptian keywords
    m2 = preprocess("اكتبها بالمصري عايز ده", hist)
    check("Arabic lang=ar", m2.language == "ar")
    check("Egyptian dialect detected", m2.dialect == "egyptian", f"got {m2.dialect!r}")
    check("Rephrase intent", m2.intent == "rephrase_request")
    check("Intent conf >= 0.60", m2.intent_confidence >= 0.60, f"conf={m2.intent_confidence:.2f}")

    # Returns NLPMetadata
    check("Returns NLPMetadata", isinstance(m, NLPMetadata))

    # Short ambiguous text
    m3 = preprocess("hi", [])
    check("Short 'hi' → en", m3.language == "en")
    check("Short 'hi' conf < 0.90", m3.lang_confidence < 0.90, f"conf={m3.lang_confidence:.2f}")


# ═══════════════════════════════════════════════════════════
# 9. Edge Cases
# ═══════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n--- 9. Edge Cases ---")

    # Mixed Arabic+English
    mixed = "كيف أعمل upload للصورة on FODWA?"
    lang, conf = detect_language_with_confidence(mixed)
    check("Mixed AR+EN → ar (dominant)", lang == "ar", f"got {lang!r} conf={conf:.2f}")

    # Numbers only
    lang2, conf2 = detect_language_with_confidence("12345")
    check("Numbers only → no crash", lang2 is not None)

    # Unicode emoji only
    lang3, conf3 = detect_language_with_confidence("😊🎉")
    check("Emoji only → no crash", lang3 is not None)

    # Very long Arabic
    long_ar = "كيف يمكنني إضافة إعلان جديد على المنصة؟ " * 5
    lang4, conf4 = detect_language_with_confidence(long_ar)
    check("Long Arabic → ar high conf", lang4 == "ar" and conf4 >= 0.85, f"conf={conf4:.2f}")

    # Dialect on non-Arabic
    d, c = detect_arabic_dialect("Hello how are you?")
    check("Dialect on English → unknown", d == "unknown" and c == 0.0)

    # Intent with None history
    intent, conf = detect_intent("اكتبها بالانجليزي", None)
    check("Intent with None history → new_question (no prior msg)", intent == "new_question")


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

def run_all():
    print("\n" + "=" * 58)
    print("  FODWA NLP Utils v2 -- Production Test Suite")
    print("=" * 58)

    test_language_confidence()
    test_short_text_heuristic()
    test_dialect_confidence()
    test_dialect_confidence_score()
    test_intent_confidence()
    test_intent_long_text_not_rephrase()
    test_build_prompt()
    test_clean_text()
    test_sentence_truncation()
    test_nlp_metadata()
    test_preprocess_pipeline()
    test_edge_cases()

    print("\n" + "=" * 58)
    total = len(_PASSED) + len(_FAILED)
    print(f"  Results: {len(_PASSED)}/{total} passed")
    if _FAILED:
        print("\n  Failed:")
        for t in _FAILED:
            print(f"    x {t}")
    else:
        print("  All tests passed!")
    print("=" * 58 + "\n")
    return len(_FAILED) == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
