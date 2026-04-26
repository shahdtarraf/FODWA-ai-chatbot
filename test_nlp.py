"""
Tests for chatbot/utils/nlp_utils.py

Run with:
    python -m pytest test_nlp.py -v

Or without pytest:
    python test_nlp.py
"""

import sys
import os

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Minimal Django setup so imports don't fail ──────────────
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fodwa_project.settings")

import django
try:
    django.setup()
except Exception:
    pass  # OK if settings not fully configured for unit tests

from chatbot.utils.nlp_utils import (
    detect_language,
    detect_arabic_dialect,
    detect_intent,
    build_dynamic_system_prompt,
    post_process_response,
    preprocess,
    NLPMetadata,
)


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

_PASSED = []
_FAILED = []


def check(test_name: str, condition: bool, detail: str = "") -> None:
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"  {status}  {test_name}"
    if detail:
        msg += f"\n          → {detail}"
    print(msg)
    (_PASSED if condition else _FAILED).append(test_name)


# ═══════════════════════════════════════════════════════════
# 1. Language Detection
# ═══════════════════════════════════════════════════════════

def test_language_detection():
    print("\n─── 1. Language Detection ───────────────────────────")

    cases = [
        ("كيف يمكنني حذف إعلاني؟", "ar"),
        ("How do I delete my listing?", "en"),
        ("Wie kann ich mein Konto löschen?", "de"),
        ("Comment puis-je supprimer mon annonce?", "fr"),
        ("İlanımı nasıl silebilirim?", "tr"),
        ("", "en"),  # Empty → fallback
    ]

    for text, expected in cases:
        result = detect_language(text)
        check(
            f"detect_language({text[:30]!r}...)",
            result == expected,
            f"expected={expected!r}, got={result!r}",
        )


# ═══════════════════════════════════════════════════════════
# 2. Arabic Dialect Detection
# ═══════════════════════════════════════════════════════════

def test_arabic_dialect_detection():
    print("\n─── 2. Arabic Dialect Detection ─────────────────────")

    cases = [
        ("ايه ده؟ عايز أعرف إزاي أحذف الإعلان بتاعي", "egyptian"),
        ("شو هلق؟ كيفك؟ ما في مشكلة", "syrian"),
        ("شنو هواية ماكو وقت", "iraqi"),
        ("وش المشكلة؟ وايد أسئلة", "gulf"),
        ("واش كيداير؟ بزاف مشاكل", "moroccan"),
        ("أود أن أستفسر عن هذه المسألة", "msa"),
        ("hello", "unknown"),  # Not Arabic at all
    ]

    for text, expected in cases:
        result = detect_arabic_dialect(text)
        check(
            f"detect_arabic_dialect({text[:35]!r}...)",
            result == expected,
            f"expected={expected!r}, got={result!r}",
        )


# ═══════════════════════════════════════════════════════════
# 3. Intent Detection
# ═══════════════════════════════════════════════════════════

def test_intent_detection():
    print("\n─── 3. Intent Detection ─────────────────────────────")

    history_with_assistant = [
        {"role": "user", "content": "كيف أحذف الإعلان؟"},
        {"role": "assistant", "content": "يمكنك حذف الإعلان من لوحة التحكم."},
    ]
    empty_history: list = []

    cases = [
        # Rephrase/translate cases (need history)
        ("اكتبها بالعربي", history_with_assistant, "rephrase_request"),
        ("اكتبه بالانجليزي", history_with_assistant, "rephrase_request"),
        ("حولها للمصري", history_with_assistant, "rephrase_request"),
        ("in English", history_with_assistant, "rephrase_request"),
        ("auf Deutsch", history_with_assistant, "rephrase_request"),
        ("translate to French", history_with_assistant, "rephrase_request"),
        ("بالمصري", history_with_assistant, "rephrase_request"),
        ("بالسوري", history_with_assistant, "rephrase_request"),
        ("بالخليجي", history_with_assistant, "rephrase_request"),
        ("بشكل أبسط", history_with_assistant, "rephrase_request"),
        ("simplify it", history_with_assistant, "rephrase_request"),
        ("بالفصحى", history_with_assistant, "rephrase_request"),
        # Same phrase but NO history → new_question
        ("اكتبها بالعربي", empty_history, "new_question"),
        # Genuine new questions (even with history)
        ("كيف يمكنني حذف إعلاني من FODWA؟", history_with_assistant, "new_question"),
        ("What is FODWA?", history_with_assistant, "new_question"),
    ]

    for text, history, expected in cases:
        result = detect_intent(text, history)
        check(
            f"detect_intent({text!r}) [history={'yes' if history else 'no'}]",
            result == expected,
            f"expected={expected!r}, got={result!r}",
        )


# ═══════════════════════════════════════════════════════════
# 4. Dynamic System Prompt Builder
# ═══════════════════════════════════════════════════════════

def test_build_dynamic_system_prompt():
    print("\n─── 4. build_dynamic_system_prompt ─────────────────")

    # English new question
    prompt = build_dynamic_system_prompt(lang="en", intent="new_question")
    check("Prompt contains English instruction", "English" in prompt)
    check("Prompt contains length rule", "3 to 4 lines" in prompt)

    # Arabic Egyptian rephrase
    prompt = build_dynamic_system_prompt(lang="ar", dialect="egyptian", intent="rephrase_request")
    check("Arabic prompt has dialect label", "Egyptian Arabic" in prompt)
    check("Rephrase prompt has REPHRASE MODE", "REPHRASE MODE" in prompt)
    check("Rephrase prompt forbids new info", "Do NOT add any new information" in prompt)

    # German new question
    prompt = build_dynamic_system_prompt(lang="de", intent="new_question")
    check("German prompt detected", "German" in prompt)

    # Separator present
    check("Prompt has separator", "────" in prompt)


# ═══════════════════════════════════════════════════════════
# 5. Post-processing: FODWA RTL Fix
# ═══════════════════════════════════════════════════════════

def test_post_process_fodwa_rtl():
    print("\n─── 5. post_process_response — FODWA RTL Fix ────────")

    LRM = "\u200E"

    cases = [
        ("FODWA is great",       True,  "FODWA surrounded by LRM"),
        ("Use fodwa to post",    True,  "lowercase fodwa fixed"),
        ("AWDOF bug appears",    True,  "AWDOF replaced with LRM-FODWA"),
        ("No platform mention",  False, "No LRM inserted when no keyword"),
    ]

    for text, expects_lrm, label in cases:
        result = post_process_response(text)
        has_lrm = LRM in result
        check(label, has_lrm == expects_lrm, f"result={result!r}")


# ═══════════════════════════════════════════════════════════
# 6. Post-processing: Line Truncation
# ═══════════════════════════════════════════════════════════

def test_post_process_truncation():
    print("\n─── 6. post_process_response — Truncation ───────────")

    short_text = "Line 1\nLine 2\nLine 3"
    check("Short text not truncated", post_process_response(short_text) == short_text)

    long_text = "\n".join([f"Line {i}: some content here" for i in range(1, 10)])
    result = post_process_response(long_text)
    line_count = len([l for l in result.splitlines() if l.strip()])
    check(
        f"Long text (9 lines) truncated to ≤4 lines",
        line_count <= 4,
        f"got {line_count} lines",
    )
    check("Truncated text ends with ellipsis", result.endswith("…"), f"result={result[-20:]!r}")

    # Text already ending with punctuation — no double ellipsis
    text_with_period = "\n".join([f"This is sentence {i}." for i in range(1, 8)])
    result2 = post_process_response(text_with_period)
    check("No double punctuation after truncation", not result2.endswith(".…"))


# ═══════════════════════════════════════════════════════════
# 7. Full Pipeline: preprocess()
# ═══════════════════════════════════════════════════════════

def test_preprocess_pipeline():
    print("\n─── 7. preprocess() full pipeline ───────────────────")

    history = [
        {"role": "assistant", "content": "يمكنك حذف الإعلان من لوحة التحكم."}
    ]

    # German → new question
    meta = preprocess("Wie kann ich mein Konto löschen?", [])
    check("German language detected", meta.language == "de", f"got {meta.language!r}")
    check("German dialect is None", meta.dialect is None)
    check("German intent is new_question", meta.intent == "new_question")

    # Arabic Egyptian rephrase
    meta2 = preprocess("اكتبها بالمصري", history)
    check("Arabic language detected", meta2.language == "ar", f"got {meta2.language!r}")
    check("Rephrase intent detected", meta2.intent == "rephrase_request", f"got {meta2.intent!r}")

    # Arabic new question
    meta3 = preprocess("كيف يمكنني حذف إعلاني من المنصة؟", history)
    check("Arabic new question detected", meta3.intent == "new_question")

    # Returns NLPMetadata
    check("Returns NLPMetadata instance", isinstance(meta, NLPMetadata))


# ═══════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════

def run_all():
    print("\n" + "═" * 55)
    print("  FODWA NLP Utils — Test Suite")
    print("═" * 55)

    test_language_detection()
    test_arabic_dialect_detection()
    test_intent_detection()
    test_build_dynamic_system_prompt()
    test_post_process_fodwa_rtl()
    test_post_process_truncation()
    test_preprocess_pipeline()

    print("\n" + "═" * 55)
    total = len(_PASSED) + len(_FAILED)
    print(f"  Results: {len(_PASSED)}/{total} passed")
    if _FAILED:
        print(f"\n  Failed tests:")
        for t in _FAILED:
            print(f"    ✗ {t}")
    else:
        print("  🎉 All tests passed!")
    print("═" * 55 + "\n")

    return len(_FAILED) == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
