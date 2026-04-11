"""
RAG pipeline test — tests the chat service directly (no server needed).
Updated imports for Django project structure.
"""

import os
import sys
import logging
import django

# Setup Django before importing any app modules
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fodwa_project.settings")
django.setup()

# Configure logging to see our new logs
logging.basicConfig(level=logging.INFO)

from chatbot.services.chat_service import process_chat


def main():
    test_queries = [
        "هل يمكن انشاء اكثر من حساب ",  # Vague, slight typo
        "الف شكر",  # Generic, should gracefully fallback or use history context
        "كيف ألغي إعلان مخالف",  # Clean expected question
        "ما هي شروط الاعلان"  # Standard query
    ]

    for query in test_queries:
        print(f"\n{'='*50}\nTesting Query: {query}")
        response = process_chat(query, user_id="test_user")
        print(f"Response: {response}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
