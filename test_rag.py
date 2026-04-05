import asyncio
import logging

# Configure logging to see our new logs
logging.basicConfig(level=logging.INFO)

from app.services.chat_service import process_chat

async def main():
    test_queries = [
        "هل يمكن انشاء اكثر من حساب ", # Vague, slight typo
        "الف شكر", # Generic, should gracefully fallback or use history context
        "كيف ألغي إعلان مخالف", # Clean expected question
        "ما هي شروط الاعلان" # Standard query
    ]

    for query in test_queries:
        print(f"\n{'='*50}\nTesting Query: {query}")
        response = await process_chat(query, user_id="test_user")
        print(f"Response: {response}")
        print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
