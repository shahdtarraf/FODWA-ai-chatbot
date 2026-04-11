"""
Chatbot App Configuration.
Logs startup message — replaces FastAPI's on_event("startup").
"""

import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class ChatbotConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chatbot"
    verbose_name = "Fodwa AI Chatbot"

    def ready(self):
        """Called when Django starts — equivalent to FastAPI startup event."""
        logger.info("🚀 Fodwa AI Chatbot started — FAISS will load on first request")
