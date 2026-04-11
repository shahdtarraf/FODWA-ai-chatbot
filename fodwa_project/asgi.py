"""
ASGI config for Fodwa AI Support Chatbot.
Exposes the ASGI callable as a module-level variable named ``application``.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "fodwa_project.settings")

application = get_asgi_application()
