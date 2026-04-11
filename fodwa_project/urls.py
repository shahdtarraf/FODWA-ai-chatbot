"""
Fodwa AI Support Chatbot — URL Configuration
"""

from django.urls import path, include

urlpatterns = [
    path("", include("chatbot.urls")),
]
