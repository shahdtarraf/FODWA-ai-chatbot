"""
Chatbot URL patterns — replaces FastAPI router.
"""

from django.urls import path
from chatbot.views import HealthCheckView, ChatView

urlpatterns = [
    path("", HealthCheckView.as_view(), name="health-check"),
    path("chat", ChatView.as_view(), name="chat"),
]
