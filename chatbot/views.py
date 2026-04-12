"""
Chat Views — DRF APIViews replacing FastAPI routes.
Preserves exact same endpoint behavior:
  GET  /     → Health check
  POST /chat → Chat endpoint (RAG pipeline)
"""

import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse

from chatbot.serializers import ChatRequestSerializer, ChatResponseSerializer
from chatbot.auth.jwt_handler import decode_token
from chatbot.services.chat_service import process_chat

logger = logging.getLogger(__name__)


class HealthCheckView(APIView):
    """Health check endpoint — GET /"""

    def get(self, request):
        """Health check endpoint."""
        return Response({
            "status": "ok",
            "service": "fodwa-ai-chatbot",
            "version": "1.0.0"
        })


class ChatView(APIView):
    """
    Chat endpoint — POST /chat
    Accepts a message and optional JWT token.
    Returns an Arabic response from the RAG pipeline.
    """

    def post(self, request):
        """
        Chat endpoint — accepts a message and optional JWT token.
        Returns an Arabic response from the RAG pipeline.
        """
        try:
            # Validate request data
            serializer = ChatRequestSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(
                    {"response": "أعتذر، الرجاء إرسال رسالة صحيحة.", "user_id": "anonymous"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            message = serializer.validated_data["message"]
            token = serializer.validated_data.get("token")

            # Extract user_id from JWT token
            user_id = decode_token(token) if token else "anonymous"

            # Process message through RAG pipeline
            response_text = process_chat(
                message=message,
                user_id=user_id,
            )

            # Build response
            response_data = {
                "response": response_text,
                "user_id": user_id,
            }
            response_serializer = ChatResponseSerializer(data=response_data)
            response_serializer.is_valid()

            return JsonResponse(response_serializer.data, json_dumps_params={'ensure_ascii': False})

        except Exception as e:
            logger.error(f"Chat endpoint error: {e}")
            return JsonResponse({
                "response": "أعتذر، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.",
                "user_id": "anonymous",
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR, json_dumps_params={'ensure_ascii': False})
