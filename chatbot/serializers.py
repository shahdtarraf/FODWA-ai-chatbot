"""
DRF Serializers — replaces Pydantic schemas from app/models/schemas.py.
"""

from rest_framework import serializers


class ChatRequestSerializer(serializers.Serializer):
    """Chat endpoint request body."""
    message = serializers.CharField(required=True)
    token = serializers.CharField(required=False, allow_blank=True, allow_null=True, default=None)


class ChatResponseSerializer(serializers.Serializer):
    """Chat endpoint response body."""
    response = serializers.CharField()
    user_id = serializers.CharField(allow_null=True)
