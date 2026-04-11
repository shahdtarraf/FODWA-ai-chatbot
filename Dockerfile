FROM python:3.10-slim

WORKDIR /app

# Install only production dependencies
RUN pip install --no-cache-dir \
    "django>=4.2,<5.0" \
    "djangorestframework>=3.14" \
    "django-cors-headers>=4.3" \
    "gunicorn>=21.2" \
    "whitenoise>=6.5" \
    faiss-cpu==1.8.0.post1 \
    "openai>=1.60.0" \
    pyjwt==2.8.0 \
    python-dotenv==1.0.0 \
    numpy==1.26.2

# Copy application code
COPY fodwa_project/ ./fodwa_project/
COPY chatbot/ ./chatbot/
COPY manage.py .

# Collect static files (required by whitenoise)
RUN DJANGO_SETTINGS_MODULE=fodwa_project.settings python manage.py collectstatic --noinput 2>/dev/null || true

# Expose port
EXPOSE 10000

# Run server with gunicorn
CMD ["gunicorn", "fodwa_project.wsgi:application", "--bind", "0.0.0.0:10000", "--workers", "2", "--timeout", "120"]
