FROM python:3.10-slim

WORKDIR /app

# Install only production dependencies
COPY requirements.txt .

# Install deps — exclude ingestion-only packages to save space
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    faiss-cpu==1.8.0.post1 \
    "openai>=1.60.0" \
    pyjwt==2.8.0 \
    python-dotenv==1.0.0 \
    numpy==1.26.2

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 10000

# Run server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
