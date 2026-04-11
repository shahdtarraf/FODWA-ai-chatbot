# Deploying Fodwa AI Chatbot to Render (Django Version)

This guide outlines the exact steps to deploy this production-ready Django AI Microservice safely to Render's free tier.

Our application is perfectly optimized to stay well under the strict 512MB limit by enforcing lazy loading for our FAISS index, eliminating `langchain` and `pypdf` bloat in production, and isolating data ingestion strictly to local machines.

## 1. Prepare your GitHub Repository
Since the `.env` file should never be uploaded with your API key, ensure everything is committed properly without leaks.

1. Init your repo locally and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Fodwa AI Chatbot (Django)"
   git branch -M main
   # Add your github origin: git remote add origin https://github.com/YOUR_GIT/repo.git
   # git push -u origin main
   ```
2. Verify that `chatbot/data/index.faiss` and `chatbot/data/chunks.json` **ARE** included in your push. Without them, the chatbot will fallback to "لا أملك معلومات".

## 2. Create the Web Service on Render
1. Log in to [Render](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub account and select your chatbot repository.

## 3. Configure the Deployment Settings
Provide these precise settings:

* **Name**: Fodwa-AI-Chatbot (or anything you prefer)
* **Region**: Choose the closest (e.g. Frankfurt)
* **Branch**: `main`
* **Environment**: `Docker`
* **Instance Type**: `Free` (512MB RAM)

Render will automatically detect the `Dockerfile` in the root folder because we are using Docker environment.

### Alternative: Native Python (without Docker)
If you prefer not to use Docker, set these instead:
* **Environment**: `Python 3`
* **Build Command**: `pip install -r requirements.txt && python manage.py collectstatic --noinput`
* **Start Command**: `gunicorn fodwa_project.wsgi:application --bind 0.0.0.0:$PORT --workers 2 --timeout 120`

## 4. Set Environment Variables
Scroll down to **Environment Variables** and expand it. Add:

* **Key**: `OPENAI_API_KEY` → **Value**: *Your OpenAI Secret Key (starting with sk-...)*
* **Key**: `DJANGO_SECRET_KEY` → **Value**: *A random string (use `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"` to generate one)*
* **Key**: `DEBUG` → **Value**: `False`

## 5. Deploy and Monitor
Click **Create Web Service**.
Render will begin to build your Docker image using the cached `python:3.10-slim`.
Since we mapped the port to **10000** in our Dockerfile, Render will safely expose it.

You can monitor the Server Logs. Look for this line right at the end to know everything successfully started:
> 🚀 Fodwa AI Chatbot started — FAISS will load on first request

## 6. Test the Live API
Your endpoint will look something like `https://fodwa-ai-chatbot.onrender.com`. Test it:

### Health Check
```bash
curl https://your-render-url.onrender.com/
```
Expected response:
```json
{"status": "ok", "service": "fodwa-ai-chatbot", "version": "1.0.0"}
```

### Chat Endpoint
```bash
curl -X POST https://your-render-url.onrender.com/chat \
-H "Content-Type: application/json" \
-d '{
  "message": "كيف يمكنني إلغاء إعلان مخالف؟",
  "token": "test-jwt-token"
}'
```

You will receive an entirely Arabic response directly from your AI agent using the localized data index context!

## 7. Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the development server
python manage.py runserver 8000

# Test health check
curl http://127.0.0.1:8000/

# Test chat endpoint
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "كيف ألغي إعلان مخالف"}'
```
