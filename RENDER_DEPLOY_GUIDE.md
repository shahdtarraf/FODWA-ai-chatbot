# Deploying Fodwa AI Chatbot to Render

This guide outlines the exact steps to deploy this production-ready AI Microservice safely to Render's free tier. 

Our application is perfectly optimized to stay well under the strict 512MB limit by enforcing lazy loading for our FAISS index, eliminating `langchain` and `pypdf` bloat in production, and isolating data ingestion strictly to local machines.

## 1. Prepare your GitHub Repository
Since the `.env` file should never be uploaded with your API key, ensure everything is committed properly without leaks.

1. Init your repo locally and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Fodwa AI Chatbot"
   git branch -M main
   # Add your github origin: git remote add origin https://github.com/YOUR_GIT/repo.git
   # git push -u origin main
   ```
2. Verify that `app/data/index.faiss` and `app/data/chunks.json` **ARE** included in your push. Without them, the chatbot will fallback to "لا أملك معلومات".

## 2. Create the Web Service on Render
1. Log in to [Render](https://render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub account and select your chatbot repository.

## 3. Configure the Deployment Settings
Provide these precise settings to ensure proper Docker compatibility and start sequences:

* **Name**: Fodwa-AI-Chatbot (or anything you prefer)
* **Region**: Choose the closest (e.g. Frankfurt)
* **Branch**: `main`
* **Environment**: `Docker`
* **Instance Type**: `Free` (512MB RAM)

Render will automatically detect the `Dockerfile` in the root folder because we are using Docker environment.

## 4. Set Environment Variables
Scroll down to **Environment Variables** and expand it. Add your OpenAI API key so the Docker image can access it at runtime:

* **Key**: `OPENAI_API_KEY`
* **Value**: *Your OpenAI Secret Key goes here (starting with sk-...)*

## 5. Deploy and Monitor
Click **Create Web Service**. 
Render will begin to build your Docker image using the cached `python:3.10-slim`. 
Since we mapped the port to **10000** in our Dockerfile, Render will safely expose it. 

You can monitor the Server Logs. Look for this line right at the end to know everything successfully started:
> 🚀 Fodwa AI Chatbot started — FAISS will load on first request

## 6. Test the Live API
Your endpoint will look something like `https://fodwa-ai-chatbot.onrender.com`. Test it exactly like you did locally:

```bash
curl -X POST https://your-render-url.onrender.com/chat \
-H "Content-Type: application/json" \
-d '{
  "message": "I made a mistake, كيف الغي إعلان مخالف؟",
  "token": "test-jwt-token"
}'
```

You will receive an entirely Arabic response directly from your AI agent using the localized data index context!
