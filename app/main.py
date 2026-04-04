"""
Fodwa AI Support Chatbot — FastAPI Application
shahd.ai
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.routes.chat import router as chat_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Fodwa AI Support Chatbot",
    description="مساعد ذكي لموقع shahd.ai",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router)


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "fodwa-ai-chatbot",
        "version": "1.0.0"
    }


@app.on_event("startup")
async def startup_event():
    """Log startup — FAISS is NOT loaded here (lazy loading)."""
    logger.info("🚀 Fodwa AI Chatbot started — FAISS will load on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown."""
    logger.info("🛑 Fodwa AI Chatbot shutting down")
