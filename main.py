from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings

# Extra imports for AI routes
from pydantic import BaseModel
import httpx
import os

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Geospatial data platform with persona-based layers"
)

# CORS middleware (allows frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo, allow all; later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# ==============================
# AI Proxy Routes
# ==============================

class ChatRequest(BaseModel):
    prompt: str

# Read API keys from environment variables (must be set in Render)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

@app.post("/api/v1/chatgpt")
async def chatgpt_chat(req: ChatRequest):
    """Proxy request to OpenAI ChatGPT API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini",   # or gpt-4.1 if available on your account
                "messages": [{"role": "user", "content": req.prompt}],
            },
        )
    return response.json()

@app.post("/api/v1/perplexity")
async def perplexity_chat(req: ChatRequest):
    """Proxy request to Perplexity Pro API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {PERPLEXITY_API_KEY}"},
            json={
                "model": "sonar-pro",   # Perplexity Pro model
                "messages": [{"role": "user", "content": req.prompt}],
            },
        )
    return response.json()


# ==============================
# Run app (only in local dev)
# ==============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT)
