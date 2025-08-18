from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings
# Extra imports for AI routes
from pydantic import BaseModel
import httpx
import os
import logging
import asyncio
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@app.get("/health/env")
async def environment_check():
    """Check if required environment variables are set"""
    env_vars = {
        "OPENAI_API_KEY": "✅" if os.getenv("OPENAI_API_KEY") else "❌",
        "PERPLEXITY_API_KEY": "✅" if os.getenv("PERPLEXITY_API_KEY") else "❌",
    }
    
    return {
        "environment_variables": env_vars,
        "all_present": all(var == "✅" for var in env_vars.values())
    }

# ==============================
# AI Proxy Routes
# ==============================

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    model_used: str
    status: str = "success"

# Read API keys from environment variables (must be set in Render)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

# Configuration
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60.0"))  # 60 seconds default
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

async def call_openai_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    """Call OpenAI API with retry logic and proper error handling."""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(LLM_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Calling OpenAI API (attempt {attempt + 1}/{max_retries})")
                
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.7
                    },
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "model_used": "gpt-4o-mini",
                        "status": "success"
                    }
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {delay} seconds before retry")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                else:
                    error_detail = response.text
                    logger.error(f"OpenAI API error: {response.status_code} - {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"OpenAI API error: {error_detail}"
                    )
                    
        except httpx.ReadTimeout:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=504,
                    detail="Request to OpenAI API timed out after all retry attempts"
                )
            else:
                delay = 2 ** attempt
                logger.warning(f"Request timed out, retrying in {delay} seconds")
                await asyncio.sleep(delay)
                continue
                
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to OpenAI API: {str(e)}"
                )
            else:
                delay = 2 ** attempt
                logger.warning(f"Connection error, retrying in {delay} seconds: {str(e)}")
                await asyncio.sleep(delay)
                continue
    
    raise HTTPException(status_code=500, detail="All retry attempts failed")

async def call_perplexity_with_retry(prompt: str, max_retries: int = MAX_RETRIES) -> dict:
    """Call Perplexity API with retry logic and proper error handling."""
    
    if not PERPLEXITY_API_KEY:
        raise HTTPException(status_code=500, detail="Perplexity API key not configured")
    
    for attempt in range(max_retries):
        try:
            timeout = httpx.Timeout(LLM_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.info(f"Calling Perplexity API (attempt {attempt + 1}/{max_retries})")
                
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.1-sonar-small-128k-online",  # Updated model name
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.2,
                        "stream": False
                    },
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "response": result["choices"][0]["message"]["content"],
                        "model_used": "llama-3.1-sonar-small-128k-online",
                        "status": "success"
                    }
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        delay = 2 ** attempt * 2  # Longer delay for rate limits
                        logger.warning(f"Rate limited, waiting {delay} seconds before retry")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
                else:
                    error_detail = response.text
                    logger.error(f"Perplexity API error: {response.status_code} - {error_detail}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Perplexity API error: {error_detail}"
                    )
                    
        except httpx.ReadTimeout:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=504,
                    detail="Request to Perplexity API timed out after all retry attempts"
                )
            else:
                delay = 2 ** attempt
                logger.warning(f"Request timed out, retrying in {delay} seconds")
                await asyncio.sleep(delay)
                continue
                
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to Perplexity API: {str(e)}"
                )
            else:
                delay = 2 ** attempt
                logger.warning(f"Connection error, retrying in {delay} seconds: {str(e)}")
                await asyncio.sleep(delay)
                continue
    
    raise HTTPException(status_code=500, detail="All retry attempts failed")

@app.post("/api/v1/chatgpt", response_model=ChatResponse)
async def chatgpt_chat(req: ChatRequest):
    """Proxy request to OpenAI ChatGPT API with improved error handling"""
    try:
        logger.info(f"Received ChatGPT request with prompt: {req.prompt[:100]}...")
        result = await call_openai_with_retry(req.prompt)
        logger.info("ChatGPT request completed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ChatGPT endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/perplexity", response_model=ChatResponse)
async def perplexity_chat(req: ChatRequest):
    """Proxy request to Perplexity API with improved error handling"""
    try:
        logger.info(f"Received Perplexity request with prompt: {req.prompt[:100]}...")
        result = await call_perplexity_with_retry(req.prompt)
        logger.info("Perplexity request completed successfully")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Perplexity endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Debug endpoints (remove in production)
@app.get("/debug/openai")
async def debug_openai():
    """Debug endpoint to check OpenAI configuration"""
    return {
        "api_key_present": bool(OPENAI_API_KEY),
        "api_key_length": len(OPENAI_API_KEY) if OPENAI_API_KEY else 0,
        "timeout_setting": LLM_TIMEOUT,
        "max_retries": MAX_RETRIES
    }

@app.get("/debug/perplexity")
async def debug_perplexity():
    """Debug endpoint to check Perplexity configuration"""
    return {
        "api_key_present": bool(PERPLEXITY_API_KEY),
        "api_key_length": len(PERPLEXITY_API_KEY) if PERPLEXITY_API_KEY else 0,
        "timeout_setting": LLM_TIMEOUT,
        "max_retries": MAX_RETRIES
    }

# ==============================
# Run app (only in local dev)
# ==============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT)
