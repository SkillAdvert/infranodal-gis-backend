# main.py - Updated with API routes
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from config import settings

# Import API routes
from api.routes import router as api_router

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Geospatial data platform with persona-based layers and configurable scoring algorithms"
)

# Add CORS middleware to allow web frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["geospatial"])

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.VERSION,
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1"
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"🚀 Starting {settings.APP_NAME}")
    print(f"📍 Running on http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"📊 API docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(f"🗺️ API endpoints: http://{settings.API_HOST}:{settings.API_PORT}/api/v1")
    
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG  # Auto-reload when code changes in debug mode
    )