from f

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
        "SUPABASE_URL": "✅" if os.getenv("SUPABASE_URL") else "❌",
        "SUPABASE_ANON_KEY": "✅" if os.getenv("SUPABASE_ANON_KEY") else "❌",
        # Keep DATABASE_URL check for backwards compatibility
        "DATABASE_URL": "✅" if os.getenv("DATABASE_URL") else "❌",
    }
    
    return {
        "environment_variables": env_vars,
        "supabase_configured": env_vars["SUPABASE_URL"] == "✅" and env_vars["SUPABASE_ANON_KEY"] == "✅",
        "ai_configured": env_vars["OPENAI_API_KEY"] == "✅" and env_vars["PERPLEXITY_API_KEY"] == "✅"
    }

# ==============================
# Supabase Configuration
# ==============================

# Supabase settings
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

async def get_supabase_data(table: str, select: str = "*", filters: Optional[Dict[str, Any]] = None) -> List[Dict[Any, Any]]:
    """Fetch data from Supabase REST API"""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(status_code=500, detail="Supabase configuration missing - check SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
    
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json"
    }
    
    params = {"select": select} if select != "*" else {}
    
    # Add filters if provided
    if filters:
        params.update(filters)
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Fetching data from Supabase table: {table}")
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully retrieved {len(data)} records from {table}")
            return data
    except httpx.HTTPError as e:
        logger.error(f"Supabase API error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error accessing Supabase: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# ==============================
# AI Proxy Routes (UNCHANGED)
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

@app.get("/debug/supabase")
async def debug_supabase():
    """Debug endpoint to check Supabase configuration"""
    return {
        "supabase_url_present": bool(SUPABASE_URL),
        "supabase_anon_key_present": bool(SUPABASE_ANON_KEY),
        "supabase_url": SUPABASE_URL[:50] + "..." if SUPABASE_URL else None,
        "configuration_complete": bool(SUPABASE_URL and SUPABASE_ANON_KEY)
    }

# ==============================
# Renewable Sites API Routes (UPDATED TO USE SUPABASE REST API)
# ==============================

def convert_to_geojson(sites_data: List[Dict]) -> Dict:
    """Convert renewable sites data to GeoJSON FeatureCollection format"""
    features = []
    
    for site in sites_data:
        # Handle potential None values and different field names
        lat = site.get('latitude')
        lng = site.get('longitude')
        
        if lat is None or lng is None:
            logger.warning(f"Skipping site {site.get('site_name', 'Unknown')} - missing coordinates")
            continue
            
        try:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]  # GeoJSON format: [longitude, latitude]
                },
                "properties": {
                    "id": site.get('id'),
                    "name": site.get('site_name'),  # Keep 'name' for compatibility with your existing frontend
                    "site_name": site.get('site_name'),  # Also include original field name
                    "developer": site.get('developer'),
                    "technology": site.get('technology'),
                    "capacity_mw": float(site.get('capacity_mw', 0)),
                    "status": site.get('status')
                }
            }
            features.append(feature)
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing site {site.get('site_name', 'Unknown')}: {e}")
            continue
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.get("/api/sites")
async def get_renewable_sites():
    """Get all renewable energy sites as GeoJSON for map display"""
    
    try:
        # Fetch data from Supabase using REST API
        sites_data = await get_supabase_data("renewable_sites")
        
        if not sites_data:
            logger.warning("No renewable sites data found in database")
            return {
                "type": "FeatureCollection",
                "features": []
            }
        
        # Convert to GeoJSON format for map display (same format as before)
        geojson = convert_to_geojson(sites_data)
        
        logger.info(f"Successfully converted {len(geojson['features'])} sites to GeoJSON")
        return geojson
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_renewable_sites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch renewable sites: {str(e)}")

@app.get("/api/sites/simple")
async def get_sites_simple():
    """Get renewable sites as simple JSON (for testing)"""
    
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        logger.info(f"Successfully retrieved {len(sites_data)} sites in simple format")
        return {"sites": sites_data, "count": len(sites_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_sites_simple: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch sites data: {str(e)}")

# Additional useful endpoints
@app.get("/api/sites/{site_id}")
async def get_renewable_site(site_id: int):
    """Get a specific renewable energy site by ID"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        # Filter by site_id
        site = next((s for s in sites_data if s.get('id') == site_id), None)
        
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
            
        return site
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching site {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch site data")

@app.get("/api/sites/stats")
async def get_renewable_sites_stats():
    """Get statistics about renewable energy sites"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        if not sites_data:
            return {
                "total_sites": 0,
                "total_capacity_mw": 0,
                "technology_breakdown": {},
                "status_breakdown": {}
            }
        
        total_sites = len(sites_data)
        total_capacity = sum(site.get('capacity_mw', 0) for site in sites_data if site.get('capacity_mw'))
        
        # Group by technology
        tech_stats = {}
        for site in sites_data:
            tech = site.get('technology', 'Unknown')
            if tech not in tech_stats:
                tech_stats[tech] = {"count": 0, "total_capacity_mw": 0}
            tech_stats[tech]["count"] += 1
            tech_stats[tech]["total_capacity_mw"] += site.get('capacity_mw', 0)
        
        # Group by status
        status_stats = {}
        for site in sites_data:
            status = site.get('status', 'Unknown')
            status_stats[status] = status_stats.get(status, 0) + 1
        
        return {
            "total_sites": total_sites,
            "total_capacity_mw": total_capacity,
            "technology_breakdown": tech_stats,
            "status_breakdown": status_stats
        }
        
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to calculate statistics")

# ==============================
# Run app (only in local dev)
# ==============================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.API_PORT)
