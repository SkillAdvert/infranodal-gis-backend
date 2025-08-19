# ADD THESE IMPORTS to your existing imports section
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import io

# ADD THESE MODELS after your existing ChatRequest/ChatResponse models
class ScoringConfig(BaseModel):
    capacity_weight: float = 0.3
    grid_proximity_weight: float = 0.25
    planning_risk_weight: float = 0.2
    market_demand_weight: float = 0.15
    competition_weight: float = 0.1

# GLOBAL VARIABLE - add after your existing global variables
current_scoring_config = ScoringConfig()

# ADD THIS CLASS after your existing classes
class SiteScorer:
    def __init__(self, config: ScoringConfig):
        self.config = config
    
    def normalize_capacity(self, capacity_mw: float, max_capacity: float = 4000.0) -> float:
        """Normalize capacity to 0-100 scale"""
        return min(100, (capacity_mw / max_capacity) * 100)
    
    def calculate_grid_proximity_score(self, site_data: dict) -> float:
        """Calculate grid proximity score based on location"""
        lat, lng = site_data['latitude'], site_data['longitude']
        
        # UK regional grid scoring
        if 51.0 <= lat <= 52.0 and -1.0 <= lng <= 1.0:  # London/Southeast
            return 90.0
        elif lat >= 56.0:  # Scotland
            return 70.0
        elif lat <= 52.0 and lng <= -2.0:  # Wales/Southwest
            return 75.0
        else:  # Midlands/North
            return 80.0
    
    def calculate_planning_risk_score(self, site_data: dict) -> float:
        """Calculate planning risk score"""
        status = site_data.get('status', '').lower()
        technology = site_data.get('technology', '').lower()
        
        if 'operational' in status:
            base_score = 100.0
        elif 'construction' in status:
            base_score = 95.0
        elif 'planning' in status:
            base_score = 60.0
        else:
            base_score = 50.0
        
        # Technology adjustments
        if 'solar' in technology:
            base_score += 10.0
        elif 'offshore' in technology:
            base_score -= 10.0
        
        return min(100.0, max(0.0, base_score))
    
    def calculate_market_demand_score(self, site_data: dict) -> float:
        """Calculate market demand score"""
        technology = site_data.get('technology', '').lower()
        
        if 'offshore wind' in technology:
            return 95.0
        elif 'battery' in technology:
            return 90.0
        elif 'solar' in technology:
            return 85.0
        elif 'onshore wind' in technology:
            return 80.0
        else:
            return 75.0
    
    def calculate_competition_score(self, site_data: dict, all_sites: list) -> float:
        """Calculate competition score"""
        lat, lng = site_data['latitude'], site_data['longitude']
        technology = site_data.get('technology', '').lower()
        
        nearby_similar = 0
        for other_site in all_sites:
            if other_site['id'] == site_data['id']:
                continue
                
            other_lat, other_lng = other_site['latitude'], other_site['longitude']
            other_tech = other_site.get('technology', '').lower()
            
            # Simple proximity check (~50km)
            if abs(lat - other_lat) < 0.5 and abs(lng - other_lng) < 0.5 and technology in other_tech:
                nearby_similar += 1
        
        if nearby_similar == 0:
            return 100.0
        elif nearby_similar <= 2:
            return 80.0
        elif nearby_similar <= 4:
            return 60.0
        else:
            return 40.0
    
    def calculate_site_score(self, site_data: dict, all_sites: list) -> float:
        """Calculate overall investment score"""
        capacity_score = self.normalize_capacity(site_data['capacity_mw'])
        grid_score = self.calculate_grid_proximity_score(site_data)
        planning_score = self.calculate_planning_risk_score(site_data)
        market_score = self.calculate_market_demand_score(site_data)
        competition_score = self.calculate_competition_score(site_data, all_sites)
        
        total_score = (
            capacity_score * self.config.capacity_weight +
            grid_score * self.config.grid_proximity_weight +
            planning_score * self.config.planning_risk_weight +
            market_score * self.config.market_demand_weight +
            competition_score * self.config.competition_weight
        )
        
        return round(total_score, 1)
    
    def get_score_color(self, score: float) -> str:
        """Get color for map visualization"""
        if score >= 80:
            return "#2E8B57"  # Green
        elif score >= 60:
            return "#FFD700"  # Gold
        elif score >= 40:
            return "#FF8C00"  # Orange
        else:
            return "#DC143C"  # Red
    
    def get_marker_size(self, capacity_mw: float) -> str:
        """Get marker size for map"""
        if capacity_mw >= 1000:
            return "large"
        elif capacity_mw >= 200:
            return "medium"
        else:
            return "small"

# REPLACE your existing convert_to_geojson function with this enhanced version
def convert_to_geojson(sites_data: List[Dict], include_scores: bool = True) -> Dict:
    """Convert renewable sites data to GeoJSON FeatureCollection format with scoring"""
    scorer = SiteScorer(current_scoring_config) if include_scores else None
    features = []
    
    for site in sites_data:
        lat = site.get('latitude')
        lng = site.get('longitude')
        
        if lat is None or lng is None:
            logger.warning(f"Skipping site {site.get('site_name', 'Unknown')} - missing coordinates")
            continue
            
        try:
            # Calculate score if requested
            investment_score = None
            score_color = None
            marker_size = None
            
            if scorer:
                investment_score = scorer.calculate_site_score(site, sites_data)
                score_color = scorer.get_score_color(investment_score)
                marker_size = scorer.get_marker_size(site.get('capacity_mw', 0))
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lng), float(lat)]
                },
                "properties": {
                    "site_id": site.get('id'),
                    "name": site.get('site_name'),
                    "developer": site.get('developer'),
                    "technology": site.get('technology'),
                    "capacity_mw": float(site.get('capacity_mw', 0)),
                    "status": site.get('status'),
                    "created_at": site.get('created_at')
                }
            }
            
            # Add scoring data if available
            if investment_score is not None:
                feature["properties"].update({
                    "investment_score": investment_score,
                    "score_color": score_color,
                    "marker_size": marker_size
                })
            
            features.append(feature)
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing site {site.get('site_name', 'Unknown')}: {e}")
            continue
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

# UPDATE your existing /api/sites endpoint to include scoring parameter
@app.get("/api/sites")
async def get_renewable_sites(include_scores: bool = True):
    """Get all renewable energy sites as GeoJSON for map display with optional scoring"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        if not sites_data:
            logger.warning("No renewable sites data found in database")
            return {
                "type": "FeatureCollection",
                "features": []
            }
        
        # Convert to GeoJSON format with scoring
        geojson = convert_to_geojson(sites_data, include_scores)
        
        logger.info(f"Successfully converted {len(geojson['features'])} sites to GeoJSON with scoring: {include_scores}")
        return geojson
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_renewable_sites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch renewable sites: {str(e)}")

# ADD THESE NEW ENDPOINTS after your existing endpoints

@app.get("/api/sites/filtered")
async def get_filtered_sites(
    technology: Optional[str] = None,
    status: Optional[str] = None,
    min_capacity: Optional[float] = None,
    max_capacity: Optional[float] = None,
    min_score: Optional[float] = None
):
    """Get filtered sites in GeoJSON format"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        # Apply filters
        filtered_sites = sites_data
        
        if technology:
            filtered_sites = [s for s in filtered_sites if technology.lower() in s['technology'].lower()]
        
        if status:
            filtered_sites = [s for s in filtered_sites if status.lower() in s['status'].lower()]
        
        if min_capacity:
            filtered_sites = [s for s in filtered_sites if s['capacity_mw'] >= min_capacity]
        
        if max_capacity:
            filtered_sites = [s for s in filtered_sites if s['capacity_mw'] <= max_capacity]
        
        # Convert to GeoJSON with scores
        geojson = convert_to_geojson(filtered_sites, include_scores=True)
        
        # Apply score filter after scoring
        if min_score:
            geojson['features'] = [
                f for f in geojson['features'] 
                if f['properties'].get('investment_score', 0) >= min_score
            ]
        
        return geojson
        
    except Exception as e:
        logger.error(f"Error filtering sites: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to filter sites: {str(e)}")

@app.get("/api/sites/{site_id}/details")
async def get_site_details(site_id: int):
    """Get detailed site information with scoring breakdown"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        site = next((s for s in sites_data if s.get('id') == site_id), None)
        
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")
        
        # Calculate detailed scoring
        scorer = SiteScorer(current_scoring_config)
        overall_score = scorer.calculate_site_score(site, sites_data)
        
        score_breakdown = {
            "capacity_score": scorer.normalize_capacity(site['capacity_mw']),
            "grid_proximity_score": scorer.calculate_grid_proximity_score(site),
            "planning_risk_score": scorer.calculate_planning_risk_score(site),
            "market_demand_score": scorer.calculate_market_demand_score(site),
            "competition_score": scorer.calculate_competition_score(site, sites_data),
            "overall_score": overall_score
        }
        
        return {
            "site": site,
            "scoring": score_breakdown,
            "investment_recommendation": "Strong" if overall_score >= 80 else "Moderate" if overall_score >= 60 else "Cautious"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting site details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get site details: {str(e)}")

# ADMIN ENDPOINTS - add these new endpoints

@app.get("/admin/scoring-config")
async def get_scoring_config():
    """Get current scoring configuration"""
    return current_scoring_config

@app.put("/admin/scoring-config")
async def update_scoring_config(config: ScoringConfig):
    """Update scoring weights"""
    global current_scoring_config
    
    # Validate weights
    total_weight = (config.capacity_weight + config.grid_proximity_weight + 
                   config.planning_risk_weight + config.market_demand_weight + 
                   config.competition_weight)
    
    if not 0.8 <= total_weight <= 1.2:
        raise HTTPException(status_code=400, detail="Weights should sum to approximately 1.0")
    
    current_scoring_config = config
    logger.info(f"Updated scoring configuration: {config}")
    
    return {"message": "Scoring configuration updated", "config": config}

@app.post("/admin/upload-sites-data")
async def upload_sites_data(file: UploadFile = File(...)):
    """Upload CSV file to update sites data"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        content = await file.read()
        csv_data = content.decode('utf-8')
        
        # Parse CSV
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Validate required columns
        required_columns = ['site_name', 'developer', 'technology', 'capacity_mw', 'latitude', 'longitude', 'status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=required_columns)
        sites_data = df.to_dict('records')
        
        # TODO: In production, you'd update Supabase here
        # For now, just validate and return preview
        logger.info(f"CSV upload validated: {len(sites_data)} sites")
        
        return {
            "message": f"CSV validated successfully with {len(sites_data)} sites",
            "preview": sites_data[:3] if sites_data else [],
            "note": "To persist data, implement Supabase update logic"
        }
        
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")

@app.get("/admin/demo-data")
async def get_demo_data():
    """Get complete demo dataset for investor presentations"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        
        # Get scored GeoJSON
        geojson = convert_to_geojson(sites_data, include_scores=True)
        
        # Calculate summary stats
        scores = [f['properties']['investment_score'] for f in geojson['features'] if f['properties'].get('investment_score')]
        
        summary_stats = {
            "total_sites": len(sites_data),
            "total_capacity_mw": sum(s['capacity_mw'] for s in sites_data),
            "average_investment_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "high_opportunity_sites": len([s for s in scores if s >= 80]),
            "technologies": list(set(s['technology'] for s in sites_data)),
            "status_breakdown": {
                "operational": len([s for s in sites_data if 'operational' in s.get('status', '').lower()]),
                "construction": len([s for s in sites_data if 'construction' in s.get('status', '').lower()]),
                "planning": len([s for s in sites_data if 'planning' in s.get('status', '').lower()])
            }
        }
        
        return {
            "geojson": geojson,
            "summary": summary_stats,
            "current_weights": current_scoring_config
        }
        
    except Exception as e:
        logger.error(f"Error generating demo data: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating demo data: {str(e)}")

@app.post("/admin/recalculate-scores")
async def recalculate_all_scores():
    """Recalculate scores for all sites with current weights"""
    try:
        sites_data = await get_supabase_data("renewable_sites")
        scorer = SiteScorer(current_scoring_config)
        
        scores = []
        for site in sites_data:
            score = scorer.calculate_site_score(site, sites_data)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "message": f"Recalculated scores for {len(sites_data)} sites",
            "average_score": round(avg_score, 1),
            "high_scoring_sites": len([s for s in scores if s >= 80]),
            "current_weights": current_scoring_config
        }
        
    except Exception as e:
        logger.error(f"Error recalculating scores: {e}")
        raise HTTPException(status_code=500, detail=f"Error recalculating scores: {str(e)}")
