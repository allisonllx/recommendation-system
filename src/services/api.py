from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import logging
import sys
from datetime import datetime
from datetime import datetime, timezone
from pathlib import Path
import json
from contextlib import asynccontextmanager

# Import request/response schemas
from schema import (
    RecommendationRequest,
    RecommendationItem,
    RecommendationResponse,
    ExplanationRequest,
    ExplanationResponse,
    InteractionLog,
    InteractionResponse,
    HealthResponse,
    ModelStatsResponse
)

# Import model manager
from model_management import HybridModelManager

BASE_DIR = Path(__file__).resolve().parents[1]  # -> <repo>/src
sys.path.insert(0, str(BASE_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan (startup and shutdown)"""
    global model_manager
    
    # Startup
    logger.info("="*70)
    logger.info("Starting Hybrid Recommendation System API")
    logger.info("="*70)
    
    try:
        model_manager = HybridModelManager(MODEL_DIR, DATA_PATH)
        
        if not model_manager.is_healthy()['model_loaded']:
            logger.warning("⚠️  Model not fully loaded! API will return errors for some requests.")
            logger.warning("   Check server logs for details")
        else:
            stats = model_manager.get_stats()
            logger.info("✓ Hybrid recommender system loaded successfully!")
            logger.info(f"  - Users: {stats['total_users']}")
            logger.info(f"  - Items: {stats['total_items']}")
            logger.info(f"  - Cold items: {stats['cold_items']}")
            logger.info(f"  - User distribution: {stats['user_distribution']}")
            logger.info(f"  - Device: {model_manager.device}")
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize model manager: {str(e)}")
        logger.error("   Check MODEL_DIR and DATA_PATH configuration")
        model_manager = None
    
    logger.info("="*70)
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("Shutting down Hybrid Recommendation System API...")

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Recommendation System API",
    description="API for multi-level hybrid recommendation system with SASRec, CF, and item-centric exploration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
MODEL_DIR = str(BASE_DIR / "models")
DATA_PATH = str(BASE_DIR / "data" / "mock_interactions.csv")

try:
    model_manager = HybridModelManager(MODEL_DIR, DATA_PATH)
except Exception as e:
    logger.error(f"Failed to initialize model manager: {str(e)}")
    model_manager = None

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Hybrid Recommendation System API",
        "version": "1.0.0",
        "description": "Multi-level hybrid recommendations with SASRec, CF, and item-centric exploration",
        "endpoints": {
            "recommendations": "/recommend",
            "explanation": "/explain",
            "interactions": "/interactions",
            "stats": "/stats",
            "health": "/healthz",
            "docs": "/docs"
        }
    }

@app.get("/healthz", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    if model_manager is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            components_loaded={},
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    components = model_manager.is_healthy()
    is_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=components['model_loaded'],
        components_loaded=components,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized recommendations for a user
    
    - **user_id**: User identifier (integer)
    - **n_recommendations**: Number of recommendations to return (1-100)
    - **exclude_items**: Optional list of item IDs to exclude from recommendations
    - **include_sources**: Include source information for each recommendation
    
    Returns recommendations with sources:
    - **sasrec**: Sequential recommendation from SASRec
    - **cf**: Collaborative filtering recommendation
    - **popularity**: Popular items
    - **exploration**: Cold-start item exploration
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    logger.info(f"Recommendation request for user: {request.user_id}")
    
    recommendations, user_class = model_manager.get_recommendations(
        user_id=request.user_id,
        n_recommendations=request.n_recommendations,
        exclude_items=request.exclude_items
    )
    
    # Format recommendations
    rec_items = []
    source_distribution = {}
    
    for rank, (item_id, source) in enumerate(recommendations, start=1):
        rec_items.append(RecommendationItem(
            item_id=item_id,
            source=source,
            rank=rank
        ))
        source_distribution[source] = source_distribution.get(source, 0) + 1
    
    return RecommendationResponse(
        user_id=request.user_id,
        user_class=user_class,
        recommendations=rec_items,
        source_distribution=source_distribution,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_version=model_manager.model_version
    )

@app.post("/explain", response_model=ExplanationResponse, tags=["Recommendations"])
async def explain_recommendation(request: ExplanationRequest):
    """
    Get explanation for why an item was recommended to a user
    
    - **user_id**: User identifier
    - **item_id**: Item identifier
    """
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    logger.info(f"Explanation request: user {request.user_id}, item {request.item_id}")
    
    explanation = model_manager.get_explanation(
        user_id=request.user_id,
        item_id=request.item_id
    )
    
    return ExplanationResponse(
        user_id=request.user_id,
        item_id=request.item_id,
        explanation=explanation,
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.get("/stats", response_model=ModelStatsResponse, tags=["Statistics"])
async def get_model_stats():
    """Get statistics about the recommendation model"""
    if model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not initialized"
        )
    
    stats = model_manager.get_stats()
    
    return ModelStatsResponse(
        total_users=stats['total_users'],
        total_items=stats['total_items'],
        cold_items=stats['cold_items'],
        user_distribution=stats['user_distribution'],
        timestamp=datetime.now(timezone.utc).isoformat()
    )

@app.post("/interactions", response_model=InteractionResponse, tags=["Interactions"])
async def log_interaction(interaction: InteractionLog):
    """
    Log a new user-item interaction
    
    - **user_id**: User identifier
    - **item_id**: Item identifier
    - **interaction_idx**: Type of interaction
        - 0 = click
        - 1 = like
        - 2 = comment
        - 3 = share
    - **timestamp**: Optional timestamp (auto-generated if not provided)
    
    Note: This endpoint logs interactions for future model retraining.
    It does not update the current model in real-time.
    """
    # Add timestamp if not provided
    if interaction.timestamp is None:
        interaction.timestamp = datetime.now(timezone.utc).isoformat()
    
    logger.info(
        f"Logging interaction: user {interaction.user_id} -> item {interaction.item_id} "
        f"(type: {interaction.interaction_idx})"
    )
    
    try:
        # Log to file for batch retraining
        log_file = Path('data/new_interactions.jsonl')
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'a') as f:
            interaction_data = {
                'user_id': interaction.user_id,
                'item_id': interaction.item_id,
                'interaction_idx': interaction.interaction_idx,
                'timestamp': interaction.timestamp
            }
            f.write(json.dumps(interaction_data) + '\n')
        
        return InteractionResponse(
            status="success",
            message=f"Interaction logged successfully. Will be included in next model update."
        )
    
    except Exception as e:
        logger.error(f"Error logging interaction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error logging interaction: {str(e)}"
        )

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    logger.info(f"Starting server on {HOST}:{PORT}")
    logger.info(f"Documentation available at http://{HOST}:{PORT}/docs")
    
    uvicorn.run(
        app, 
        host=HOST, 
        port=PORT,
        log_level="info"
    )