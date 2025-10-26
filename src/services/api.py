from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
import torch
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hybrid Recommendation System API",
    description="API for multi-level hybrid recommendation system with SASRec, CF, and item-centric exploration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# REQUEST/RESPONSE SCHEMAS (shift to schema.py)
# ============================================================================

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    n_recommendations: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    exclude_items: Optional[List[int]] = Field(default=None, description="Item IDs to exclude")
    include_sources: bool = Field(default=True, description="Include recommendation sources")

class RecommendationItem(BaseModel):
    item_id: int
    source: str = Field(..., description="Recommendation source: sasrec, cf, popularity, or exploration")
    rank: int = Field(..., description="Ranking position (1-based)")

class RecommendationResponse(BaseModel):
    user_id: int
    user_class: str = Field(..., description="User classification: cold, warm, or hot")
    recommendations: List[RecommendationItem]
    source_distribution: dict = Field(..., description="Count of recommendations by source")
    timestamp: str
    model_version: str

class ExplanationRequest(BaseModel):
    user_id: int
    item_id: int

class ExplanationResponse(BaseModel):
    user_id: int
    item_id: int
    explanation: str
    timestamp: str

class InteractionLog(BaseModel):
    user_id: int
    item_id: int
    interaction_idx: int = Field(..., description="0=click, 1=like, 2=comment, 3=share")
    timestamp: Optional[str] = None

class InteractionResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    components_loaded: dict
    timestamp: str

class ModelStatsResponse(BaseModel):
    total_users: int
    total_items: int
    cold_items: int
    user_distribution: dict
    timestamp: str

# ============================================================================
# MODEL MANAGEMENT (shift to model_management.py)
# ============================================================================

class HybridModelManager:
    """Manages the hybrid recommender system components"""
    
    def __init__(self, model_dir: str, data_path: str):
        self.model_dir = Path(model_dir)
        self.data_path = Path(data_path)
        self.hybrid_recommender = None
        self.sasrec = None
        self.df = None
        self.model_version = "1.0.0"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_model()
    
    def load_model(self):
        """Load all hybrid recommender components"""
        try:
            logger.info(f"Loading hybrid recommender from {self.model_dir}")
            
            # Import required modules
            from models.sasrec_rec import load_sasrec_recommender
            from models.hybrid_rec import (
                DataPreprocessor,
                PopularityRecommender,
                CollaborativeFilteringRecommender,
                ItemCentricAudienceFinder,
                MultiLevelHybridRecommender
            )
            
            # 1. Load interaction data
            logger.info(f"Loading interaction data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} interactions")
            
            # 2. Load SASRec model
            sasrec_path = self.model_dir / '../models/weights/sasrec_recommender.pt'
            logger.info(f"Loading SASRec model from {sasrec_path}")
            self.sasrec = load_sasrec_recommender(str(sasrec_path), self.device)
            logger.info("✓ SASRec loaded")
            
            # 3. Build hybrid system components
            logger.info("Building hybrid system components...")
            preprocessor = DataPreprocessor(self.df, cold_threshold=10, cold_window_days=7)
            
            cold_items = preprocessor.identify_cold_items()
            user_classes = preprocessor.classify_users()
            user_sequences = preprocessor.build_user_sequences(max_seq_len=50)
            interaction_matrix = preprocessor.build_interaction_matrix()
            item_features = preprocessor.compute_item_features()
            popularity_scores = preprocessor.get_popularity_scores()
            
            logger.info(f"✓ Found {len(cold_items)} cold items")
            logger.info(f"✓ User distribution: {dict(pd.Series(user_classes).value_counts())}")
            
            # 4. Build recommender components
            popularity_rec = PopularityRecommender(popularity_scores)
            cf_rec = CollaborativeFilteringRecommender(interaction_matrix)
            audience_finder = ItemCentricAudienceFinder(item_features, user_sequences)
            
            logger.info("✓ All components built")
            
            # 5. Create hybrid recommender
            self.hybrid_recommender = MultiLevelHybridRecommender(
                sasrec_model=self.sasrec,
                cf_recommender=cf_rec,
                popularity_recommender=popularity_rec,
                audience_finder=audience_finder,
                user_classes=user_classes,
                cold_items=cold_items,
                user_sequences=user_sequences
            )
            
            logger.info("✓ Hybrid recommender system loaded successfully!")
            
        except FileNotFoundError as e:
            logger.error(f"Required files not found: {str(e)}")
            self.hybrid_recommender = None
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.hybrid_recommender = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.exception(e)
            self.hybrid_recommender = None
    
    def get_recommendations(
        self, 
        user_id: int, 
        n_recommendations: int = 10,
        exclude_items: Optional[List[int]] = None
    ) -> Tuple[List[Tuple[int, str]], str]:
        """
        Get recommendations from the hybrid model
        
        Returns:
            Tuple of (recommendations, user_class)
            recommendations: List of (item_id, source) tuples
            user_class: User classification (cold/warm/hot)
        """
        if self.hybrid_recommender is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Check server logs for details."
            )
        
        try:
            # Convert exclude_items to set
            if exclude_items is None:
                exclude_items_set = set()
            else:
                exclude_items_set = set(exclude_items)
            
            # Get recommendations with sources
            recommendations = self.hybrid_recommender.recommend(
                user=user_id,
                n_recommendations=n_recommendations,
                user_history=exclude_items_set
            )
            
            # Get user class
            user_class = self.hybrid_recommender.user_classes.get(user_id, 'cold')
            
            return recommendations, user_class
        
        except KeyError:
            # User not in training data - treat as cold start
            logger.warning(f"User {user_id} not found in training data. Treating as cold start.")
            
            # Use popularity-based recommendations for unknown users
            popularity_recs = self.hybrid_recommender.popularity.recommend(
                user_id, 
                n_recommendations,
                exclude_items=exclude_items_set if exclude_items else set()
            )
            
            return [(item, 'popularity') for item in popularity_recs], 'cold'
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            logger.exception(e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating recommendations: {str(e)}"
            )
    
    def get_explanation(self, user_id: int, item_id: int) -> str:
        """Get explanation for why an item was recommended"""
        if self.hybrid_recommender is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        try:
            explanation = self.hybrid_recommender.explain_recommendation(user_id, item_id)
            return explanation
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"Unable to generate explanation: {str(e)}"
    
    def get_stats(self) -> dict:
        """Get model statistics"""
        if self.hybrid_recommender is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        user_dist = pd.Series(self.hybrid_recommender.user_classes).value_counts().to_dict()
        
        return {
            'total_users': len(self.hybrid_recommender.user_classes),
            'total_items': self.df['item_idx'].nunique() if self.df is not None else 0,
            'cold_items': len(self.hybrid_recommender.cold_items),
            'user_distribution': user_dist
        }
    
    def is_healthy(self) -> dict:
        """Check health of all components"""
        return {
            'model_loaded': self.hybrid_recommender is not None,
            'sasrec_loaded': self.sasrec is not None,
            'data_loaded': self.df is not None,
            'device': str(self.device)
        }

# Initialize model manager
# Update these paths to match your setup
MODEL_DIR = "models/"
DATA_PATH = "data/interactions.csv"

try:
    model_manager = HybridModelManager(MODEL_DIR, DATA_PATH)
except Exception as e:
    logger.error(f"Failed to initialize model manager: {str(e)}")
    model_manager = None

# ============================================================================
# API ENDPOINTS
# ============================================================================

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
            timestamp=datetime.utcnow().isoformat()
        )
    
    components = model_manager.is_healthy()
    is_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=components['model_loaded'],
        components_loaded=components,
        timestamp=datetime.utcnow().isoformat()
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
        timestamp=datetime.utcnow().isoformat(),
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
        timestamp=datetime.utcnow().isoformat()
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
        timestamp=datetime.utcnow().isoformat()
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
        interaction.timestamp = datetime.now(datetime.timezone.utc).isoformat()
    
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
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("Starting Hybrid Recommendation System API")
    logger.info("="*70)
    
    if model_manager is None:
        logger.error("❌ Model manager not initialized!")
        logger.error("   Check MODEL_DIR and DATA_PATH configuration")
    elif not model_manager.is_healthy()['model_loaded']:
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
    
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Hybrid Recommendation System API...")

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