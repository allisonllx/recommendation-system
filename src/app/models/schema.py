from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

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