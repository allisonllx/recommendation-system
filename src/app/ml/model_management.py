import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from typing import List, Optional, Tuple
import pandas as pd
import logging
import sys

BASE_DIR = Path(__file__).resolve().parents[1]  # -> <repo>/src
sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)

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