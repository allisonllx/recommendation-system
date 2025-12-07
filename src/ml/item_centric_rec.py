import numpy as np
import pandas as pd
from typing import Dict, List

class ItemCentricAudienceFinder:
    """Finds suitable audience for cold-start items"""
    
    def __init__(self, item_features: pd.DataFrame, user_sequences: Dict):
        self.item_features = item_features
        self.user_sequences = user_sequences
        self.user_profiles = self._build_user_profiles()
        
    def _build_user_profiles(self) -> Dict[int, Dict]:
        """Build user preference profiles"""
        user_profiles = {}
        
        for user, sequence in self.user_sequences.items():
            items = [item for item, _ in sequence]
            interactions = [inter for _, inter in sequence]
            
            profile = {
                'interacted_items': set(items),
                'avg_engagement': np.mean(interactions),
                'engagement_style': self._classify_engagement_style(interactions),
                'recent_items': items[-10:],
            }
            user_profiles[user] = profile
            
        return user_profiles
    
    def _classify_engagement_style(self, interactions: List[int]) -> str:
        """Classify user engagement style"""
        avg = np.mean(interactions)
        if avg < 0.5:
            return 'passive'
        elif avg < 1.5:
            return 'active'
        else:
            return 'super_engaged'
    
    def find_similar_items(self, cold_item: int, top_k: int = 10) -> List[int]:
        """Find warm items similar to cold item"""
        if cold_item not in self.item_features.index:
            return []
        
        cold_features = self.item_features.loc[cold_item]
        feature_cols = ['click_rate', 'like_rate', 'comment_rate', 'share_rate', 'avg_engagement']
        cold_vec = cold_features[feature_cols].values
        
        similarities = []
        for item in self.item_features.index:
            if item == cold_item:
                continue
            item_vec = self.item_features.loc[item][feature_cols].values
            sim = np.dot(cold_vec, item_vec) / (np.linalg.norm(cold_vec) * np.linalg.norm(item_vec) + 1e-8)
            similarities.append((item, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in similarities[:top_k]]
    
    def find_suitable_audience(self, cold_item: int, top_n: int = 100) -> List[int]:
        """Find top-N users most likely to engage with cold item"""
        similar_items = self.find_similar_items(cold_item, top_k=10)
        
        if not similar_items:
            return self._get_most_active_users(top_n)
        
        user_scores = []
        for user, profile in self.user_profiles.items():
            score = 0
            
            similar_interactions = len(profile['interacted_items'].intersection(similar_items))
            score += similar_interactions * 2.0
            
            if profile['engagement_style'] == 'super_engaged':
                score += 1.0
            elif profile['engagement_style'] == 'active':
                score += 0.5
            
            recent_similar = len(set(profile['recent_items']).intersection(similar_items))
            score += recent_similar * 1.5
            
            if cold_item in profile['interacted_items']:
                score = 0
            
            user_scores.append((user, score))
        
        user_scores.sort(key=lambda x: x[1], reverse=True)
        return [user for user, _ in user_scores[:top_n]]
    
    def _get_most_active_users(self, top_n: int) -> List[int]:
        """Fallback: return most active users"""
        user_activity = [(user, len(seq)) for user, seq in self.user_sequences.items()]
        user_activity.sort(key=lambda x: x[1], reverse=True)
        return [user for user, _ in user_activity[:top_n]]