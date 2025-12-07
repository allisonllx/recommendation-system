from typing import List, Dict, Set

class PopularityRecommender:
    """Popularity-based recommender"""
    
    def __init__(self, popularity_scores: Dict[int, float]):
        self.popularity_scores = popularity_scores
        self.sorted_items = sorted(popularity_scores.items(), 
                                   key=lambda x: x[1], reverse=True)
    
    def recommend(self, user: int, n: int, exclude_items: Set[int] = None) -> List[int]:
        """Get top-N popular items"""
        if exclude_items is None:
            exclude_items = set()
        
        recommendations = []
        for item, _ in self.sorted_items:
            if item not in exclude_items:
                recommendations.append(item)
            if len(recommendations) >= n:
                break
        
        return recommendations