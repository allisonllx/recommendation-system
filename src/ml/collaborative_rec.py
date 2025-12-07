from typing import List, Set
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringRecommender:
    """Item-based collaborative filtering"""
    
    def __init__(self, interaction_matrix: csr_matrix):
        self.interaction_matrix = interaction_matrix
        self.n_users, self.n_items = interaction_matrix.shape
        
        # Compute item-item similarity matrix
        print("Computing item similarity matrix...")
        self.item_similarity = cosine_similarity(interaction_matrix.T, dense_output=False)
    
    def recommend(self, user: int, n: int, exclude_items: Set[int] = None) -> List[int]:
        """Get top-N CF recommendations for user"""
        if exclude_items is None:
            exclude_items = set()
        
        # Get user's interaction history
        user_items = self.interaction_matrix[user].toarray().flatten()
        
        # Compute scores for all items
        scores = self.item_similarity.dot(user_items)
        
        # Sort and filter
        item_scores = [(i, scores[i]) for i in range(self.n_items) 
                      if i not in exclude_items and user_items[i] == 0]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in item_scores[:n]]