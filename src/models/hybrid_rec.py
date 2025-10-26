import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# Import baseline recommenders
from popularity_rec import PopularityRecommender
from collaborative_rec import CollaborativeFilteringRecommender
from item_centric_rec import ItemCentricAudienceFinder
# from sasrec_rec import SASRecRecommender, SASRec

# PART 1: Data Preprocessing

class DataPreprocessor:
    """Preprocesses interaction data for all recommendation methods"""
    
    def __init__(self, df: pd.DataFrame, cold_threshold: int = 10, cold_window_days: int = 7):
        """
        Args:
            df: DataFrame with columns [user_idx, item_idx, interaction_idx, timestamp]
            cold_threshold: Items with <= this many interactions are cold
            cold_window_days: Consider items cold in their first N days
        """
        self.df = df.sort_values('timestamp').copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        self.cold_threshold = cold_threshold
        self.cold_window_days = cold_window_days
        
        # Interaction weights: click=1, like=2, comment=3, share=4
        self.interaction_weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
        
        self.n_users = df['user_idx'].nunique()
        self.n_items = df['item_idx'].nunique()
        
    def identify_cold_items(self) -> Set[int]:
        """Identify cold-start items"""
        item_first_seen = self.df.groupby('item_idx')['timestamp'].min()
        latest_timestamp = self.df['timestamp'].max()
        item_counts = self.df['item_idx'].value_counts()
        
        cold_items = set()
        for item in self.df['item_idx'].unique():
            count = item_counts.get(item, 0)
            days_since_first = (latest_timestamp - item_first_seen[item]).days
            
            if count <= self.cold_threshold or days_since_first <= self.cold_window_days:
                cold_items.add(item)
                
        return cold_items
    
    def classify_users(self) -> Dict[int, str]:
        """Classify users by activity level"""
        user_counts = self.df['user_idx'].value_counts()
        user_classes = {}
        
        for user in self.df['user_idx'].unique():
            count = user_counts.get(user, 0)
            if count <= 5:
                user_classes[user] = 'cold'
            elif count <= 20:
                user_classes[user] = 'warm'
            else:
                user_classes[user] = 'hot'
        
        return user_classes
    
    def build_user_sequences(self, max_seq_len: int = 50) -> Dict[int, List[Tuple[int, int]]]:
        """Build user interaction sequences for SASRec"""
        user_sequences = defaultdict(list)
        
        for _, row in self.df.iterrows():
            user_sequences[row['user_idx']].append(
                (row['item_idx'], row['interaction_idx'])
            )
        
        for user in user_sequences:
            if len(user_sequences[user]) > max_seq_len:
                user_sequences[user] = user_sequences[user][-max_seq_len:]
                
        return dict(user_sequences)
    
    def build_interaction_matrix(self) -> csr_matrix:
        """Build user-item interaction matrix for collaborative filtering"""
        # Apply interaction weights
        df_weighted = self.df.copy()
        df_weighted['weight'] = df_weighted['interaction_idx'].map(self.interaction_weights)
        
        # Aggregate multiple interactions
        interaction_matrix = df_weighted.groupby(['user_idx', 'item_idx'])['weight'].sum().reset_index()
        
        # Create sparse matrix
        matrix = csr_matrix(
            (interaction_matrix['weight'], 
             (interaction_matrix['user_idx'], interaction_matrix['item_idx'])),
            shape=(self.n_users, self.n_items)
        )
        
        return matrix
    
    def compute_item_features(self) -> pd.DataFrame:
        """Compute item features for similarity matching"""
        item_features = []
        
        for item in self.df['item_idx'].unique():
            item_data = self.df[self.df['item_idx'] == item]
            total_interactions = len(item_data)
            interaction_dist = item_data['interaction_idx'].value_counts()
            
            features = {
                'item_idx': item,
                'total_interactions': total_interactions,
                'unique_users': item_data['user_idx'].nunique(),
                'click_rate': interaction_dist.get(0, 0) / total_interactions,
                'like_rate': interaction_dist.get(1, 0) / total_interactions,
                'comment_rate': interaction_dist.get(2, 0) / total_interactions,
                'share_rate': interaction_dist.get(3, 0) / total_interactions,
                'avg_engagement': item_data['interaction_idx'].mean(),
            }
            item_features.append(features)
            
        return pd.DataFrame(item_features).set_index('item_idx')
    
    def get_popularity_scores(self) -> Dict[int, float]:
        """Compute item popularity scores"""
        # Weighted by interaction type
        df_weighted = self.df.copy()
        df_weighted['weight'] = df_weighted['interaction_idx'].map(self.interaction_weights)
        
        popularity = df_weighted.groupby('item_idx')['weight'].sum()
        
        # Normalize
        max_pop = popularity.max()
        popularity = popularity / max_pop
        
        return popularity.to_dict()



# PART 2: Multi-Level Hybrid Recommender

class MultiLevelHybridRecommender:
    """
    Sophisticated hybrid system with multiple recommendation strategies
    
    Strategy Selection Based on Context:
    - Cold users: Popularity (70%) + CF (30%)
    - Warm users: CF (40%) + SASRec (40%) + Popularity (20%)
    - Hot users: SASRec (50%) + CF (30%) + Item-centric exploration (20%)
    - All: Cold item exploration when applicable
    """
    
    def __init__(self,
                 sasrec_model, # SASRecRecommender instance
                 cf_recommender: CollaborativeFilteringRecommender,
                 popularity_recommender: PopularityRecommender,
                 audience_finder: ItemCentricAudienceFinder,
                 user_classes: Dict[int, str],
                 cold_items: Set[int],
                 user_sequences: Dict[int, List[Tuple[int, int]]]):
        """
        Args:
            sasrec_model: SASRecRecommender instance from sasrec_rec.py
            user_sequences: Dict mapping user_idx to their interaction sequences
                           Required for SASRec predictions
        """
        self.sasrec = sasrec_model
        self.cf = cf_recommender
        self.popularity = popularity_recommender
        self.audience_finder = audience_finder
        self.user_classes = user_classes
        self.cold_items = cold_items
        self.user_sequences = user_sequences  # Store for SASRec
    
    def recommend(self, user: int, n_recommendations: int = 10, 
                  user_history: Optional[Set[int]] = None) -> List[Tuple[int, str]]:
        """
        Generate recommendations with source tracking
        
        Returns:
            List of (item_id, source) tuples where source is the recommender used
        """
        if user_history is None:
            user_history = set()
        
        user_class = self.user_classes.get(user, 'cold')
        
        # Determine strategy based on user class
        if user_class == 'cold':
            strategy = {
                'popularity': 0.7,
                'cf': 0.3,
                'sasrec': 0.0,
                'exploration': 0.0
            }
        elif user_class == 'warm':
            strategy = {
                'popularity': 0.2,
                'cf': 0.4,
                'sasrec': 0.4,
                'exploration': 0.0
            }
        else:  # hot
            strategy = {
                'popularity': 0.0,
                'cf': 0.3,
                'sasrec': 0.5,
                'exploration': 0.2
            }
        
        # Calculate number of recommendations from each source
        recommendations = []
        
        # 1. Popularity-based
        n_pop = int(n_recommendations * strategy['popularity'])
        if n_pop > 0:
            pop_recs = self.popularity.recommend(user, n_pop, exclude_items=user_history)
            recommendations.extend([(item, 'popularity') for item in pop_recs])
        
        # 2. Collaborative Filtering
        n_cf = int(n_recommendations * strategy['cf'])
        if n_cf > 0:
            cf_recs = self.cf.recommend(user, n_cf, exclude_items=user_history)
            recommendations.extend([(item, 'cf') for item in cf_recs])
        
        # 3. SASRec
        n_sasrec = int(n_recommendations * strategy['sasrec'])
        if n_sasrec > 0:
            sasrec_recs = self._get_sasrec_recommendations(user, n_sasrec, user_history)
            recommendations.extend([(item, 'sasrec') for item in sasrec_recs])
        
        # 4. Item-centric exploration
        n_explore = int(n_recommendations * strategy['exploration'])
        if n_explore > 0:
            explore_recs = self._get_exploration_items(user, n_explore, user_history)
            recommendations.extend([(item, 'exploration') for item in explore_recs])
        
        # Remove duplicates (keep first occurrence)
        seen = set()
        unique_recs = []
        for item, source in recommendations:
            if item not in seen:
                seen.add(item)
                unique_recs.append((item, source))
        
        # Fill up to n_recommendations if needed with fallback
        if len(unique_recs) < n_recommendations:
            fallback = self.popularity.recommend(
                user, 
                n_recommendations - len(unique_recs),
                exclude_items=seen.union(user_history)
            )
            unique_recs.extend([(item, 'fallback') for item in fallback])
        
        return unique_recs[:n_recommendations]
    
    def _get_sasrec_recommendations(self, user: int, n: int, 
                                   exclude: Set[int]) -> List[int]:
        """Get recommendations from SASRec using user's actual sequence"""
        # Get user's interaction sequence
        user_sequence = self.user_sequences.get(user, [])
        
        if not user_sequence:
            # No sequence available (i.e. user is new), return empty
            return []
        
        # Use SASRecRecommender.predict() method
        predictions = self.sasrec.predict(
            user_sequence=user_sequence,
            top_k=n * 2,  # Get extra to account for filtering
            exclude_items=exclude
        )
        
        return predictions[:n]
    
    def _get_exploration_items(self, user: int, n: int, 
                              exclude: Set[int]) -> List[int]:
        """Get cold items for exploration"""
        eligible_cold_items = []
        
        for cold_item in self.cold_items:
            if cold_item in exclude:
                continue
            
            suitable_audience = self.audience_finder.find_suitable_audience(
                cold_item, top_n=100
            )
            
            if user in suitable_audience:
                rank = suitable_audience.index(user)
                score = 1.0 / (rank + 1)
                eligible_cold_items.append((cold_item, score))
        
        eligible_cold_items.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in eligible_cold_items[:n]]
    
    def explain_recommendation(self, user: int, item: int) -> str:
        """Explain why an item was recommended"""
        user_class = self.user_classes.get(user, 'cold')
        
        explanation = f"User class: {user_class}\n"
        
        if item in self.cold_items:
            explanation += "This is a cold-start item selected through item-centric exploration.\n"
            similar = self.audience_finder.find_similar_items(item, top_k=3)
            explanation += f"Similar items you've enjoyed: {similar}\n"
        else:
            explanation += "This item was recommended based on:\n"
            explanation += "- Your interaction history (CF)\n"
            explanation += "- Sequential patterns (SASRec)\n"
            explanation += "- Overall popularity trends\n"
        
        return explanation



# PART 3: Main Pipeline

def run_full_pipeline(df: pd.DataFrame, sasrec_model):
    """Complete multi-level hybrid recommendation pipeline"""
    
    print("="*70)
    print("STEP 1: Data Preprocessing")
    print("="*70)
    preprocessor = DataPreprocessor(df, cold_threshold=10, cold_window_days=7)
    
    cold_items = preprocessor.identify_cold_items()
    user_classes = preprocessor.classify_users()
    user_sequences = preprocessor.build_user_sequences(max_seq_len=50)
    interaction_matrix = preprocessor.build_interaction_matrix()
    item_features = preprocessor.compute_item_features()
    popularity_scores = preprocessor.get_popularity_scores()
    
    print(f"✓ Found {len(cold_items)} cold items")
    print(f"✓ User distribution: {dict(pd.Series(user_classes).value_counts())}")
    print(f"✓ Built sequences for {len(user_sequences)} users")
    print(f"✓ Interaction matrix: {interaction_matrix.shape}")
    
    print("\n" + "="*70)
    print("STEP 2: Building Recommender Components")
    print("="*70)
    
    popularity_rec = PopularityRecommender(popularity_scores)
    print("✓ Popularity recommender ready")
    
    cf_rec = CollaborativeFilteringRecommender(interaction_matrix)
    print("✓ Collaborative filtering ready")
    
    audience_finder = ItemCentricAudienceFinder(item_features, user_sequences)
    print("✓ Item-centric audience finder ready")
    
    print("\n" + "="*70)
    print("STEP 3: Creating Multi-Level Hybrid System")
    print("="*70)
    
    hybrid = MultiLevelHybridRecommender(
        sasrec_model=sasrec_model,
        cf_recommender=cf_rec,
        popularity_recommender=popularity_rec,
        audience_finder=audience_finder,
        user_classes=user_classes,
        cold_items=cold_items,
        user_sequences=user_sequences 
    )
    print("✓ Hybrid recommender initialized")
    
    print("\n" + "="*70)
    print("STEP 4: Example Recommendations")
    print("="*70)
    
    # Demo for different user types
    for user_type in ['cold', 'warm', 'hot']:
        users_of_type = [u for u, c in user_classes.items() if c == user_type]
        if users_of_type:
            user = users_of_type[0]
            recs = hybrid.recommend(user, n_recommendations=10)
            
            print(f"\n{user_type.upper()} User {user}:")
            source_counts = {}
            for item, source in recs:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print(f"  Items: {[item for item, _ in recs[:5]]}...")
            print(f"  Sources: {source_counts}")
    
    return hybrid



if __name__ == "__main__":
    print("="*70)
    print("HYBRID RECOMMENDER SYSTEM PIPELINE")
    print("="*70)
    
    # Load data
    print("\n📊 Loading interaction data...")
    df = pd.read_csv("/Users/allisonlawlixuan/Documents/repos/recommendation_system/src/data/mock_interactions.csv")
    
    # Load pre-trained SASRec model
    print("\n🔧 Loading SASRec model...")
    try:
        from sasrec_rec import load_sasrec_recommender
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sasrec = load_sasrec_recommender('./weights/sasrec_recommender.pt', device)
        print("✓ Loaded pre-trained SASRec model")
    except FileNotFoundError:
        print("⚠️  No pre-trained model found. Train SASRec first with sasrec_rec.py")
        print("   Then run this script again.")
        exit(1)
    
    # Build hybrid system
    print("\n🏗️  Building hybrid recommendation system...")
    hybrid = run_full_pipeline(df, sasrec)
    
    # Generate sample recommendations
    print("\n" + "="*70)
    print("EXAMPLE RECOMMENDATIONS")
    print("="*70)
    
    # Show recommendations for different user types
    for user_type in ['cold', 'warm', 'hot']:
        users_of_type = [u for u, c in hybrid.user_classes.items() if c == user_type]
        if users_of_type:
            user = users_of_type[0]
            recs = hybrid.recommend(user, n_recommendations=10)
            
            print(f"\n{user_type.upper()} User {user}:")
            source_counts = {}
            for item, source in recs:
                source_counts[source] = source_counts.get(source, 0) + 1
            
            print(f"  Items: {[item for item, _ in recs[:5]]}...")
            print(f"  Sources: {source_counts}")
            
            # Show explanation for first item
            if recs:
                explanation = hybrid.explain_recommendation(user, recs[0][0])
                print(f"  Why item {recs[0][0]}? {explanation.split(chr(10))[0]}")
    
    print("\n" + "="*70)
    print("SYSTEM READY!")
    print("="*70)
    print("\nUsage:")
    print("  recs = hybrid.recommend(user_id, n_recommendations=10)")
    print("  explanation = hybrid.explain_recommendation(user_id, item_id)")
    print("\nSource distribution across all users:")
    
    # Analyze source distribution
    all_sources = []
    sample_users = list(hybrid.user_classes.keys())[:100]  # Sample 100 users
    for user in sample_users:
        recs = hybrid.recommend(user, n_recommendations=10)
        all_sources.extend([source for _, source in recs])
    
    source_dist = {}
    for source in all_sources:
        source_dist[source] = source_dist.get(source, 0) + 1
    
    total = sum(source_dist.values())
    for source, count in sorted(source_dist.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}/{total} ({100*count/total:.1f}%)")