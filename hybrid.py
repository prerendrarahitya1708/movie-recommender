import numpy as np
from content_based import ContentBasedRecommender
from collaborative import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, cb_weight=0.5, collab_weight=0.5):
        print("Building Hybrid Recommender...\n")
        self.cb = ContentBasedRecommender()
        self.collab = CollaborativeRecommender()
        self.cb_weight = cb_weight
        self.collab_weight = collab_weight
        self.df = self.cb.df

    def _get_cb_scores(self, title):
        matches = self.df[self.df["title"].str.lower() == title.lower()]
        if matches.empty:
            return None, None
        idx = matches.index[0]
        scores = self.cb.similarity[idx]
        return scores, idx

    def _get_collab_scores(self, user):
        if user not in self.collab.ratings_matrix.index:
            return None
        user_idx = self.collab.ratings_matrix.index.get_loc(user)
        sim_scores = self.collab.user_similarity[user_idx].copy()
        sim_scores[user_idx] = 0
        weighted = np.dot(sim_scores, self.collab.normalized.values)
        return weighted

    def _normalize(self, scores):
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s == 0:
            return scores
        return (scores - min_s) / (max_s - min_s)

    def recommend(self, title, user="User1", n=5):
        cb_scores, cb_idx = self._get_cb_scores(title)
        if cb_scores is None:
            print(f"Movie '{title}' not found.")
            return

        collab_scores = self._get_collab_scores(user)
        if collab_scores is None:
            print(f"User '{user}' not found.")
            return

        # Normalize both score arrays to 0-1 range
        cb_norm = self._normalize(cb_scores)
        collab_norm = self._normalize(collab_scores)

        # Blend scores using weights
        hybrid_scores = (self.cb_weight * cb_norm) + (self.collab_weight * collab_norm)

        # Exclude the input movie itself
        hybrid_scores[cb_idx] = -999

        top_indices = np.argsort(hybrid_scores)[::-1][:n]

        print(f"\n🎬 Hybrid Recommendations")
        print(f"   Because you liked '{title}' and based on {user}'s taste:\n")
        for rank, idx in enumerate(top_indices, 1):
            movie = self.df.iloc[idx]
            print(f"  {rank}. {movie['title']} ({movie['genres']})")