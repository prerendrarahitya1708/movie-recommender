import numpy as np
import pandas as pd
from recommender import MovieRecommender

class CollaborativeRecommender(MovieRecommender):
    def __init__(self):
        super().__init__()
        self._build_user_ratings()
        self._build_model()

    def _build_user_ratings(self):
        # Simulated user ratings matrix (users x movies)
        np.random.seed(42)
        num_users = 10
        num_movies = len(self.df)

        ratings = np.random.randint(1, 10, size=(num_users, num_movies)).astype(float)

        # Simulate "unseen" movies by zeroing ~40% of entries
        mask = np.random.rand(num_users, num_movies) < 0.4
        ratings[mask] = 0

        self.ratings_matrix = pd.DataFrame(
            ratings,
            columns=self.df["title"],
            index=[f"User{i+1}" for i in range(num_users)]
        )
        print("User ratings matrix created.")

    def _build_model(self):
        # Normalize ratings by subtracting each user's mean
        matrix = self.ratings_matrix.copy().replace(0, np.nan)
        self.normalized = matrix.subtract(matrix.mean(axis=1), axis=0).fillna(0)

        # Compute user-user similarity via dot product
        norm = self.normalized.values
        self.user_similarity = np.dot(norm, norm.T)
        print("Collaborative model built.")

    def recommend(self, user="User1", n=5):
        if user not in self.ratings_matrix.index:
            print(f"User '{user}' not found.")
            return

        user_idx = self.ratings_matrix.index.get_loc(user)
        sim_scores = self.user_similarity[user_idx]

        # Weighted sum of other users' ratings
        sim_scores[user_idx] = 0  # exclude self
        weighted = np.dot(sim_scores, self.normalized.values)

        # Only recommend unrated movies
        already_rated = self.ratings_matrix.iloc[user_idx] > 0
        weighted[already_rated.values] = -999

        top_indices = np.argsort(weighted)[::-1][:n]

        print(f"\nRecommendations for {user}:\n")
        for rank, idx in enumerate(top_indices, 1):
            title = self.df.iloc[idx]["title"]
            print(f"  {rank}. {title}")