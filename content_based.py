from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from recommender import MovieRecommender

class ContentBasedRecommender(MovieRecommender):
    def __init__(self):
        super().__init__()
        self._build_model()

    def _build_model(self):
        # Combine genres + tags into one text feature
        self.df["features"] = self.df["genres"].str.replace("|", " ") + " " + self.df["tags"]

        # Convert text to TF-IDF vectors
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.df["features"])

        # Compute cosine similarity between all movies
        self.similarity = cosine_similarity(tfidf_matrix)
        print("Content-based model built.")

    def recommend(self, title, n=5):
        matches = self.df[self.df["title"].str.lower() == title.lower()]
        if matches.empty:
            print(f"Movie '{title}' not found.")
            return

        idx = matches.index[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:n]

        print(f"\nBecause you liked '{title}', try:\n")
        for rank, (i, score) in enumerate(scores, 1):
            print(f"  {rank}. {self.df.iloc[i]['title']} (similarity: {score:.2f})")