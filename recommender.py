import pandas as pd

class MovieRecommender:
    def __init__(self, data_path="data/movies.csv"):
        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df)} movies.")

    def get_movie_list(self):
        return self.df[["movie_id", "title", "genres"]].to_string(index=False)