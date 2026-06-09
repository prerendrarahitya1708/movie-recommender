from flask import Flask, render_template, request
from hybrid import HybridRecommender

app = Flask(__name__)
recommender = HybridRecommender(cb_weight=0.6, collab_weight=0.4)
movie_titles = recommender.df["title"].tolist()
users = ["User1", "User2", "User3", "User4", "User5"]

@app.route("/")
def index():
    return render_template("index.html", movies=movie_titles, users=users)

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form.get("title")
    user = request.form.get("user")

    results = []
    scores, cb_idx = recommender._get_cb_scores(title)
    collab_scores = recommender._get_collab_scores(user)

    if scores is not None and collab_scores is not None:
        import numpy as np
        cb_norm = recommender._normalize(scores)
        collab_norm = recommender._normalize(collab_scores)
        hybrid = (0.6 * cb_norm) + (0.4 * collab_norm)
        hybrid[cb_idx] = -999
        top_indices = np.argsort(hybrid)[::-1][:5]

        for idx in top_indices:
            movie = recommender.df.iloc[idx]
            results.append({
                "title": movie["title"],
                "genres": movie["genres"],
                "rating": movie["rating"],
                "score": round(float(hybrid[idx]), 3)
            })

    return render_template("results.html",
                           results=results,
                           title=title,
                           user=user)

if __name__ == "__main__":
    app.run(debug=True)