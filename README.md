# 🎬 Hybrid Movie Recommender System

A Flask-based Movie Recommendation System that combines **Content-Based Filtering** and **Collaborative Filtering** to generate personalized movie recommendations.

Users can select a movie and a user profile through a web interface, and the system recommends movies using a hybrid scoring approach.

---

## 🚀 Features

* Content-Based Recommendation using TF-IDF and Cosine Similarity
* Collaborative Filtering using User-User Similarity
* Hybrid Recommendation Engine
* Interactive Flask Web Interface
* Displays movie genres, ratings, and recommendation scores
* Supports multiple user profiles

---

## 🛠️ Technologies Used

* Python
* Flask
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS

---

## 📂 Project Structure

movie-recommender/

├── app.py

├── main.py

├── recommender.py

├── content_based.py

├── collaborative.py

├── hybrid.py

├── data/

│ └── movies.csv

├── templates/

│ ├── index.html

│ └── results.html

├── requirements.txt

└── README.md

---

## ⚙️ How It Works

### Content-Based Filtering

The system combines movie genres and tags into a single feature vector and uses:

* TF-IDF Vectorization
* Cosine Similarity

to find movies that are similar to the selected movie.

### Collaborative Filtering

A simulated user-rating matrix is generated and used to:

* Compute user-user similarity
* Predict unseen movie preferences
* Recommend movies based on similar users

### Hybrid Recommendation

The final recommendation score is computed using:

Hybrid Score =

0.6 × Content-Based Score

*

0.4 × Collaborative Score

The movies with the highest scores are recommended to the user.

---

## 📊 Dataset

The dataset contains 15 popular movies with:

* Movie ID
* Title
* Genres
* Tags
* IMDb-style Rating
* Vote Count

Example Movies:

* The Dark Knight
* Inception
* Interstellar
* The Matrix
* The Godfather
* Parasite
* Joker

---

## ▶️ Installation

Clone the repository:

git clone https://github.com/prerendrarahitya1708/movie-recommender.git

Move into the project folder:

cd movie-recommender

Create a virtual environment:

python -m venv venv

Activate the environment:

Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open your browser and visit:

http://127.0.0.1:5000

---

## 🎯 Usage

1. Select a movie from the dropdown menu.
2. Select a user profile.
3. Click "Get Recommendations".
4. View the top recommended movies along with:

   * Genre
   * Rating
   * Recommendation Score

---

## 🔮 Future Enhancements

* Real MovieLens Dataset Integration
* User Authentication
* Movie Poster Support
* IMDb API Integration
* Deep Learning Based Recommendations
* Real User Rating Collection

---

## 👨‍💻 Author

Prerendra Rahitya

GitHub:
https://github.com/prerendrarahitya1708
