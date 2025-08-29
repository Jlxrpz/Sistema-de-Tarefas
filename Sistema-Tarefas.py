#!/usr/bin/env python3
# Sistema de Recomendação Simples

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from pathlib import Path


class RecommendationSystem:
    def __init__(self):
        self.movies = None
        self.ratings = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def load_sample_data(self):
        self.movies = pd.DataFrame({
            'movieId': range(1, 11),
            'title': [
                'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight',
                'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'The Godfather',
                'The Shawshank Redemption', 'The Silence of the Lambs'
            ],
            'genres': [
                'Action|Sci-Fi', 'Action|Sci-Fi|Thriller',
                'Adventure|Drama|Sci-Fi', 'Action|Crime|Drama',
                'Crime|Drama', 'Drama', 'Drama|Romance',
                'Crime|Drama', 'Drama', 'Crime|Drama|Thriller'
            ]
        })

        self.ratings = pd.DataFrame({
            'userId': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
            'movieId': [1, 2, 3, 4, 5, 1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
            'rating': [5, 4, 5, 5, 4, 4, 5, 3, 4, 5, 5, 4, 3, 5, 4]
        })

    def prepare_data(self):
        self.movies['content'] = (
            self.movies['title'] + ' ' +
            self.movies['genres'].str.replace('|', ' ')
        )
        tfidf_matrix = self.vectorizer.fit_transform(self.movies['content'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_user_profile(self, user_id):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        user_movies = user_ratings.merge(self.movies, on='movieId')
        return user_movies

    def recommend_movies(self, user_id, n_recommendations=5):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        rated_movies = user_ratings['movieId'].values

        movie_scores = {}
        for idx, movie_id in enumerate(self.movies['movieId']):
            if movie_id in rated_movies:
                continue

            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            sim_scores = sorted(
                sim_scores, key=lambda x: x[1], reverse=True
            )

            total_score = 0
            weight_sum = 0

            for i, (similar_idx, similarity) in enumerate(sim_scores[:10]):
                similar_movie_id = self.movies.iloc[similar_idx]['movieId']

                if similar_movie_id in rated_movies:
                    rating = user_ratings[
                        user_ratings['movieId'] == similar_movie_id
                    ]['rating'].values[0]
                    total_score += similarity * rating
                    weight_sum += similarity

            if weight_sum > 0:
                movie_scores[movie_id] = total_score / weight_sum

        recommended_movies = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )[:n_recommendations]

        recommendations = []
        for movie_id, score in recommended_movies:
            movie_title = self.movies[
                self.movies['movieId'] == movie_id
            ]['title'].values[0]
            recommendations.append({
                'movie': movie_title,
                'score': round(score, 2)
            })

        return recommendations

    def get_similar_movies(self, movie_title, n_similar=5):
        movie_idx = self.movies[self.movies['title'] == movie_title].index[0]
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        sim_scores = sorted(
            sim_scores, key=lambda x: x[1], reverse=True
        )[1:n_similar+1]

        similar_movies = []
        for idx, score in sim_scores:
            similar_movies.append({
                'title': self.movies.iloc[idx]['title'],
                'similarity': round(score, 2)
            })

        return similar_movies

    def save_model(self, filename='recommendation_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'similarity_matrix': self.similarity_matrix,
                'vectorizer': self.vectorizer,
                'movies': self.movies
            }, f)

    def load_model(self, filename='recommendation_model.pkl'):
        if Path(filename).exists():
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.similarity_matrix = data['similarity_matrix']
                self.vectorizer = data['vectorizer']
                self.movies = data['movies']


if __name__ == "__main__":
    rs = RecommendationSystem()
    rs.load_sample_data()
    rs.prepare_data()

    print("Recomendações para usuário 1:")
    recommendations = rs.recommend_movies(1)
    for rec in recommendations:
        print(f"{rec['movie']} - Score: {rec['score']}")

    print("\nFilmes similares a 'The Matrix':")
    similar = rs.get_similar_movies('The Matrix')
    for sim in similar:
        print(f"{sim['title']} - Similaridade: {sim['similarity']}")

    rs.save_model()
