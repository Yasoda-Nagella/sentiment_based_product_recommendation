import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

def get_recommendations(input):
    df_pivot = train.pivot(index='reviews_username',columns='product_id',values='reviews_rating')
    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T - mean).T
    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    user_correlation[user_correlation < 0] = 0
    dummy_train = train.copy()
    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x >= 1 else 1)
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='product_id',
        values='reviews_rating'
    ).fillna(1)
    user_final_rating = np.multiply(user_predicted_ratings, dummy_train)
    d = user_final_rating.loc[input].sort_values(ascending=False)[0:5]
    return d.to_json(orient='table')


app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/recommend")

def recommend():
    user_name = request.args.get('user_name')
    user_name = user_name.lower()
    return get_recommendations(user_name)

if __name__ == '__main__':
    app.debug = True
    app.run()