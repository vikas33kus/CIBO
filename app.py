from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import pandas as pd
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)


# Load the pre-trained model
tfidf_vectorizer = load('tfidf_vectorizer.joblib')
tfidf_matrix = load('tfidf_matrix.joblib')

@app.route('/recommend', methods=['POST'])
def recommend_dish():
    try:
        data = request.get_json()
        ingredients = data['ingredients']
        df=pd.read_csv("cleaned_file.csv")

        # Use the model for recommendation
        user_idf = tfidf_vectorizer.transform([ingredients])
        sim_ing = cosine_similarity(user_idf, tfidf_matrix)
        li=sorted(list(enumerate(sim_ing[0])),reverse=True,key=lambda x:x[1])[0:5]
        li
        indices = [index for index, _ in li]
        newdf=df.loc[indices]
        json=newdf.to_json(orient='records')
        return json

    except Exception as e:
        # Log the exception details
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500
    
def get_recipe_image():
    try:
        recipe_name = request.args.get('recipe_name')
        app_id = 'ebb6041c'  # Replace with your Edamam app ID
        app_key = '3c33ad913ab23b8554082bfb5fdd78b5'  # Replace with your Edamam app key

        url = f"https://api.edamam.com/search?q={recipe_name}&app_id={app_id}&app_key={app_key}"
        response = requests.get(url)
        data = response.json()

        # Check if there are hits and if the first hit has an image
        if 'hits' in data and data['hits'] and 'image' in data['hits'][0]['recipe']:
            print("before gettinh url")
            image_url = data['hits'][0]['recipe']['image']
            print("after gettinh url")
            return jsonify({'image_url': image_url})
        else:
            return jsonify({'error': 'Recipe image not found'}), 404

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500


if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=5000)

