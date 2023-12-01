import sys
import subprocess
from flask import Flask, jsonify, request
import os
import joblib
from nltk.corpus import PlaintextCorpusReader
from sklearn.feature_extraction.text import CountVectorizer

# Install required dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

app = Flask(__name__)

# Path to the directory containing movie reviews data
data_path = 'jude/movie_reviews'

# Load movie reviews dataset from the specified path
corpus = PlaintextCorpusReader(data_path, '.*\.txt')

# Load the trained classifier and vectorizer from the pickle files
classifier = joblib.load('jude/classifier.pkl')
vectorizer = joblib.load('jude/vectorizer.pkl')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    try:
        # Get the comment from the request data
        data = request.get_json()
        comment = data.get('comment', '')

        # Vectorize the comment using the loaded vectorizer
        comment_vectorized = vectorizer.transform([comment])

        # Predict sentiment for the comment
        sentiment = classifier.predict(comment_vectorized)[0]

        # Return the sentiment prediction as JSON
        return jsonify({"sentiment": sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Use a production-ready server (e.g., Gunicorn) for deployment
    app.run(debug=True, port=int(os.getenv("PORT", default=5000)))
