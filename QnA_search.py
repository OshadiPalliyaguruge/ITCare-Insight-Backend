from flask import Flask, request, jsonify
import pickle
import glob
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Initialize Flask app

# Load NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

# Function to get the latest preprocessed dataset
def get_latest_preprocessed_file():
    files = glob.glob('Models\preprocessed_data.pkl')
    if not files:
        raise FileNotFoundError("No preprocessed data files found.")
    latest_file = max(files, key=os.path.getctime)
    return latest_file

# Search function
def search(user_question, top_n=3, exact_match_threshold=0.8):
    preprocessed_file = get_latest_preprocessed_file()
    
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)
        dataset = data['dataset']
        inverted_index = data['inverted_index']

    user_tokens = preprocess_text(user_question)
    
    matches = defaultdict(int)
    for token in user_tokens:
        if token in inverted_index:
            for doc_id, summary, resolution in inverted_index[token]:
                matches[(doc_id, summary, resolution)] += 1

    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    
    filtered_results = []
    for (doc_id, summary, resolution), count in sorted_matches:
        match_percentage = count / len(user_tokens)
        if match_percentage >= exact_match_threshold:
            filtered_results.append({"question": summary, "answer": resolution})

    print("DEBUG - Final Processed Results:", filtered_results)  # Debugging
    return filtered_results[:top_n] if filtered_results else [{"question": "No relevant question found.", "answer": "No relevant answer found."}]

# API Route to search
@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    results = search(user_query, exact_match_threshold=0.8)
    
    print("DEBUG - Actual Search Results:", results)  # Add this for debugging
    return jsonify({'results': results})


# Run Flask app
if __name__ == '__main__':
    app.run(port=5002, debug=True)
