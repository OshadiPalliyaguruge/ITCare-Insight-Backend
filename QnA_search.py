# from flask import Flask, request, jsonify
# import pickle
# import glob
# import os
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from collections import defaultdict
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)

# # Initialize Flask app

# # Load NLTK resources
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

# # Initialize stemmer and stopwords
# stemmer = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # Function to preprocess text
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     tokens = word_tokenize(text)
#     tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
#     return tokens

# # Function to get the latest preprocessed dataset
# def get_latest_preprocessed_file():
#     files = glob.glob('Models\preprocessed_data.pkl')
#     if not files:
#         raise FileNotFoundError("No preprocessed data files found.")
#     latest_file = max(files, key=os.path.getctime)
#     return latest_file

# # Search function
# def search(user_question, top_n=3, exact_match_threshold=0.8):
#     preprocessed_file = get_latest_preprocessed_file()
    
#     with open(preprocessed_file, 'rb') as f:
#         data = pickle.load(f)
#         dataset = data['dataset']
#         inverted_index = data['inverted_index']

#     user_tokens = preprocess_text(user_question)
    
#     matches = defaultdict(int)
#     for token in user_tokens:
#         if token in inverted_index:
#             for doc_id, summary, resolution in inverted_index[token]:
#                 matches[(doc_id, summary, resolution)] += 1

#     sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)
    
#     filtered_results = []
#     for (doc_id, summary, resolution), count in sorted_matches:
#         match_percentage = count / len(user_tokens)
#         if match_percentage >= exact_match_threshold:
#             filtered_results.append({"question": summary, "answer": resolution})

#     print("DEBUG - Final Processed Results:", filtered_results)  # Debugging
#     return filtered_results[:top_n] if filtered_results else [{"question": "No relevant question found.", "answer": "No relevant answer found."}]

# # API Route to search
# @app.route('/api/search', methods=['POST'])
# def search_api():
#     data = request.json
#     user_query = data.get('query', '')

#     if not user_query:
#         return jsonify({'error': 'No query provided'}), 400

#     results = search(user_query, exact_match_threshold=0.8)
    
#     print("DEBUG - Actual Search Results:", results)  # Add this for debugging
#     return jsonify({'results': results})


# # Run Flask app
# if __name__ == '__main__':
#     app.run(port=5002, debug=True)



from flask import Flask, request, jsonify
import pickle
import glob
import os
import re
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize NLP tools
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
embedder = None  # Will be initialized when loading models

def load_models():
    """Load the preprocessed data and models"""
    global embedder
    preprocessed_file = get_latest_preprocessed_file()
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)
    embedder = data['embedder']

def preprocess_text(text):
    """
    Enhanced text preprocessing with lemmatization
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

def get_latest_preprocessed_file():
    """
    Find the most recent preprocessed data file
    """
    files = glob.glob('Models/preprocessed_data_*.pkl')
    if not files:
        raise FileNotFoundError("No preprocessed data files found.")
    return max(files, key=os.path.getctime)

@lru_cache(maxsize=1000)
def cached_encode(text):
    """Cache embeddings for frequently asked questions"""
    return embedder.encode([text], convert_to_tensor=True)

def find_exact_matches(user_question, dataset):
    """Find all exact matches in dataset (case-insensitive)"""
    exact_matches = []
    user_question_lower = user_question.lower().strip()
    
    for _, row in dataset.iterrows():
        if user_question_lower == row['Summary'].lower().strip():
            exact_matches.append((row['Summary'], row['Resolution']))
    
    return exact_matches

def search(user_question, top_n=5, min_similarity=0.3):
    """
    Enhanced hybrid search with exact match priority
    Returns list of dicts with question/answer pairs
    """
    preprocessed_file = get_latest_preprocessed_file()
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)
    dataset, inverted_index, vectorizer, tfidf_matrix, _, embeddings = (
        data['dataset'], data['inverted_index'], data['vectorizer'], 
        data['tfidf_matrix'], data['embedder'], data['embeddings']
    )

    # 1. Check for exact matches first
    exact_matches = find_exact_matches(user_question, dataset)
    if exact_matches:
        return [{"question": q, "answer": a} for q, a in exact_matches[:top_n]]

    # 2. Proceed with semantic search if no exact matches
    user_query = preprocess_text(user_question)
    if not user_query:
        return [{"question": "No relevant question found.", "answer": "No relevant answer found."}]

    # Hybrid search components
    user_tokens = user_query.split()
    combined_scores = defaultdict(float)
    
    # Token matches
    token_matches = defaultdict(int)
    for token in user_tokens:
        if token in inverted_index:
            for idx, _, _ in inverted_index[token]:
                token_matches[idx] += 1
    max_token_score = max(token_matches.values(), default=1)
    
    # TF-IDF similarity
    user_tfidf = vectorizer.transform([user_query])
    tfidf_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Semantic similarity
    user_embedding = cached_encode(user_query)
    semantic_similarities = cosine_similarity(user_embedding.cpu().numpy(), 
                                           embeddings.cpu().numpy()).flatten()
    
    # Combine scores with weights
    for idx in range(len(dataset)):
        token_score = token_matches.get(idx, 0) / max_token_score
        tfidf_score = tfidf_similarities[idx]
        semantic_score = semantic_similarities[idx]
        combined_scores[idx] = (token_score * 1.0) + (tfidf_score * 0.7) + (semantic_score * 0.5)
    
    # Sort and filter results
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, score in sorted_results if score >= min_similarity][:top_n]
    
    # Get unique results
    seen = set()
    results = []
    for idx in top_indices:
        question = dataset.iloc[idx]['Summary']
        answer = dataset.iloc[idx]['Resolution']
        if (question, answer) not in seen:
            seen.add((question, answer))
            results.append({"question": question, "answer": answer})
    
    return results if results else [{"question": "No relevant question found.", "answer": "No relevant answer found."}]

@app.route('/api/search', methods=['POST'])
def search_api():
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    results = search(user_query)
    print("DEBUG - Search Results:", results)  # For debugging
    return jsonify({'results': results})

if __name__ == '__main__':
    load_models()
    app.run(port=5002, debug=True)