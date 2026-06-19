from datasets import load_dataset
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.decomposition import TruncatedSVD
import numpy as np


# All english stop words
stop_words = set(ENGLISH_STOP_WORDS)

# Preprocessing function
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Removes punctuation and unknown characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Load unsupervised reviews
dataset = load_dataset('stanfordnlp/imdb')
unsupervised_texts = [item['text'] for item in dataset['unsupervised']]
print(f"Total unsupervised reviews: {len(unsupervised_texts)}")

# Load saved model and vectorizer
vectorizer = joblib.load('sentiment_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')
print(model)

# Preprocess the unsupervised reviews
unsupervised_clean = [preprocess(text) for text in unsupervised_texts]

# Get sentiment predictions using the saved sentiment vectorizer + model
X_sentiment_input = vectorizer.transform(unsupervised_clean)
sentiment_pred = model.predict(X_sentiment_input)

print(f"Sentiment breakdown: {sum(sentiment_pred)} positive, {len(sentiment_pred) - sum(sentiment_pred)} negative")

# Create a new dedicated vectorizer for clustering
cluster_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 1))
X_cluster = cluster_vectorizer.fit_transform(unsupervised_clean)


# Reduce to 100 dimensions 
svd =  TruncatedSVD(n_components=1000, random_state=42)
X_reduced = svd.fit_transform(X_cluster)



print(f"Matrix shape: {X_cluster.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {svd.explained_variance_ratio_.sum()*100:.2f}%")

