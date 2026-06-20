from datasets import load_dataset
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.cluster import KMeans


# All english stop words
stop_words = set(ENGLISH_STOP_WORDS)

# Preprocessing function
def preprocess(text):
    # Remove HTML tags (e.g. <br/>)
    text = re.sub(r'<.*?>', ' ', text)
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
cluster_vectorizer = TfidfVectorizer(max_features=20000, max_df=0.3, ngram_range=(1, 1))
X_cluster = cluster_vectorizer.fit_transform(unsupervised_clean)


# Reduce to 100 dimensions 
svd =  TruncatedSVD(n_components=1000, random_state=42)
X_reduced = svd.fit_transform(X_cluster)

def run_and_inspect_clusters(X_reduced, vectorizer, X_original, k=15, label=""):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    print(f"\n--- {label} ---")
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    for cluster_id in range(k):
        # Get all reviews in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_size = cluster_mask.sum()
        
        # Average TF-IDF/count score per word, just for this cluster's reviews
        avg_scores = X_original[cluster_mask].mean(axis=0)
        avg_scores = np.asarray(avg_scores).flatten()
        
        # Top 10 words for this cluster
        top_indices = avg_scores.argsort()[-10:][::-1]
        top_words = feature_names[top_indices]
        
        print(f"Cluster {cluster_id} ({cluster_size} reviews): {', '.join(top_words)}")
    
    return cluster_labels

# Run on TF-IDF version
labels_tfidf = run_and_inspect_clusters(X_reduced, cluster_vectorizer, X_cluster, k=15, label="TF-IDF + SVD (1000 components)")

print(f"Matrix shape: {X_cluster.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {svd.explained_variance_ratio_.sum()*100:.2f}%")

