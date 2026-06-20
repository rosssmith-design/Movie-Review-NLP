from datasets import load_dataset
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.decomposition import TruncatedSVD
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

# Custom specific stop words to help with clustering
custom_stop_words = [
    'movie', 'movies', 'film', 'films', 'just', 'like', 'really',
    'good', 'bad', 'time', 'watch', 'story', 'great', 'best',
    'character', 'characters', 'people', 'think', 'know', 'dont',
    'seen', 'did', 'didnt', 'make', 'way', 'little'
]
# Create a new dedicated vectorizer for clustering
cluster_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), stop_words=custom_stop_words)
X_cluster = cluster_vectorizer.fit_transform(unsupervised_clean)

# Reduce to 100 dimensions 
svd =  TruncatedSVD(n_components=1000, random_state=42)
X_reduced = svd.fit_transform(X_cluster)

# Keywords used for labelling our clusters
genre_keywords = {
    'Horror': ['horror', 'gore', 'scary', 'dead', 'zombie', 'vampire', 'dracula', 'frankenstein'],
    'Comedy': ['funny', 'comedy', 'jokes', 'laugh', 'humor', 'hilarious'],
    'Action': ['action', 'martial', 'fight', 'arts', 'sequences'],
    'Musical': ['music', 'song', 'songs', 'musical'],
    'War': ['war', 'soldiers', 'army', 'battle'],
    'TV Show': ['series', 'episode', 'episodes', 'season', 'tv', 'television'],
    'Book Adaptation': ['book', 'read', 'novel', 'books', 'adaptation'],
    'Video Game': ['game', 'games', 'graphics', 'gameplay', 'video'],
    'Negative Reviews': ['worst', 'terrible', 'awful', 'horrible', 'waste'],
    'Family/Drama': ['family', 'mother', 'father'],
}

# Used for labelling our clusters
def label_cluster(top_words):
    scores = {}
    for genre, keywords in genre_keywords.items():
        overlap = len(set(top_words) & set(keywords))
        if overlap > 0:
            scores[genre] = overlap
    
    if not scores:
        return "Unclassified"
    
    return max(scores, key=scores.get)

# Used for identifying and calculating clusters
def run_and_inspect_clusters(X_reduced, vectorizer, X_original, k=15, label=""):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_reduced)
    
    print(f"\n--- {label} ---")
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_names = {}
    
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

        # Auto-label the cluster based on keyword overlap
        genre_label = label_cluster(list(top_words))
        cluster_names[cluster_id] = genre_label
        
        print(f"Cluster {cluster_id} ({cluster_size} reviews) [{genre_label}]: {', '.join(top_words)}")
    
    return cluster_labels, cluster_names

# Run on TF-IDF version
cluster_labels, cluster_names = run_and_inspect_clusters(X_reduced, cluster_vectorizer, X_cluster, k=15, label="TF-IDF + SVD (1000 components)")

# Reduce to 2D for visualisation
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_reduced)

print("t-SNE complete")
print(f"Matrix shape: {X_cluster.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {svd.explained_variance_ratio_.sum()*100:.2f}%")

# Check for outliers in the t-SNE output
print(f"X range: {X_tsne[:, 0].min():.1f} to {X_tsne[:, 0].max():.1f}")
print(f"Y range: {X_tsne[:, 1].min():.1f} to {X_tsne[:, 1].max():.1f}")

# Eliminate the extreme outliers
plot_mask = X_tsne[:, 0] < 500

# Apply plot mask
X_tsne_plot = X_tsne[plot_mask]
cluster_labels_plot = cluster_labels[plot_mask]
sentiment_pred_plot = sentiment_pred[plot_mask]


plt.figure(figsize=(16, 11))

# Get a distinct colour for each cluster
num_clusters = len(cluster_names)
colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))

# Plot each cluster separately so we can control colour, shape, and build a clean legend
for cluster_id in range(num_clusters):
    cluster_mask = cluster_labels_plot == cluster_id
    genre = cluster_names[cluster_id]
    
    # Split this cluster by sentiment
    positive_mask = cluster_mask & (sentiment_pred_plot == 1)
    negative_mask = cluster_mask & (sentiment_pred_plot == 0)
    
    # Positive reviews - circles
    plt.scatter(X_tsne_plot[positive_mask, 0], X_tsne_plot[positive_mask, 1],
                color=colors[cluster_id], marker='o', s=8, alpha=0.6,
                label=f"{genre} (Cluster {cluster_id})")
    
    # Negative reviews - crosses, same colour, no duplicate legend entry
    plt.scatter(X_tsne_plot[negative_mask, 0], X_tsne_plot[negative_mask, 1],
                color=colors[cluster_id], marker='x', s=8, alpha=0.6)

plt.title('Movie Review Clusters by Genre and Sentiment (t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, markerscale=2)
plt.tight_layout()
plt.savefig('cluster_visualisation.png', dpi=150, bbox_inches='tight')
plt.show()
