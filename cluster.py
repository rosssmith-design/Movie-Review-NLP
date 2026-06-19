from datasets import load_dataset
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib
from sklearn.decomposition import PCA



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

# Load saved model and vectorizer
vectorizer = joblib.load('sentiment_vectorizer.pkl')
model = joblib.load('sentiment_model.pkl')

# Preprocess the unsupervised reviews
unsupervised_clean = [preprocess(text) for text in unsupervised_texts]

# Vectorize unsupervised dataset
X_unsupervised = vectorizer.transform(unsupervised_clean)

# Reduce to 100 dimensions 
pca = PCA(n_components=500, random_state=42)
X_reduced = pca.fit_transform(X_unsupervised.toarray())


print(f"Total unsupervised reviews: {len(unsupervised_texts)}")
print(f"Matrix shape: {X_unsupervised.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")