import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('movie_reviews')
nltk.download('stopwords')

# Load reviews and labels

reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# All english stop words
stop_words = set(stopwords.words('english'))

# Preprocessing function for training
def preprocess(text):
    # Lowercase
    text = text.lower()
    # Removes punctuation and unknown characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Applying preprocessing to all reviews
texts = [preprocess(review[0]) for review in reviews]
labels = [review[1] for review in reviews]

# Vectorizing the preprocessed text
vectorizer = CountVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000, C=0.1)
model.fit(X_train, y_train)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(f"Total preprocessed reviews: {len(texts)}")
print(f"Labels: {labels.count('neg')} negative, {labels.count('pos')} positive")
print(f"Matrix shape: {X.shape}")
print(f"Training Accuracy: {accuracy_score(y_train, train_pred) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(y_test, test_pred) * 100:.2f}%")


