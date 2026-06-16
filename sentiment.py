import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

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

vectorizer = CountVectorizer(max_features=2000)
X = vectorizer.fit_transform(texts)


print(f"Total preprocessed reviews: {len(texts)}")
print(f"Labels: {labels.count('neg')} negative, {labels.count('pos')} positive")
print(f"Matrix shape: {X.shape}")


