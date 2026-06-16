import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import re

nltk.download('movie_reviews')
nltk.download('stopwords')

# Load reviews and labels

reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# All english stop words
stop_words = set(stopwords.words('english'))

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Removes punctuation and unknown characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)


print(f"Total reviews: {len(reviews)}")
print(f"Categories: {movie_reviews.categories()}")

print(preprocess(reviews[0][0])[:500])

