import nltk 
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords

nltk.download('movie_reviews')
nltk.download('stopwords')

# Load reviews and labels

reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

stop_words = set(stopwords.words('english'))

print(list(stop_words)[:20])

print(f"Total reviews: {len(reviews)}")
print(f"Categories: {movie_reviews.categories()}")
print(reviews[0][0][:500])
print(f"\nLabel: {reviews[0][1]}")