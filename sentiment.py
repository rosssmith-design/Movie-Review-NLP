import nltk 
from nltk.corpus import movie_reviews

nltk.download('movie_reviews')

# Load reviews and labels

reviews = [(movie_reviews.raw(fileid), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

print(f"Total reviews: {len(reviews)}")
print(f"Categories: {movie_reviews.categories()}")
print(reviews[0][0][:500])
print(f"\nLabel: {reviews[0][1]}")