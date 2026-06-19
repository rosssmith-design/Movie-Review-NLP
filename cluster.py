from datasets import load_dataset
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


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
print(unsupervised_texts[0][:300])