import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset
import joblib

dataset = load_dataset('stanfordnlp/imdb')

print(dataset)



# Load reviews and labels

train_texts = [item['text'] for item in dataset['train']]
train_labels = [item['label'] for item in dataset['train']]
test_texts = [item['text'] for item in dataset['test']]
test_labels = [item['label'] for item in dataset['test']]

# All english stop words
stop_words = set(ENGLISH_STOP_WORDS)

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
train_texts_clean = [preprocess(text) for text in train_texts]
test_texts_clean = [preprocess(text) for text in test_texts]

# Vectorizing the preprocessed text
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts_clean)
X_test = vectorizer.transform(test_texts_clean)


# Train model
model = LogisticRegression(max_iter=1000, C=2.0)
model.fit(X_train, train_labels)

# Evaluate
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)



print(f"Training Accuracy: {accuracy_score(train_labels, train_pred) * 100:.2f}%")
print(f"Test Accuracy: {accuracy_score(test_labels, test_pred) * 100:.2f}%")

joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'sentiment_vectorizer.pkl')

print('Model and vectorizer saved successfully')