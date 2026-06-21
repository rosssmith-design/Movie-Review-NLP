# Movie Review NLP: Sentiment Analysis & Genre Clustering

A two-phase NLP project exploring both **supervised** and **unsupervised** machine learning on 100,000 IMDB movie reviews — first training a sentiment classifier, then using that same dataset to discover genre-like groupings with zero labels.

## Project Overview

This project started as a simple sentiment analysis task (the "hello world" of NLP) and grew into something more ambitious: using the model trained in Phase 1 alongside an entirely unsupervised clustering pipeline in Phase 2 to explore whether genre information can emerge naturally from review text alone, with no genre labels ever provided.

**Phase 1 — Sentiment Analysis (Supervised)**
Classify a review as positive or negative based on its text.

**Phase 2 — Genre Clustering (Unsupervised)**
Group 50,000 unlabelled reviews by underlying theme/genre using K-Means, then label each cluster automatically and visualise the results — while reusing the Phase 1 model to overlay sentiment onto the discovered clusters.

---

## Phase 1: Sentiment Analysis

### Getting Started

Development began on NLTK's built-in `movie_reviews` corpus (2,000 reviews) to learn the fundamentals before scaling up to the full 50,000-review IMDB dataset via HuggingFace's `datasets` library.

### Text Preprocessing

Raw review text was cleaned through a custom `preprocess()` function:
- HTML tag removal (a bug was found and fixed mid-project — `<br />` tags were leaking through as the word "br" in the vocabulary, since stripping punctuation left the letters behind. Fixing this slightly improved sentiment accuracy and meaningfully cleaned up cluster results later in Phase 2)
- Lowercasing
- Punctuation and number stripping
- Stop word removal

### Vectorisation Journey

| Approach | Test Accuracy |
|---|---|
| Bag of Words (CountVectorizer), 2000 features | 80.50% |
| Bag of Words, 5000 features | 83.00% |
| Bag of Words, 10000 features | 80.75% (overfitting noise) |
| TF-IDF, 5000 features | 81.25% |
| Bag of Words + bigrams, 20000 features | 87.52% |
| **TF-IDF + bigrams, `C=2`** | **88.33% (final)** |

Character-level n-grams (3-5 character sequences) were also tested as an alternative to word-level tokens, but underperformed (86.24%) and were significantly slower, so this was dropped in favour of word-level bigrams.

### Model Selection

Two models were compared:
- **Multinomial Naive Bayes** — a strong, fast baseline well suited to word-count style features
- **Logistic Regression** — ultimately chosen for its compatibility with TF-IDF and tunable regularisation

Regularisation strength (`C`) was tuned manually, balancing training/test accuracy gap against raw test performance — `C=2` was found to give the best test accuracy (88.33%) without the overfitting seen at higher values.

### Final Configuration

- `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`
- `LogisticRegression(max_iter=1000, C=2)`
- Trained on the full 25,000-review IMDB training split, evaluated on the held-out 25,000-review test split
- **Final test accuracy: 88.33%**

The trained model and vectorizer are saved to disk using `joblib` (`sentiment_model.pkl`, `sentiment_vectorizer.pkl`) so they can be reused directly in Phase 2 without retraining.

---

## Phase 2: Genre Clustering

### The Idea

Using the 50,000 unlabelled "unsupervised" reviews from the IMDB dataset (a separate, distinct split from the labelled train/test data), the goal was to see whether unsupervised clustering could surface genre-like groupings purely from review text — with no genre labels anywhere in the dataset to guide it.

### Dimensionality Reduction: A Debugging Story

The original plan was to vectorise the reviews and reduce dimensionality with **PCA** before clustering. This immediately ran into a wall:

| Attempt | Variance Explained (100 components) |
|---|---|
| Bag of Words vectoriser + PCA | 33.08% |
| Saved TF-IDF (sentiment) vectoriser + PCA | 9.03% |
| Dedicated TF-IDF vectoriser + PCA | 9.75% |

A lot of debugging went into understanding this drop. Several hypotheses were tested and ruled out in turn:
- **Bigram pollution** — ruled out (vocabulary was only ~35% bigrams, not enough to explain the gap)
- **Sparse data breaking standard PCA's centering step** — tested by switching to `TruncatedSVD` (which avoids centering and works natively on sparse matrices). Result was nearly identical (9.69%), ruling this out too
- **TF-IDF's normalisation removing "fake" variance driven by review length/verbosity** — the most likely explanation. Raw word counts (Bag of Words) let long, verbose reviews dominate the variance calculation in a way that's about writing style, not topic. TF-IDF's normalisation removes this, but spreads genuine topical signal much more thinly across components as a result

Increasing `n_components` to 1000 raised variance explained to 37.24%, confirming the signal was real but genuinely spread across many dimensions, with no small set of dominant "super-topics" in such a diverse 50,000-review dataset.

**Conclusion:** variance explained is a useful diagnostic but not the actual goal — the real test of whether dimensionality reduction worked is whether the resulting K-Means clusters are interpretable, not how much raw variance was retained.

### Connecting Phase 1 and Phase 2

An early design question was whether to reuse the Phase 1 sentiment vectorizer for clustering too, for the sake of a single unified pipeline. Testing showed this hurt cluster quality — a vectorizer's vocabulary, selected specifically to separate positive from negative sentiment, is not the same vocabulary that best separates reviews by topic/genre. Words highly predictive of sentiment ("masterpiece", "boring") tend to appear across all genres equally, while genre-distinguishing words ("zombie", "courtroom", "wedding") may not be sentiment-predictive enough to make the cut.

**Final approach:** a dedicated TF-IDF vectorizer is fit specifically for clustering. The saved Phase 1 model and vectorizer are still used in Phase 2, but only for their original purpose — generating a sentiment prediction for each of the 50,000 unsupervised reviews, which is then overlaid onto the discovered clusters.

### Removing Generic Noise Words

Early clustering attempts (K=5, K=10) found a handful of genuinely interpretable clusters (Horror, TV Shows) but most others were dominated by generic, genre-irrelevant words like "movie", "film", "good", "just", "time" appearing as "top words" in nearly every cluster.

Two techniques were tested to address this:

1. **`max_df` filtering** (excluding words appearing in too high a percentage of documents) — effective, but at aggressive settings (`max_df=0.3`) it overcorrected, causing K-Means to latch onto narrow franchise/named-entity clusters (e.g. distinct clusters for James Bond and Batman reviews specifically) rather than broader genres
2. **A custom stop word list**, built empirically from words observed to repeat across multiple unrelated clusters (`movie`, `film`, `just`, `like`, `good`, `time`, `story`, etc.) — this gave more surgical control and was the approach ultimately used

### Choosing K

K was increased incrementally, inspecting cluster interpretability at each step rather than relying purely on metrics like the elbow method:

- **K=5** — only Horror and TV Shows emerged as clearly distinct
- **K=10** — Action, Book Adaptations, and a Zombie/Horror sub-cluster emerged; a tiny, hyper-specific Chuck Norris cluster (91 reviews) also appeared, an interesting curiosity showing K-Means' sensitivity to very specific patterns
- **K=15 (final)** — Comedy, Musicals, War, and Video Games emerged, alongside non-genre but interesting clusters: a Woody Allen-specific cluster and a "negative reviews" cluster that grouped by sentiment rather than genre

### Final Genre Clusters (K=15)

| Genre | Example Top Words |
|---|---|
| Horror | horror, gore, scary, dead, effects, budget |
| Comedy | funny, comedy, jokes, laugh, humor, hilarious |
| Action | action, scenes, martial, fight, arts |
| Musical | music, song, songs, musical |
| Book Adaptation | book, read, novel, books, adaptation |
| TV Show | series, episode, episodes, season, television |
| Video Game | game, games, graphics, gameplay |
| Family/Drama | family, mother, father, love, life |
| Negative Reviews | worst, terrible, awful, horrible, waste |

Several clusters remained "Unclassified" — generic commentary without strong distinguishing vocabulary, or non-genre patterns (a performance/acting-critique-focused cluster, a tiny 58-review noise cluster). These are discussed openly below rather than forced into artificial categories.

### Automatic Cluster Labelling

Each cluster's top 10 TF-IDF words are matched against a hand-built dictionary of genre keywords, and labelled with whichever genre has the highest keyword overlap (falling back to "Unclassified" if there's no match).

**Why rule-based instead of an LLM?** Using the Claude API to generate labels from each cluster's top words was considered and would have been a natural fit. However, this required a separate, billed Anthropic Console account (API access is not included with a Claude Pro subscription) and an explicit decision was made to keep the project entirely free and self-contained. This is a deliberate trade-off: the rule-based system is fast, free, and fully reproducible, but is less nuanced than an LLM would be — for example, it cannot identify that Cluster 12 represents Woody Allen films specifically, only that it overlaps most with "Comedy" (which is, incidentally, not an unreasonable label given his body of work).

### Visualisation

Two complementary plots were produced using **t-SNE** to compress the clustering data down to 2D:

1. **`cluster_visualisation.png`** — coloured by genre cluster, shaped by sentiment (circle = positive, cross = negative). Genres with distinctive, consistent vocabulary (Horror, Action, Book Adaptations, Video Games) formed tight, visually separated regions; vaguer/broader clusters overlapped more, mirroring the word-list findings
2. **`sentiment_visualisation.png`** — the same 2D layout, recoloured purely by sentiment (green = positive, red = negative) for clarity, since sentiment was difficult to read in the combined plot. Notably, the genre Cluster 13 ("Negative Reviews") appears as a visibly dense red region here too — an independent cross-check, given the genre clustering and sentiment model were never connected during training, that the two pipelines agree

One extreme t-SNE outlier point was excluded from the plots (not the underlying data) after it was found to be badly distorting the axis scale.

---

## Key Findings & Limitations

- **Genuine genre signal exists in review text alone** — Horror, Comedy, Action, Musicals, and others all emerged from raw text with no labels, validating the core hypothesis behind this project
- **Sentiment and genre are competing signals** — some clusters (e.g. "Negative Reviews") group by tone rather than topic, suggesting that for some reviews, *how* something is said dominates *what* it's about
- **K-Means is sensitive to very specific patterns**, occasionally surfacing director- or franchise-level clusters (Woody Allen, Chuck Norris, James Bond, Batman) rather than broad genres, depending on vocabulary filtering settings
- **No ground truth genre labels exist** for this dataset, so all genre interpretation is qualitative (by inspection of top words and visual clustering), not a measured accuracy score
- **Rule-based labelling is inherently limited** — it can only recognise genres it was explicitly told to look for, and can't infer specific or novel categories (like director-based clusters) the way an LLM-based approach could
- **Hardware constraints shaped multiple decisions throughout the project.** This was deliberately built to run entirely on a personal laptop with no GPU and no paid services, which meant several genuinely promising directions were scoped out or scaled back rather than pursued fully:
  - Word2Vec and spaCy's pretrained embeddings were both blocked by a missing C++ compiler toolchain on Windows, and installing it was judged not worth the time cost for an incremental upgrade over TF-IDF
  - `n_components` for SVD was capped at 1000 rather than pushed further, partly due to the increasing memory and runtime cost of `t-SNE` and `KMeans` on a CPU at 50,000 reviews
  - K was explored up to 15 by hand rather than via a more exhaustive elbow/silhouette search across a much wider range, which would have meant re-running the full clustering pipeline (TF-IDF → SVD → K-Means → relabelling) many more times
  - LLM-based cluster labelling via the Claude API was scoped out specifically to keep the project free, rather than for a technical reason
  - BERT/transformer-based approaches, which would likely outperform TF-IDF significantly for both phases, were considered out of reach without GPU access entirely

  None of these were hard blockers — they're documented here as honest scope decisions, and several are listed under Future Improvements as the natural next step if GPU access (e.g. via a university HPC cluster) becomes available.

---

## Future Improvements

- **Word2Vec or BERT embeddings** in place of TF-IDF, to capture word meaning and context rather than just frequency (attempted during development but blocked by a Windows C++ Build Tools dependency issue with Gensim/spaCy — noted as a setup limitation rather than a methodological one)
- **LLM-based cluster labelling** via the Claude API, for more nuanced, context-aware genre names than the current keyword-matching approach allows
- **A genre-labelled dataset** (e.g. the CMU Movie Summary Corpus) to properly validate cluster quality against ground truth, rather than by inspection alone
- **Walk-forward or cross-validated evaluation** for more robust accuracy reporting in Phase 1

---

## How to Run

Install dependencies:
```bash
pip install scikit-learn pandas numpy matplotlib datasets joblib
```

Run Phase 1 first (trains and saves the sentiment model):
```bash
python sentiment.py
```

Then run Phase 2 (loads the saved model, performs clustering and visualisation):
```bash
python cluster.py
```